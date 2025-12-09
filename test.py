```python
import os
import datetime
import torch
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from kv_policy.config import CONFIG
from kv_policy.model_utils import load_model_and_tokenizer
from kv_policy.eval_kv import (
    eval_full_offline,
    eval_stream_sink_window_baseline,
    eval_stream_remote_policy,
)
from kv_policy.policy_net import RetentionPolicyMLP


# ======================================================
#  Global configuration for evaluation sweep
# ======================================================
MAX_CHARS = 8000          # truncate each validation sample to this many chars
MAX_LEN = 2048            # max sequence length passed into the model
NUM_SAMPLES = 100         # number of validation samples to evaluate

# cache sizes to sweep over
CACHE_SIZES = [32, 64, 128, 256, 384, 512]

# file paths for logging and aggregated results
CSV_FILE = f"final_sweep_results{MAX_CHARS}.csv"
LOG_FILE = f"eval_log{MAX_CHARS}.txt"
CHECKPOINT_DIR = r"checkpoints\0.5b"  # you can select between 0.5b and 7b


# ======================================================
#  Sink size: ~10% of cache
# ======================================================
def get_sink_size(max_cache: int) -> int:
    """Return sink size as ~10% of cache, clipped to [4, 32]."""
    s = int(max_cache * 0.10)
    return max(4, min(s, 32))


# ======================================================
#  B_remote as a fraction of cache (scheme B)
# ======================================================
def get_B_remote(max_cache: int) -> int:
    """
    Return number of remote slots:
        B_remote = cache_size / 4, clipped to [4, 32].
    """
    B = max_cache // 4
    return max(4, min(B, 32))


# ======================================================
#  Logging utility
# ======================================================
def log_info(msg: str) -> None:
    """Print a timestamped message and also append it to a log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")


# ======================================================
#  Policy loading
# ======================================================
def load_policy(model, ckpt_path: str, device):
    """
    Load a RetentionPolicyMLP checkpoint if it exists.
    Returns:
        policy (nn.Module) or None if checkpoint is missing or invalid.
    """
    if not os.path.exists(ckpt_path):
        log_info(f"[WARN] Missing checkpoint: {ckpt_path}")
        return None
    try:
        d_in = model.config.hidden_size + 2  # hidden + position/age features
        policy = RetentionPolicyMLP(d_in, d_hidden=256)
        state = torch.load(ckpt_path, map_location="cpu")
        policy.load_state_dict(state)
        policy.to(device).eval()
        log_info(f"[INFO] Loaded policy from {ckpt_path}")
        return policy
    except Exception as e:
        log_info(f"[ERROR] Failed to load {ckpt_path}: {e}")
        return None


# ======================================================
#  Main evaluation script
# ======================================================
def main():
    """
    Run an evaluation sweep over different KV cache sizes.

    For each cache size, we compute perplexities for:
      - Offline full inference (no KV pruning, single pass).
      - Sink + sliding-window baseline (SW).
      - BC-trained policy (if checkpoint exists).
      - DPO-trained policy (if checkpoint exists).

    Results are appended to a CSV file and a text log for analysis.
    """
    # determine which cache sizes are already finished (if resuming)
    finished = []
    if os.path.exists(CSV_FILE):
        try:
            df_exist = pd.read_csv(CSV_FILE)
            finished = df_exist["Cache Size"].tolist()
        except Exception:
            pass

    tasks = [c for c in CACHE_SIZES if c not in finished]
    if not tasks:
        log_info("All cache-size tasks for this MAX_CHARS setting are already finished.")
        return

    # load student model and tokenizer
    student_model, tokenizer, device, dtype = load_model_and_tokenizer(
        CONFIG["student_model_name"]
    )
    student_model.eval()
    log_info(f"Loaded student model on device={device}, dtype={dtype}")

    # load validation data
    log_info(f"Loading validation data ({NUM_SAMPLES} samples)...")
    ds_val = load_dataset("emozilla/pg19", split=f"validation[:{NUM_SAMPLES}]")
    texts = [ex["text"][:MAX_CHARS] for ex in ds_val]

    # compute full offline perplexity (reference)
    log_info("Running offline full inference (no KV pruning)...")
    full_ppls = []
    for text in tqdm(texts, desc="Full Offline"):
        _, ppl = eval_full_offline(
            student_model,
            tokenizer,
            text,
            max_len=MAX_LEN,
            device=device,
        )
        full_ppls.append(ppl)
    avg_full = sum(full_ppls) / len(full_ppls)
    log_info(f"Offline Full PPL = {avg_full:.2f}")

    # sweep over cache sizes
    for max_cache in tasks:
        sink = get_sink_size(max_cache)
        B_remote = get_B_remote(max_cache)

        # policy uses sink + window_policy + B_remote = max_cache
        window_policy = max_cache - sink - B_remote
        # SW baseline uses sink + window_sw = max_cache (no remote)
        window_sw = max_cache - sink

        log_info("=" * 60)
        log_info(
            f"Cache={max_cache} | Sink={sink} | B_remote={B_remote} | "
            f"Window_policy={window_policy} | Window_SW={window_sw}"
        )
        log_info("=" * 60)

        if window_policy <= 0 or window_sw <= 0:
            log_info("Window size <= 0 for this configuration. Skipping...")
            continue

        # ensure checkpoint directory exists (for loading we just check paths)
        bc_ckpt = os.path.join(CHECKPOINT_DIR, f"remote_policy_bc_cache{max_cache}.pt")
        dpo_ckpt = os.path.join(CHECKPOINT_DIR, f"remote_policy_dpo_cache{max_cache}.pt")

        # load BC and DPO policies (if available)
        bc_policy = load_policy(student_model, bc_ckpt, device)
        dpo_policy = load_policy(student_model, dpo_ckpt, device)

        # sliding-window baseline (uses full max_cache: sink + window_sw)
        sw_ppls = []
        for text in tqdm(texts, desc=f"C{max_cache}-SW"):
            _, ppl = eval_stream_sink_window_baseline(
                student_model,
                tokenizer,
                text,
                max_len=MAX_LEN,
                max_cache=max_cache,
                sink_size=sink,
                window_size=window_sw,
                device=device,
            )
            sw_ppls.append(ppl)
        avg_sw = sum(sw_ppls) / len(sw_ppls)
        log_info(f"SW baseline PPL = {avg_sw:.2f}")

        # BC policy evaluation
        if bc_policy is not None:
            bc_ppls = []
            for text in tqdm(texts, desc=f"C{max_cache}-BC"):
                _, ppl = eval_stream_remote_policy(
                    student_model,
                    tokenizer,
                    bc_policy,
                    text,
                    max_len=MAX_LEN,
                    max_cache=max_cache,
                    sink_size=sink,
                    window_size=window_policy,
                    B_remote=B_remote,
                    device=device,
                )
                bc_ppls.append(ppl)
            avg_bc = sum(bc_ppls) / len(bc_ppls)
        else:
            avg_bc = float("nan")
        log_info(f"BC policy PPL  = {avg_bc:.2f}")

        # DPO policy evaluation
        if dpo_policy is not None:
            dpo_ppls = []
            for text in tqdm(texts, desc=f"C{max_cache}-DPO"):
                _, ppl = eval_stream_remote_policy(
                    student_model,
                    tokenizer,
                    dpo_policy,
                    text,
                    max_len=MAX_LEN,
                    max_cache=max_cache,
                    sink_size=sink,
                    window_size=window_policy,
                    B_remote=B_remote,
                    device=device,
                )
                dpo_ppls.append(ppl)
            avg_dpo = sum(dpo_ppls) / len(dpo_ppls)
        else:
            avg_dpo = float("nan")
        log_info(f"DPO policy PPL = {avg_dpo:.2f}")

        # append row to CSV
        row = {
            "Cache Size": max_cache,
            "Sink Size": sink,
            "B_remote": B_remote,
            "Window Size": window_policy,  # window used by policy methods
            "Full (Offline)": avg_full,
            "Baseline (SW)": avg_sw,
            "Policy (BC)": avg_bc,
            "Policy (DPO)": avg_dpo,
        }

        df_row = pd.DataFrame([row])
        # if CSV did not exist before, write header; otherwise append without header
        write_header = not os.path.exists(CSV_FILE)
        df_row.to_csv(CSV_FILE, mode="a", index=False, header=write_header)

        log_info(f"Saved results for cache size = {max_cache}")

    log_info("Evaluation sweep completed.")


if __name__ == "__main__":
    main()
```
