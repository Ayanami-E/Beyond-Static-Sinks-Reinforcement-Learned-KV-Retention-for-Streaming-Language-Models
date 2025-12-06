import os
import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm

from kv_policy.config import CONFIG
from kv_policy.model_utils import load_model_and_tokenizer
from kv_policy.eval_kv import (
    eval_stream_sink_window_baseline,
    eval_stream_remote_policy,
)
from kv_policy.policy_net import RetentionPolicyMLP


# ======================================================
#  Evaluation and visualization settings
# ======================================================
MAX_CHARS = 8000           # truncate validation sample
MAX_LEN = 1638             # max sequence length for model input
TARGET_CACHE = 64          # cache size to visualize
FIG_DIR = "figs_attention"
os.makedirs(FIG_DIR, exist_ok=True)

# where policy checkpoints are stored, e.g. checkpoints/0.5b
CHECKPOINT_DIR = os.path.join("checkpoints", "0.5b")


# ======================================================
#  Sink / Remote selection rules
# ======================================================
def get_sink_size(max_cache: int) -> int:
    """Compute sink size as ~10% of cache, clipped to [4, 32]."""
    s = int(max_cache * 0.10)
    return max(4, min(s, 32))


def get_B_remote(max_cache: int) -> int:
    """Number of remote tokens allowed: cache_size / 4, clipped to [4, 32]."""
    B = max_cache // 4
    return max(4, min(B, 32))


# ======================================================
#  Logging helper
# ======================================================
def log_info(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


# ======================================================
#  Policy loading
# ======================================================
def load_policy(model, ckpt_path, device):
    """Load a policy checkpoint if available."""
    if not os.path.exists(ckpt_path):
        log_info(f"Missing checkpoint: {ckpt_path}")
        return None
    try:
        d_in = model.config.hidden_size + 2
        policy = RetentionPolicyMLP(d_in, d_hidden=256)
        state = torch.load(ckpt_path, map_location="cpu")
        policy.load_state_dict(state)
        policy.to(device).eval()
        return policy
    except Exception as e:
        log_info(f"Error loading checkpoint {ckpt_path}: {e}")
        return None


# ======================================================
#  Visualization
# ======================================================
def plot_trace(
    trace: np.ndarray,
    title: str,
    max_t: int = 512,
    max_tokens: int = 512,
    save_path: str | None = None,
):
    """
    Visualize a binary retention map.

    trace[t, i] = whether token i is in cache after processing step t.
    """
    T, L = trace.shape
    sub = trace[:min(T, max_t), :min(L, max_tokens)]

    plt.figure(figsize=(6, 5))
    plt.imshow(sub.T, aspect="auto", origin="lower")
    plt.xlabel("time step")
    plt.ylabel("token index")
    plt.title(title)
    plt.colorbar(label="kept (1) / dropped (0)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        log_info(f"Saved figure to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_diff(
    trace_ref: np.ndarray,
    trace_other: np.ndarray,
    title: str,
    max_t: int = 512,
    max_tokens: int = 512,
    save_path: str | None = None,
):
    """
    Visualize retention difference between two methods:

       diff = trace_other - trace_ref

       -1 : only kept by reference
        0 : same decision
       +1 : only kept by 'other'
    """
    assert trace_ref.shape == trace_other.shape
    diff = trace_other.astype(int) - trace_ref.astype(int)

    T, L = diff.shape
    sub = diff[:min(T, max_t), :min(L, max_tokens)]

    plt.figure(figsize=(6, 5))
    plt.imshow(sub.T, aspect="auto", origin="lower", cmap="bwr", vmin=-1, vmax=1)
    plt.xlabel("time step")
    plt.ylabel("token index")
    plt.title(title)
    plt.colorbar(label="-1: ref only, 0: same, +1: other only")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        log_info(f"Saved figure to {save_path}")
    else:
        plt.show()
    plt.close()


# ======================================================
#  Main: run one sample and visualize retention maps
# ======================================================
def main():
    log_info("Loading model...")
    student_model, tokenizer, device, dtype = load_model_and_tokenizer(
        CONFIG["student_model_name"]
    )
    student_model.eval()

    log_info("Loading validation sample...")
    ds_val = load_dataset("emozilla/pg19", split="validation[:1]")
    text = ds_val[0]["text"][:MAX_CHARS]

    max_cache = TARGET_CACHE
    sink = get_sink_size(max_cache)
    B_remote = get_B_remote(max_cache)
    window = max_cache - sink - B_remote

    log_info(
        f"Configuration: cache={max_cache}, sink={sink}, "
        f"B_remote={B_remote}, window={window}"
    )

    # Build checkpoint paths under CHECKPOINT_DIR
    bc_ckpt = os.path.join(CHECKPOINT_DIR, f"remote_policy_bc_cache{max_cache}.pt")
    dpo_ckpt = os.path.join(CHECKPOINT_DIR, f"remote_policy_dpo_cache{max_cache}.pt")

    # Load BC and DPO policies
    bc_policy = load_policy(student_model, bc_ckpt, device)
    dpo_policy = load_policy(student_model, dpo_ckpt, device)

    # ------------------ Sliding Window baseline ------------------
    log_info("Running sliding-window baseline with trace...")
    loss_sw, ppl_sw, trace_sw = eval_stream_sink_window_baseline(
        student_model,
        tokenizer,
        text,
        max_len=MAX_LEN,
        max_cache=max_cache,
        sink_size=sink,
        window_size=window,
        device=device,
        record_trace=True,
    )
    log_info(f"SW PPL = {ppl_sw:.2f}")

    # ------------------ BC Policy ------------------
    if bc_policy:
        log_info("Running BC policy with trace...")
        loss_bc, ppl_bc, trace_bc = eval_stream_remote_policy(
            student_model,
            tokenizer,
            bc_policy,
            text,
            max_len=MAX_LEN,
            max_cache=max_cache,
            sink_size=sink,
            window_size=window,
            B_remote=B_remote,
            device=device,
            record_trace=True,
        )
        log_info(f"BC PPL = {ppl_bc:.2f}")
    else:
        trace_bc = None
        log_info("BC policy checkpoint not found.")

    # ------------------ DPO Policy ------------------
    if dpo_policy:
        log_info("Running DPO policy with trace...")
        loss_dpo, ppl_dpo, trace_dpo = eval_stream_remote_policy(
            student_model,
            tokenizer,
            dpo_policy,
            text,
            max_len=MAX_LEN,
            max_cache=max_cache,
            sink_size=sink,
            window_size=window,
            B_remote=B_remote,
            device=device,
            record_trace=True,
        )
        log_info(f"DPO PPL = {ppl_dpo:.2f}")
    else:
        trace_dpo = None
        log_info("DPO policy checkpoint not found.")

    # Save traces (optional)
    np.save(os.path.join(FIG_DIR, f"trace_sw_cache{max_cache}.npy"), trace_sw)
    log_info("Saved SW trace.")
    if trace_bc is not None:
        np.save(os.path.join(FIG_DIR, f"trace_bc_cache{max_cache}.npy"), trace_bc)
    if trace_dpo is not None:
        np.save(os.path.join(FIG_DIR, f"trace_dpo_cache{max_cache}.npy"), trace_dpo)

    # Visualization settings
    MAX_T = 512
    MAX_TOKENS = 512

    # --- retention maps ---
    plot_trace(
        trace_sw,
        title=f"SW retention map (cache={max_cache})",
        max_t=MAX_T,
        max_tokens=MAX_TOKENS,
        save_path=os.path.join(FIG_DIR, f"retention_sw_cache{max_cache}.png"),
    )

    if trace_bc is not None:
        plot_trace(
            trace_bc,
            title=f"BC policy retention map (cache={max_cache})",
            max_t=MAX_T,
            max_tokens=MAX_TOKENS,
            save_path=os.path.join(FIG_DIR, f"retention_bc_cache{max_cache}.png"),
        )
        plot_diff(
            trace_sw,
            trace_bc,
            title=f"BC - SW (cache={max_cache})",
            max_t=MAX_T,
            max_tokens=MAX_TOKENS,
            save_path=os.path.join(FIG_DIR, f"diff_bc_minus_sw_cache{max_cache}.png"),
        )

    if trace_dpo is not None:
        plot_trace(
            trace_dpo,
            title=f"DPO policy retention map (cache={max_cache})",
            max_t=MAX_T,
            max_tokens=MAX_TOKENS,
            save_path=os.path.join(FIG_DIR, f"retention_dpo_cache{max_cache}.png"),
        )
        plot_diff(
            trace_sw,
            trace_dpo,
            title=f"DPO - SW (cache={max_cache})",
            max_t=MAX_T,
            max_tokens=MAX_TOKENS,
            save_path=os.path.join(FIG_DIR, f"diff_dpo_minus_sw_cache{max_cache}.png"),
        )

    log_info("Done. All figures saved.")


if __name__ == "__main__":
    main()
