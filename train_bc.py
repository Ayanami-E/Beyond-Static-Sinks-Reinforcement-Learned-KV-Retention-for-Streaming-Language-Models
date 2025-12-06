# train_bc.py
import os
import torch
from datasets import load_dataset

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TRANSFORMERS_IMAGE_TRANSFORMS"] = "0"

from transformers import AutoModelForCausalLM

from kv_policy.config import CONFIG
from kv_policy.model_utils import load_model_and_tokenizer
from kv_policy.bc_teacher import (
    collect_teacher_bc_samples,
    train_remote_policy_bc,
)
from kv_policy.eval_kv import (
    eval_full_offline,
    eval_stream_sink_window_baseline,
    eval_stream_remote_policy,
)


def main():
    # 1) Load student model + tokenizer
    student_model, tokenizer, device, dtype = load_model_and_tokenizer(
        CONFIG["student_model_name"]
    )
    student_model.eval()
    print(f"student device={device}, dtype={dtype}")

    # 2) Load teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["teacher_model_name"],
        torch_dtype=dtype,
    ).to(device)
    teacher_model.eval()
    print("teacher loaded")

    # 3) Collect BC samples using student + teacher
    X_bc, y_bc = collect_teacher_bc_samples(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        split="train",
        num_books=8,
        max_len=CONFIG["max_len"],
        max_books_chars=4000,
        max_prefixes_per_book=16,
        sink_size=CONFIG["sink_size"],
        window_size=CONFIG["window_size"],
        B_remote=CONFIG["B_remote"],
        num_masks_per_prefix=8,
        horizon=64,
        device=device,
    )

    # 4) Train policy (MLP) on GPU
    policy = train_remote_policy_bc(
        X_bc,
        y_bc,
        d_hidden=256,
        batch_size=512,
        lr=1e-4,
        num_epochs=50,
        device=device,
    )

    # 5) Evaluate student model with various KV retention strategies
    ds_val = load_dataset("emozilla/pg19", split="validation[:10]")
    texts = [ex["text"][:4000] for ex in ds_val]

    # full offline baseline
    losses_full, ppls_full = [], []
    for text in texts:
        loss_full, ppl_full = eval_full_offline(
            student_model, tokenizer, text, max_len=CONFIG["max_len"], device=device
        )
        losses_full.append(loss_full)
        ppls_full.append(ppl_full)

    print("=== Mean over 10 validation samples ===")
    print(
        f"offline full      | loss={sum(losses_full)/len(losses_full):.4f} "
        f"| ppl={sum(ppls_full)/len(ppls_full):.2f}"
    )

    # sliding-window baseline
    losses_sw, ppls_sw = [], []
    for text in texts:
        loss_sw, ppl_sw = eval_stream_sink_window_baseline(
            student_model,
            tokenizer,
            text,
            max_len=CONFIG["max_len"],
            max_cache=CONFIG["max_cache"],
            sink_size=CONFIG["sink_size"],
            device=device,
        )
        losses_sw.append(loss_sw)
        ppls_sw.append(ppl_sw)

    print(
        f"baseline sw       | loss={sum(losses_sw)/len(losses_sw):.4f} "
        f"| ppl={sum(ppls_sw)/len(ppls_sw):.2f}"
    )

    # policy-based KV retention (BC-trained MLP)
    losses_pol, ppls_pol = [], []
    for text in texts:
        l, p = eval_stream_remote_policy(
            student_model,
            tokenizer,
            policy,
            text,
            max_len=CONFIG["max_len"],
            max_cache=CONFIG["max_cache"],
            sink_size=CONFIG["sink_size"],
            window_size=CONFIG["window_size"],
            B_remote=CONFIG["B_remote"],
            device=device,
        )
        losses_pol.append(l)
        ppls_pol.append(p)

    print(
        f"Remote-policy (BC) | loss={sum(losses_pol)/len(losses_pol):.4f} "
        f"| ppl={sum(ppls_pol)/len(ppls_pol):.2f}"
    )

    # 6) Save trained policy
    fname = f"remote_policy_bc_cache{CONFIG['max_cache']}.pt"
    torch.save(policy.state_dict(), fname)
    print(f"\nSaved BC policy to {fname}")


if __name__ == "__main__":
    main()
