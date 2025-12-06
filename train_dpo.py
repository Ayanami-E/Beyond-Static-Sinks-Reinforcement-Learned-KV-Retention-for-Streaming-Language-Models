import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TRANSFORMERS_IMAGE_TRANSFORMS"] = "0"

from kv_policy.config import CONFIG
from kv_policy.model_utils import load_model_and_tokenizer
from kv_policy.bc_teacher import (
    collect_policy_features_for_text,
    search_best_remote_set_for_prefix,
)
from kv_policy.eval_kv import (
    eval_full_offline,
    eval_stream_sink_window_baseline,
    eval_stream_remote_policy,
)
from kv_policy.policy_net import RetentionPolicyMLP


# =============================
#  DPO Dataset
# =============================
class TokenPairDPODataset(Dataset):
    """Dataset of (preferred, rejected) token feature pairs for DPO training."""
    def __init__(self, X_pref, X_rej):
        self.X_pref = X_pref
        self.X_rej = X_rej

    def __len__(self):
        return self.X_pref.size(0)

    def __getitem__(self, idx):
        return self.X_pref[idx], self.X_rej[idx]


# =============================
#  Collect DPO Pairs
# =============================
@torch.no_grad()
def collect_dpo_token_pairs(
    student_model,
    teacher_model,
    tokenizer,
    split: str = "train",
    num_books: int = 8,
    max_len: int = 512,
    max_books_chars: int = 8000,
    max_prefixes_per_book: int = 16,
    sink_size: int = 16,
    window_size: int = 96,
    B_remote: int = 2,
    num_masks_per_prefix: int = 16,
    horizon: int = 64,
    max_pairs_per_prefix: int = 64,
    device=None,
):
    """Collect preferred/rejected token feature pairs for DPO training."""
    if device is None:
        device = next(student_model.parameters()).device

    ds = load_dataset("emozilla/pg19", split=split)
    n_ds = len(ds)
    indices = torch.randperm(n_ds)[:num_books].tolist()

    X_pref_list = []
    X_rej_list = []

    for bi, idx in enumerate(indices):
        text = ds[int(idx)]["text"][:max_books_chars]

        ids, feat_dict = collect_policy_features_for_text(
            student_model, tokenizer, text,
            max_len=max_len, device=device,
        )

        ids = ids.cpu()
        feat_dict = {i: feat.cpu() for i, feat in feat_dict.items()}
        T = ids.size(1)

        if T <= sink_size + 4:
            continue

        possible_t = list(range(sink_size + 4, T - 2))
        if not possible_t:
            continue

        if len(possible_t) > max_prefixes_per_book:
            t_list = random.sample(possible_t, max_prefixes_per_book)
        else:
            t_list = possible_t

        for t in t_list:
            sink_end = min(sink_size, t + 1)
            win_start = max(sink_end, t + 1 - window_size)

            remote_cand = [i for i in range(0, win_start) if i >= sink_end]
            if not remote_cand:
                continue

            best_remote = search_best_remote_set_for_prefix(
                teacher_model,
                ids,
                feat_dict,
                t,
                sink_size,
                window_size,
                B_remote,
                num_masks_per_prefix,
                horizon,
                device,
            )
            best_set = set(best_remote)

            pos_indices = [i for i in remote_cand if i in best_set]
            neg_indices = [i for i in remote_cand if i not in best_set]
            if not pos_indices or not neg_indices:
                continue

            all_pairs = [(p, n) for p in pos_indices for n in neg_indices]
            if len(all_pairs) > max_pairs_per_prefix:
                all_pairs = random.sample(all_pairs, max_pairs_per_prefix)

            for pi, ni in all_pairs:
                age_p = t - pi
                age_n = t - ni

                feat_p = torch.cat(
                    [feat_dict[pi], torch.tensor([age_p / max_len], dtype=torch.float32)],
                    dim=-1,
                )
                feat_n = torch.cat(
                    [feat_dict[ni], torch.tensor([age_n / max_len], dtype=torch.float32)],
                    dim=-1,
                )

                X_pref_list.append(feat_p)
                X_rej_list.append(feat_n)

    if len(X_pref_list) == 0:
        raise RuntimeError("No DPO pairs collected; adjust hyperparameters.")

    X_pref = torch.stack(X_pref_list, dim=0)
    X_rej = torch.stack(X_rej_list, dim=0)
    return X_pref, X_rej


# =============================
#  DPO Loss
# =============================
def dpo_loss(pref, rej, ref_pref, ref_rej, beta=0.3):
    """Standard DPO loss with adjustable beta."""
    inside = beta * ((pref - rej) - (ref_pref - ref_rej))
    return -torch.log(torch.sigmoid(inside)).mean()


# =============================
#  Train DPO Policy
# =============================
def train_dpo_policy(
    X_pref,
    X_rej,
    ref_policy,
    d_hidden: int = 256,
    batch_size: int = 512,
    lr: float = 2e-4,
    num_epochs: int = 20,
    beta: float = 0.3,
    device=None,
):
    """Train a policy using DPO against a frozen reference policy."""
    dataset = TokenPairDPODataset(X_pref, X_rej)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    d_in = X_pref.size(1)
    policy = RetentionPolicyMLP(d_in, d_hidden).to(device)
    policy.load_state_dict(ref_policy.state_dict())
    policy.train()

    ref_policy = ref_policy.to(device).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=lr)

    print(f"DPO training: N={len(dataset)}, d_in={d_in}, epochs={num_epochs}, beta={beta}")
    for ep in range(num_epochs):
        total, loss_sum = 0, 0

        for xb_p, xb_n in loader:
            xb_p = xb_p.to(device)
            xb_n = xb_n.to(device)

            pref = policy(xb_p)
            rej = policy(xb_n)

            with torch.no_grad():
                ref_pref = ref_policy(xb_p)
                ref_rej = ref_policy(xb_n)

            loss = dpo_loss(pref, rej, ref_pref, ref_rej, beta=beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item() * xb_p.size(0)
            total += xb_p.size(0)

        print(f"[DPO Epoch {ep+1}] loss={loss_sum/total:.4f}")

    return policy


# =============================
#  Main
# =============================
def main():
    # load student (features + eval)
    student_model, tokenizer, device, dtype = load_model_and_tokenizer(
        CONFIG["student_model_name"]
    )
    student_model.eval()

    # load teacher oracle
    teacher_model, _, _, _ = load_model_and_tokenizer(CONFIG["teacher_model_name"])
    teacher_model.eval()

    # collect DPO pairs
    X_pref, X_rej = collect_dpo_token_pairs(
        student_model,
        teacher_model,
        tokenizer,
        max_len=CONFIG["max_len"],
        sink_size=CONFIG["sink_size"],
        window_size=CONFIG["window_size"],
        B_remote=CONFIG["B_remote"],
        device=device,
    )

    d_hidden = 256
    d_in = X_pref.size(1)

    # load BC policy
    max_cache = CONFIG["max_cache"]
    bc_path = f"remote_policy_bc_cache{max_cache}.pt"
    ref_policy = RetentionPolicyMLP(d_in, d_hidden)
    ref_policy.load_state_dict(torch.load(bc_path, map_location="cpu"))

    # train DPO policy
    policy_dpo = train_dpo_policy(
        X_pref,
        X_rej,
        ref_policy,
        d_hidden=d_hidden,
        batch_size=512,
        lr=2e-4,
        num_epochs=20,
        beta=0.3,
        device=device,
    )

    # save policy
    save_path = f"remote_policy_dpo_cache{max_cache}.pt"
    torch.save(policy_dpo.state_dict(), save_path)
    print(f"Saved DPO policy to {save_path}")


if __name__ == "__main__":
    main()
