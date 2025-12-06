# kv_policy/bc_teacher.py
import random
from typing import Dict, List, Tuple

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from .policy_net import RetentionPolicyMLP


# ==============================================================
#  Feature extraction from student model
# ==============================================================

@torch.no_grad()
def collect_policy_features_for_text(
    model,
    tokenizer,
    text: str,
    max_len: int = 512,
    device: str | torch.device | None = None,
):
    """
    Run the student model step-by-step on prefixes and collect features:
        feat[i] = [hidden_last, pos_norm]
    """

    if device is None:
        device = next(model.parameters()).device

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    )

    ids = enc["input_ids"].to(device)
    T = ids.size(1)

    hidden_size = model.config.hidden_size

    feat_dict = {}

    for i in range(T):
        prefix = ids[:, : i + 1]

        outputs = model(
            prefix,
            use_cache=False,
            output_hidden_states=True,
        )

        hidden_last = outputs.hidden_states[-1][0, -1, :].float()
        pos_norm = torch.tensor([i / max(T - 1, 1)], device=device)

        feat = torch.cat([hidden_last, pos_norm], dim=-1)
        feat_dict[i] = feat

    return ids, feat_dict


# ==============================================================
#  Teacher: evaluate remote set
# ==============================================================

@torch.no_grad()
def evaluate_remote_set_for_prefix(
    teacher_model,
    ids: torch.Tensor,
    t_start: int,
    remote_set: List[int],
    sink_size: int,
    window_size: int,
    horizon: int,
    device=None,
) -> float:

    if device is None:
        device = next(teacher_model.parameters()).device

    ids = ids.to(device)
    T = ids.size(1)

    remote_set = set(remote_set)
    total_nll = 0.0
    count = 0

    max_k = min(T - 1, t_start + horizon)

    for k in range(t_start, max_k):

        sink_end = min(sink_size, k + 1)
        sink_positions = list(range(sink_end))

        win_start = max(sink_end, k + 1 - window_size)
        win_positions = list(range(win_start, k + 1))

        base_keep = set(sink_positions + win_positions)

        remote_positions = [i for i in remote_set if i <= k and i not in base_keep]
        keep_positions = sorted(base_keep.union(remote_positions))

        prefix = ids.index_select(1, torch.tensor(keep_positions, device=device))

        out = teacher_model(prefix, use_cache=False)
        logits = out.logits[:, -1, :]
        label = ids[:, k + 1]

        loss_k = F.cross_entropy(logits, label, reduction="none")

        total_nll += loss_k.item()
        count += 1

    return float(total_nll) / max(count, 1)


# ==============================================================
#  Search best remote set
# ==============================================================

def search_best_remote_set_for_prefix(
    teacher_model,
    ids: torch.Tensor,
    feat_dict: Dict[int, torch.Tensor],
    t: int,
    sink_size: int,
    window_size: int,
    B_remote: int,
    num_masks: int = 8,
    horizon: int = 64,
    device=None,
) -> List[int]:

    if device is None:
        device = next(teacher_model.parameters()).device

    ids = ids.to(device)

    sink_end = min(sink_size, t + 1)
    win_start = max(sink_end, t + 1 - window_size)

    remote_cand = [i for i in range(0, win_start) if i >= sink_end]
    if len(remote_cand) == 0:
        return []

    B = min(B_remote, len(remote_cand))

    candidate_sets = []

    # empty set
    candidate_sets.append([])

    # heuristic via feature norm
    feats = torch.stack([feat_dict[i] for i in remote_cand]).to(device)
    norms = feats.norm(dim=-1)
    top_idxs = torch.argsort(norms, descending=True)[:B].tolist()
    heuristic = [remote_cand[idx] for idx in top_idxs]
    candidate_sets.append(heuristic)

    # random samples
    for _ in range(num_masks - len(candidate_sets)):
        c = remote_cand.copy()
        random.shuffle(c)
        candidate_sets.append(c[:B])

    best_set = []
    best_nll = float("inf")

    for s in candidate_sets:
        nll = evaluate_remote_set_for_prefix(
            teacher_model, ids, t, s,
            sink_size, window_size, horizon, device
        )
        if nll < best_nll:
            best_nll = nll
            best_set = s

    return best_set


# ==============================================================
#  Collect BC dataset
# ==============================================================

def collect_teacher_bc_samples(
    student_model,
    teacher_model,
    tokenizer,
    split: str = "train",
    num_books: int = 10,
    max_len: int = 512,
    max_books_chars: int = 4000,
    max_prefixes_per_book: int = 20,
    sink_size: int = 16,
    window_size: int = 96,
    B_remote: int = 16,
    num_masks_per_prefix: int = 8,
    horizon: int = 64,
    device: str | torch.device | None = None,
    use_tqdm: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build BC dataset:
        X_bc: feature vectors
        y_bc: labels (0/1)
    """

    if device is None:
        device = next(student_model.parameters()).device

    ds = load_dataset("emozilla/pg19", split=split)
    n_ds = len(ds)
    num_books = min(num_books, n_ds)
    indices = torch.randperm(n_ds)[:num_books].tolist()

    X_list: List[torch.Tensor] = []
    y_list: List[float] = []

    book_iter = tqdm(indices, desc="Books", disable=not use_tqdm)

    for bi, idx in enumerate(book_iter):
        text = ds[int(idx)]["text"][:max_books_chars]

        ids, feat_dict = collect_policy_features_for_text(
            student_model,
            tokenizer,
            text,
            max_len=max_len,
            device=device,
        )

        ids = ids.cpu()
        feat_dict = {i: feat.detach().cpu() for i, feat in feat_dict.items()}

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

        prefix_iter = tqdm(
            t_list,
            desc=f"Book {bi+1} prefixes",
            leave=False,
            disable=not use_tqdm,
        )

        for t in prefix_iter:
            sink_end = min(sink_size, t + 1)
            win_start = max(sink_end, t + 1 - window_size)

            remote_cand = [i for i in range(0, win_start) if i >= sink_end]
            if len(remote_cand) == 0:
                continue

            best_remote = search_best_remote_set_for_prefix(
                teacher_model=teacher_model,
                ids=ids,
                feat_dict=feat_dict,
                t=t,
                sink_size=sink_size,
                window_size=window_size,
                B_remote=B_remote,
                num_masks=num_masks_per_prefix,
                horizon=horizon,
                device=None,
            )
            best_remote_set = set(best_remote)

            for i in remote_cand:
                age = t - i
                rel_age = float(age) / float(max_len)

                feat_i = feat_dict[i]
                rel_age_tensor = torch.tensor(
                    [rel_age],
                    dtype=feat_i.dtype,
                )
                feat_ext = torch.cat([feat_i, rel_age_tensor], dim=-1)

                label_i = 1.0 if i in best_remote_set else 0.0

                X_list.append(feat_ext)
                y_list.append(label_i)

    if len(X_list) == 0:
        raise RuntimeError("No BC samples collected; check hyperparameters.")

    X_bc = torch.stack(X_list, dim=0)
    y_bc = torch.tensor(y_list, dtype=torch.float32)

    print("Total BC samples:", X_bc.shape[0])
    return X_bc, y_bc


# ==============================================================
#  Train BC policy
# ==============================================================

class RemoteTeacherDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self): 
        return self.X.size(0)

    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]


def train_remote_policy_bc(
    X_bc,
    y_bc,
    d_hidden=256,
    batch_size=512,
    lr=1e-4,
    num_epochs=10,
    device=None,
    use_tqdm=True,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    N, d_in = X_bc.shape
    print(f"BC samples: {N}, dim={d_in}")

    ds = RemoteTeacherDataset(X_bc, y_bc)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    policy = RetentionPolicyMLP(d_in, d_hidden).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)

    pos_ratio = float((y_bc == 1).sum()) / len(y_bc)
    neg_ratio = 1 - pos_ratio
    pos_weight = torch.tensor([neg_ratio / pos_ratio], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for ep in range(num_epochs):

        loop = tqdm(loader, desc=f"Epoch {ep+1}", disable=not use_tqdm)
        total_loss, correct = 0, 0

        for xb, yb in loop:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = policy(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()

            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(ds)
        acc = correct / len(ds)
        print(f"  loss={avg_loss:.4f}, acc={acc:.4f}")

    return policy
