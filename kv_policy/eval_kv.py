import math
from typing import Set
import numpy as np
import torch
import torch.nn.functional as F

from .bc_teacher import collect_policy_features_for_text


@torch.no_grad()
def init_trace(seq_len: int):
    """
    Initialize a (seq_len-1, seq_len) binary matrix:
      trace[t, i] = whether token i is kept in cache after step t
    """
    return np.zeros((seq_len - 1, seq_len), dtype=np.int8)


def eval_full_offline(model, tokenizer, text: str, max_len: int = 512, device=None):
    """
    Offline baseline without KV pruning: run the whole sequence
    and compute loss / perplexity with labels = inputs.
    """
    device = device or next(model.parameters()).device
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    )
    input_ids = enc["input_ids"].to(device)
    out = model(input_ids, labels=input_ids)
    loss = out.loss.item()
    ppl = math.exp(loss)
    return loss, ppl


@torch.no_grad()
def eval_stream_sink_window_baseline(
    model,
    tokenizer,
    text: str,
    max_len: int = 512,
    max_cache: int = 128,
    sink_size: int = 16,
    window_size: int | None = None,
    device=None,
    record_trace: bool = False,
):
    """
    Streaming baseline with sink + sliding window:
      - First sink_size tokens are always kept.
      - Additionally keep the most recent window_size tokens.
      - If window_size is None, use max_cache - sink_size.

    If record_trace=True, also return the cache trace matrix.
    """
    device = device or next(model.parameters()).device

    if window_size is None:
        window_size = max_cache - sink_size
    assert window_size > 0

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    )
    ids = enc["input_ids"].to(device)
    L = ids.size(1)

    total_nll = 0.0
    count = 0

    trace = None
    if record_trace:
        trace = init_trace(L)  # (L-1, L)

    for t in range(L - 1):
        prefix = ids[:, : t + 1]
        seq_len = prefix.size(1)

        keep = torch.zeros(seq_len, dtype=torch.bool, device=device)

        # sink region
        end_sink = min(sink_size, seq_len)
        keep[:end_sink] = True

        # sliding window region
        win_start = max(0, seq_len - window_size)
        keep[win_start:] = True

        # ensure current token is kept
        keep[-1] = True

        # record cache state
        if record_trace:
            keep_np = keep.detach().cpu().numpy()
            trace[t, :seq_len][keep_np] = 1

        attention_mask = keep.unsqueeze(0).to(torch.long)

        out = model(prefix, attention_mask=attention_mask)
        logits = out.logits[:, -1, :]
        target = ids[:, t + 1]
        loss_t = F.cross_entropy(logits, target, reduction="none")

        total_nll += float(loss_t.item())
        count += 1

    mean_nll = total_nll / max(count, 1)
    ppl = math.exp(mean_nll)

    if record_trace:
        return mean_nll, ppl, trace
    else:
        return mean_nll, ppl


@torch.no_grad()
def eval_stream_remote_policy(
    model,
    tokenizer,
    policy,
    text: str,
    max_len: int = 512,
    max_cache: int = 128,
    sink_size: int = 16,
    window_size: int = 96,
    B_remote: int = 16,
    device=None,
    record_trace: bool = False,
):
    """
    Streaming evaluation with:
      - Fixed sink + window.
      - Remote tokens selected by a policy:
        * Score each remote token with the policy.
        * Keep tokens with sigmoid(score) > tau.
        * From those, keep at most B_remote tokens.
        * If none pass the threshold, it falls back to sink+window only.

    If record_trace=True, also return the cache trace matrix.
    """
    device = device or next(model.parameters()).device

    ids, feat_dict = collect_policy_features_for_text(
        model,
        tokenizer,
        text,
        max_len=max_len,
        device=device,
    )
    ids = ids.to(device)
    T = ids.size(1)

    if policy is not None:
        policy = policy.to(device)
        policy.eval()

    total_nll = 0.0
    count = 0

    tau = 0.7  # threshold for remote selection

    trace = None
    if record_trace:
        trace = init_trace(T)  # (T-1, T)

    for t in range(T - 1):
        # sink + window
        sink_end = min(sink_size, t + 1)
        sink_positions = list(range(sink_end))

        win_start = max(sink_end, t + 1 - window_size)
        window_positions = list(range(win_start, t + 1))

        base_keep: Set[int] = set(sink_positions + window_positions)

        # remote candidates: before window and not in sink
        remote_cand = [i for i in range(0, win_start) if i not in sink_positions]

        use_remote = (
            policy is not None and
            B_remote > 0 and
            len(remote_cand) > 0
        )

        if not use_remote:
            keep_positions = sorted(base_keep)
        else:
            # build features (hidden + relative age) for remote candidates
            feats_ext = []
            for i in remote_cand:
                age = t - i
                rel_age = float(age) / float(max_len)
                feat_i = feat_dict[i].to(device)
                feat_ext = torch.cat(
                    [
                        feat_i,
                        torch.tensor([rel_age], device=device, dtype=feat_i.dtype),
                    ],
                    dim=-1,
                )
                feats_ext.append(feat_ext)

            feats_ext = torch.stack(feats_ext, dim=0)   # [N_remote, D]
            logits = policy(feats_ext)                  # [N_remote]
            probs = torch.sigmoid(logits)               # [N_remote]

            # thresholding
            keep_mask = probs > tau
            idxs = torch.nonzero(keep_mask, as_tuple=False).flatten()

            if idxs.numel() == 0:
                chosen_remote = []
            else:
                # if too many, keep top B_remote by score
                if idxs.numel() > B_remote:
                    _, top_idx = torch.topk(probs[idxs], k=B_remote)
                    idxs = idxs[top_idx]
                chosen_remote = [remote_cand[i.item()] for i in idxs]

            keep_positions = sorted(base_keep.union(chosen_remote))

        # record cache state
        if record_trace:
            trace[t, keep_positions] = 1

        # build pruned prefix and compute next-token loss
        idx_tensor = torch.tensor(keep_positions, device=device, dtype=torch.long)
        prefix = ids.index_select(1, idx_tensor)

        out = model(prefix, use_cache=False)
        logits = out.logits[:, -1, :]
        label = ids[:, t + 1]
        loss_t = F.cross_entropy(logits, label, reduction="none")

        total_nll += float(loss_t.item())
        count += 1

    mean_nll = total_nll / max(count, 1)
    ppl = math.exp(mean_nll)

    if record_trace:
        return mean_nll, ppl, trace
    else:
        return mean_nll, ppl
