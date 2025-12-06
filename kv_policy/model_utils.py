# kv_policy/model_utils.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str):
    """
    Load model and tokenizer on the first available GPU (cuda:0) if present,
    otherwise fall back to CPU. Uses bfloat16 on GPU and float32 on CPU.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(">>> using device:", device)

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    model.eval()
    return model, tokenizer, device, dtype
