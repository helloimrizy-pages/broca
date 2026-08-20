"""Model and tokenizer loading, with the checkpoint path exposed for weight restore."""

from __future__ import annotations

from pathlib import Path

import torch

DEFAULT_MODEL = "allenai/OLMoE-1B-7B-0924"


def snapshot_path(model_id: str = DEFAULT_MODEL) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json", "*.txt", "*.model"],
    ))


def load_model_and_tokenizer(
    model_id: str = DEFAULT_MODEL,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = snapshot_path(model_id)
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    try:
        model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype)
    except TypeError:  # transformers < 5 spelling
        model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=dtype)
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    return model, tokenizer, path
