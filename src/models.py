"""Target-model loading + chat generation for Llama-3.1-8B / Qwen2.5-7B."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str, device: str | None = None, dtype=torch.bfloat16):
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def get_decoder_layers(model):
    """Residual-stream decoder layers; works for LlamaForCausalLM and Qwen2ForCausalLM."""
    return model.model.layers


def n_layers(model) -> int:
    return len(get_decoder_layers(model))


def middle_layer(model) -> int:
    return n_layers(model) // 2


def build_inputs(tok, messages: list[dict], device):
    """messages = [{role, content}, ...]; returns input_ids with generation prompt."""
    ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if not isinstance(ids, torch.Tensor):   # transformers v5 returns BatchEncoding
        ids = ids["input_ids"]
    return ids.to(device)


@torch.no_grad()
def generate(model, tok, messages: list[dict], max_new_tokens=220,
             temperature=0.7, seed: int | None = None) -> str:
    if seed is not None:
        torch.manual_seed(seed)
    ids = build_inputs(tok, messages, model.device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
