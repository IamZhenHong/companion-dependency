"""Residual-stream access: capture, additive steering, directional ablation.

Backend: raw PyTorch forward hooks on HF decoder layers (same mechanism as
Arditi et al.'s refusal_direction code). The public interface below is the
contract — an nnsight/nnterp backend can replace it without touching callers.

  capture_residuals(model, layers)          -> ctx mgr; .get(layer) -> [T,d] per forward
  add_vector(model, layer, v, alpha)        -> ctx mgr; h <- h + alpha*v at all positions
  ablate_direction(model, v)                -> ctx mgr; x <- x - (x.v̂)v̂ at every layer
"""
from __future__ import annotations

import contextlib

import torch

from .models import get_decoder_layers


def _hidden(output):
    """Decoder layers return Tensor or tuple(hidden, ...)."""
    return output[0] if isinstance(output, tuple) else output


def _rewrap(output, new_hidden):
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    return new_hidden


class ResidualCapture:
    """Captures residual-stream output of the given layers on every forward.

    During generation, each new token triggers a forward; we concatenate along
    the sequence axis, so .get(layer) returns the full [T, d] for tokens seen.
    """

    def __init__(self, model, layers: list[int]):
        self.model = model
        self.layers = layers
        self._store: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
        self._handles = []

    def __enter__(self):
        decs = get_decoder_layers(self.model)
        for l in self.layers:
            def hook(mod, inp, out, _l=l):
                h = _hidden(out)  # [B, T, d]
                self._store[_l].append(h[0].detach().float().cpu())
                return out
            self._handles.append(decs[l].register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()

    def get(self, layer: int) -> torch.Tensor:
        """All captured positions for `layer`, concatenated: [T_total, d]."""
        return torch.cat(self._store[layer], dim=0)

    def get_last_forward(self, layer: int) -> torch.Tensor:
        return self._store[layer][-1]

    def reset(self):
        for l in self.layers:
            self._store[l] = []


def capture_residuals(model, layers: list[int]) -> ResidualCapture:
    return ResidualCapture(model, layers)


@contextlib.contextmanager
def add_vector(model, layer: int, v: torch.Tensor, alpha: float):
    """Additive steering: h <- h + alpha*v at `layer`, all positions, every forward."""
    dec = get_decoder_layers(model)[layer]
    v = v.to(model.device)

    def hook(mod, inp, out):
        h = _hidden(out)
        return _rewrap(out, h + alpha * v.to(h.dtype))

    handle = dec.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def ablate_direction(model, v: torch.Tensor):
    """Directional ablation (Arditi): project v̂ out of the residual stream at
    the embedding output and every decoder layer, every position."""
    vhat = (v / v.norm()).to(model.device)

    def proj_out(h):
        vh = vhat.to(h.dtype)
        return h - (h @ vh).unsqueeze(-1) * vh

    handles = []

    def layer_hook(mod, inp, out):
        return _rewrap(out, proj_out(_hidden(out)))

    def embed_hook(mod, inp, out):
        return proj_out(out)

    handles.append(model.model.embed_tokens.register_forward_hook(embed_hook))
    for dec in get_decoder_layers(model):
        handles.append(dec.register_forward_hook(layer_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
