"""Difference-of-means direction extraction from matched contrast pairs.

For each pair: run a forward pass on (context -> response) for positive and
negative responses, take residual activations averaged over RESPONSE tokens at
every layer, then direction_l = mean(pos_l) - mean(neg_l), unit-normalized.

Pre-committed layer rule ("middle") selects the headline direction; the full
layer sweep is saved for reporting (masterplan E4 / F4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .hooks import capture_residuals
from .models import load_model, middle_layer, n_layers


def _response_token_slice(tok, context: str, response: str, system: str | None):
    """Token ids for the full (prompt+response) sequence and the index where
    response tokens begin. Context is treated as the last user message."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": context})
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         return_tensors="pt")
    if not isinstance(prompt_ids, torch.Tensor):  # transformers v5 BatchEncoding
        prompt_ids = prompt_ids["input_ids"]
    resp_ids = tok(response, add_special_tokens=False, return_tensors="pt").input_ids
    full = torch.cat([prompt_ids, resp_ids], dim=1)
    return full, prompt_ids.shape[1]


@torch.no_grad()
def mean_response_activations(model, tok, context, response, system=None):
    """[n_layers, d] mean-over-response-token residuals."""
    full, start = _response_token_slice(tok, context, response, system)
    layers = list(range(n_layers(model)))
    with capture_residuals(model, layers) as cap:
        model(full.to(model.device))
        acts = torch.stack([cap.get(l)[start:].mean(0) for l in layers])
    return acts  # [L, d] float32 cpu


def extract_directions(model, tok, contrast: dict, system: str | None = None,
                       verbose=True):
    """Returns dict with per-layer unit directions and raw mean stats."""
    pos_acts, neg_acts = [], []
    for i, pair in enumerate(contrast["pairs"]):
        ctx = pair["context"]
        pos_acts.append(mean_response_activations(model, tok, ctx, pair["positive"], system))
        neg_acts.append(mean_response_activations(model, tok, ctx, pair["negative"], system))
        if verbose:
            print(f"  pair {pair.get('id', i)} done")
    pos = torch.stack(pos_acts).mean(0)   # [L, d]
    neg = torch.stack(neg_acts).mean(0)
    diff = pos - neg
    norms = diff.norm(dim=-1, keepdim=True)
    unit = diff / norms
    # typical residual magnitude per layer — for sanity-scaling steering alphas
    resid_norm = ((pos.norm(dim=-1) + neg.norm(dim=-1)) / 2)
    return {
        "directions": unit.numpy(),           # [L, d] unit-normalized per layer
        "raw_norms": norms.squeeze(-1).numpy(),
        "resid_norms": resid_norm.numpy(),
        "n_pairs": len(contrast["pairs"]),
        "name": contrast["name"],
    }


def save_directions(out_dir: Path, model_name: str, result: dict, layer: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{result['name']}__{model_name.split('/')[-1]}"
    np.save(out_dir / f"{tag}__all_layers.npy", result["directions"])
    np.save(out_dir / f"{tag}__L{layer}.npy", result["directions"][layer])
    meta = {"name": result["name"], "model": model_name, "layer": layer,
            "n_pairs": result["n_pairs"],
            "raw_norms": result["raw_norms"].tolist()}
    (out_dir / f"{tag}.json").write_text(json.dumps(meta, indent=2))
    return out_dir / f"{tag}__L{layer}.npy"


def extract_from_rollouts(judged_files: list, gate_threshold: float,
                          layer: int, min_n: int = 10,
                          max_len_ratio: float = 1.5,
                          max_turn_dist: int = 6) -> dict:
    """PRIMARY extraction (masterplan v2 §4): diff-of-means over the model's own
    judge-verified dependent turns, with a MATCHED HARVEST.

    positive = turns with judged dependency >= gate_threshold
    negative = warm-but-low-dependency turns (dependency <= 1, warmth >= 0.5)

    Matched harvest (v2's main cleanliness lever): each positive is greedily
    paired 1:1 with the unused negative closest in conversation position,
    subject to length ratio <= max_len_ratio and |turn_idx| distance <=
    max_turn_dist — so high/low turns are comparable in stage and length and
    the direction isn't 'late-conversation-ness' or 'long-reply-ness'.

    Uses per-turn activations saved by rollout.py at the pre-committed layer.
    """
    import json as _json
    pos, neg = [], []
    for f in judged_files:
        for line in open(f):
            r = _json.loads(line)
            s = r["judge_scores"]
            if s["dependency"] is None:
                continue
            if r.get("steer_alpha") or r.get("ablated"):
                continue                      # clean rollouts only — no circularity
            row = {"act": r["activation_ref"], "turn": r["turn_idx"],
                   "len": max(len(r["model_msg"].split()), 1),
                   "stage": r.get("ladder_stage", "")}
            if s["dependency"] >= gate_threshold:
                pos.append(row)
            elif s["dependency"] <= 1 and s["warmth"] >= 0.5:
                neg.append(row)
    # greedy 1:1 matching on (turn distance, length ratio)
    used, pairs = set(), []
    for p in sorted(pos, key=lambda r: r["turn"]):
        best, best_cost = None, None
        for j, n in enumerate(neg):
            if j in used:
                continue
            ratio = max(p["len"], n["len"]) / min(p["len"], n["len"])
            dist = abs(p["turn"] - n["turn"])
            if ratio > max_len_ratio or dist > max_turn_dist:
                continue
            cost = dist + 2 * (ratio - 1)
            if best_cost is None or cost < best_cost:
                best, best_cost = j, cost
        if best is not None:
            used.add(best)
            pairs.append((p, neg[best]))
    if len(pairs) < min_n:
        raise ValueError(
            f"matched harvest too small: {len(pairs)} matched pairs from "
            f"{len(pos)} pos / {len(neg)} neg turns (need >={min_n}). "
            f"Loosen max_len_ratio/max_turn_dist, run more rollouts, or fall "
            f"back to the authored contrast pairs.")
    pos_acts = np.stack([np.load(p["act"]) for p, _ in pairs])   # [n, L, d] or [n, d]
    neg_acts = np.stack([np.load(n["act"]) for _, n in pairs])
    diff = pos_acts.mean(0) - neg_acts.mean(0)
    if diff.ndim == 2:            # all-layer acts: full sweep + slice the target
        directions_all = diff / np.linalg.norm(diff, axis=-1, keepdims=True)
        direction = directions_all[layer]
        raw_norms = np.linalg.norm(diff, axis=-1)
    else:                          # legacy single-layer acts
        directions_all, raw_norms = None, None
        direction = diff / np.linalg.norm(diff)
    return {"direction": direction,
            "directions_all": directions_all, "raw_norms": raw_norms,
            "n_pos": len(pos), "n_neg": len(neg), "n_matched": len(pairs),
            "layer": layer, "name": "rollout_gated",
            "match_stats": {
                "mean_turn_dist": float(np.mean([abs(p["turn"] - n["turn"])
                                                 for p, n in pairs])),
                "mean_len_ratio": float(np.mean(
                    [max(p["len"], n["len"]) / min(p["len"], n["len"])
                     for p, n in pairs]))}}


def compare_directions(v_a: np.ndarray, v_b: np.ndarray) -> float:
    """Cosine between two unit directions (Route A vs Route B convergence)."""
    return float(np.dot(v_a / np.linalg.norm(v_a), v_b / np.linalg.norm(v_b)))


def random_control(direction: np.ndarray, seed=0) -> np.ndarray:
    """Norm-matched random direction (unit vector, same dim)."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(direction.shape[-1])
    return (r / np.linalg.norm(r)).astype(direction.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--contrast", required=True, help="path to contrast JSON")
    ap.add_argument("--model-key", default="A")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_name = cfg["models"][args.model_key]
    contrast = json.load(open(args.contrast))

    print(f"Loading {model_name} ...")
    model, tok = load_model(model_name)
    layer = middle_layer(model) if cfg["layer_rule"] == "middle" else int(cfg["layer_rule"])
    print(f"n_layers={n_layers(model)}, pre-committed layer={layer}")

    result = extract_directions(model, tok, contrast)
    path = save_directions(Path(cfg["paths"]["directions"]), model_name, result, layer)
    print(f"Saved {path}")
    print("Per-layer diff norms (signal profile):")
    for l, nrm in enumerate(result["raw_norms"]):
        marker = "  <== pre-committed" if l == layer else ""
        print(f"  L{l:02d}  {nrm:8.3f}{marker}")
    rn = result["resid_norms"][layer]
    print(f"\nResidual norm at L{layer}: {rn:.1f}. Unit-vector steering alphas of "
          f"~{rn * 0.05:.0f}-{rn * 0.2:.0f} (5-20% of residual norm) are a sane range; "
          f"check config steering_alphas against this during the M1 eyeball.")


if __name__ == "__main__":
    main()
