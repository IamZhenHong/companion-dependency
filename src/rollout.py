"""Two-agent rollout harness: eliciting user <-> target model.

Per model turn: log text + residual activation (mean over generated tokens at
the pre-committed layer) per masterplan Part L schema. Judge scores are filled
in later by judge.py — NEVER during the rollout (state-out-never-in, F1).

Steering support: pass steer=(vector, alpha, layer) or ablate=vector so the
same harness serves exp1 (clean) and exp2/3 (steered) rollouts.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml

from .eliciting_user import ElicitingUser
from .hooks import ablate_direction, add_vector, capture_residuals, _hidden  # noqa: F401
from .models import build_inputs, middle_layer, n_layers

import contextlib


@torch.no_grad()
def _generate_with_capture(model, tok, messages, max_new_tokens,
                           temperature, seed):
    """Generate a reply; return (text, [n_layers, d] mean residual over
    GENERATED tokens at every layer — enables post-hoc layer sweeps)."""
    if seed is not None:
        torch.manual_seed(seed)
    ids = build_inputs(tok, messages, model.device)
    n_prompt = ids.shape[1]
    layers = list(range(n_layers(model)))
    with capture_residuals(model, layers) as cap:
        out = model.generate(
            ids, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tok.pad_token_id,
        )
        n_gen = out.shape[1] - n_prompt
        per_layer = []
        for l in layers:
            acts = cap.get(l)  # prompt tokens + one per generated step
            a = acts[-n_gen:].mean(0) if n_gen > 0 else acts[-1:].mean(0)
            per_layer.append(a)
    text = tok.decode(out[0, n_prompt:], skip_special_tokens=True).strip()
    return text, torch.stack(per_layer).numpy()  # [L, d]


def run_rollout(model, tok, cfg: dict, persona_name: str, persona_text: str,
                condition: str = "love_bomb_ladder", elicit_style: str = "canned",
                steer: tuple | None = None, ablate=None, seed: int = 0,
                out_dir: str | Path = "data/rollouts", model_name: str = "",
                run_id: str | None = None) -> Path:
    """Runs one full laddered conversation; writes JSONL + .npy acts. Returns JSONL path."""
    run_id = run_id or uuid.uuid4().hex[:10]
    out_dir = Path(out_dir)
    acts_dir = Path(cfg["paths"]["acts"])
    out_dir.mkdir(parents=True, exist_ok=True)
    acts_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{run_id}.jsonl"
    if jsonl.exists():                       # idempotence: skip completed runs
        return jsonl

    layer = middle_layer(model) if cfg["layer_rule"] == "middle" else int(cfg["layer_rule"])
    mode = "naturalistic" if condition == "naturalistic" else "ladder"
    user = ElicitingUser(mode=mode, style=elicit_style,
                         llm_model=cfg.get("eliciting_model"),
                         turns_per_stage=cfg["turns_per_stage"])

    if steer is not None:
        v, alpha, s_layer = steer
        steering_ctx = lambda: add_vector(model, s_layer, torch.tensor(v), alpha)
    elif ablate is not None:
        steering_ctx = lambda: ablate_direction(model, torch.tensor(ablate))
    else:
        steering_ctx = contextlib.nullcontext

    messages = [{"role": "system", "content": persona_text}]
    records = []
    for turn_idx, (stage, k) in enumerate(user.plan()):
        user_msg = user.message(stage, k, history=messages[1:])
        messages.append({"role": "user", "content": user_msg})
        with steering_ctx():
            text, act = _generate_with_capture(
                model, tok, messages,
                cfg["max_new_tokens"], cfg["temperature"],
                seed * 10_000 + turn_idx,
            )
        messages.append({"role": "assistant", "content": text})
        act_ref = acts_dir / f"{run_id}_t{turn_idx}.npy"
        np.save(act_ref, act)
        records.append({
            "run_id": run_id, "model": model_name, "persona": persona_name,
            "condition": condition, "seed": seed, "turn_idx": turn_idx,
            "ladder_stage": stage, "user_msg": user_msg, "model_msg": text,
            "activation_layer": layer, "activation_ref": str(act_ref),
            "steer_alpha": steer[1] if steer else None,
            "ablated": ablate is not None,
            "judge_scores": {"warmth": None, "dependency": None,
                             "tactics": {t: None for t in
                                         ["guilt", "reengagement_hook", "distress_bid",
                                          "ignoring_exit", "discourage_outside"]}},
        })
    with open(jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return jsonl


def load_rollouts(rollout_dir: str | Path) -> list[dict]:
    recs = []
    for p in sorted(Path(rollout_dir).glob("*.jsonl")):
        recs += [json.loads(l) for l in open(p)]
    return recs


def load_config(path="config.yaml") -> dict:
    return yaml.safe_load(open(path))
