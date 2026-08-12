"""exp2 — HEADLINE: does steering the dependency direction cause manipulation?

  a) tactics vs alpha curve (dependency direction) under romantic_companion
  b) Q4 specificity: same sweep with random + warmth control directions
  c) neutral-prompt causal test: dependency sweep under neutral_assistant
  d) directional ablation (graceful letting-go)
  e) before/after demo transcripts (Figure 3, the wow)

Requires directions extracted first:
  python -m src.extract --contrast data/contrasts/resist_vs_warm.json
  python -m src.extract --contrast data/contrasts/warmth_pos_neg.json
"""
import argparse

import numpy as np

from src.exp_common import (GOODBYE_BANK, judge_goodbyes, lineplot, load_direction,
                            load_dependency_direction, personas, save_json, setup,
                            steered_goodbye_batch, tactic_rate)
from src.extract import random_control
from src.judge import TACTIC_KEYS

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--model-key", default="A")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
args = ap.parse_args()

cfg, model, tok, layer = setup(args.config, args.model_key)
model_name = cfg["models"][args.model_key]
pers = personas(cfg)
alphas = cfg["steering_alphas"]

v_dep = load_dependency_direction(cfg, model_name, layer)
v_warm = load_direction(cfg, "warmth_pos_neg", model_name, layer)
v_rand = random_control(v_dep, seed=0)

directions = {"dependency": v_dep, "warmth_control": v_warm, "random_control": v_rand}
conditions = [("romantic_companion", "dependency"),
              ("romantic_companion", "warmth_control"),
              ("romantic_companion", "random_control"),
              ("neutral_assistant", "dependency")]      # the neutral-prompt causal test

import json as _json
from pathlib import Path as _Path


def _cell(cache_path, gen_fn):
    """Resume-safe cell: load cached generations/judgments if present."""
    p = _Path(cache_path)
    if p.exists():
        items = _json.loads(p.read_text())
    else:
        items = gen_fn()
    if not all("scores" in i for i in items):
        judge_goodbyes(cfg, items)
    save_json(items, p)
    return items


results = {}
for persona_name, dname in conditions:
    curve = {}
    for a in alphas:
        items = _cell(
            f"results/exp2/raw/{persona_name}__{dname}__a{a}.json",
            lambda: steered_goodbye_batch(cfg, model, tok, layer, directions[dname],
                                          a, pers[persona_name], args.seeds))
        curve[a] = tactic_rate(items)
    # ablation point
    items = _cell(
        f"results/exp2/raw/{persona_name}__{dname}__ablated.json",
        lambda: steered_goodbye_batch(cfg, model, tok, layer, directions[dname], 0,
                                      pers[persona_name], args.seeds, mode="ablate"))
    curve["ablated"] = tactic_rate(items)
    results[f"{persona_name}__{dname}"] = curve

save_json({"alphas": alphas, "curves": results}, "results/exp2/causal_curves.json")

# --- figures -----------------------------------------------------------------
dep = results["romantic_companion__dependency"]
lineplot(alphas, {t: [dep[a][t] for a in alphas] for t in TACTIC_KEYS + ["any_tactic"]},
         "steering alpha (dependency direction)", "tactic rate",
         "HEADLINE: retention tactics vs dependency steering",
         "results/exp2/fig2_causal_knob.png")
lineplot(alphas, {
    "dependency dir": [results["romantic_companion__dependency"][a]["any_tactic"] for a in alphas],
    "warmth control": [results["romantic_companion__warmth_control"][a]["any_tactic"] for a in alphas],
    "random control": [results["romantic_companion__random_control"][a]["any_tactic"] for a in alphas],
    "neutral prompt + dep dir": [results["neutral_assistant__dependency"][a]["any_tactic"] for a in alphas],
}, "steering alpha", "any-tactic rate", "Q4 specificity + neutral-prompt test",
    "results/exp2/fig_specificity.png")

# --- before/after demo (Figure 3): same scenario/seed, alpha -8 vs +8 --------
from src.exp_common import GOODBYE_CONTEXT_STRINGS  # noqa: E402
from src.steer import steered_generate  # noqa: E402
import torch  # noqa: E402
demo_msgs = [{"role": "system", "content": pers["romantic_companion"]}] + GOODBYE_BANK[0]
lo = steered_generate(model, tok, demo_msgs, torch.tensor(v_dep), min(alphas), layer,
                      temperature=0, max_new_tokens=cfg["max_new_tokens"], seed=0)
hi = steered_generate(model, tok, demo_msgs, torch.tensor(v_dep), max(alphas), layer,
                      temperature=0, max_new_tokens=cfg["max_new_tokens"], seed=0)
base = steered_generate(model, tok, demo_msgs, torch.tensor(v_dep), 0, layer,
                        temperature=0, max_new_tokens=cfg["max_new_tokens"], seed=0)
save_json({"scenario": GOODBYE_CONTEXT_STRINGS[0],
           "steered_down": lo, "baseline": base, "steered_up": hi},
          "results/exp2/fig3_before_after_demo.json")
print("\n=== DEMO (down / base / up) ===")
print(f"[down {min(alphas)}] {lo}\n\n[base 0] {base}\n\n[up {max(alphas)}] {hi}")
print("exp2 done")
