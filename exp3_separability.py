"""exp3 — PAYOFF: steer dependency DOWN, is warmth preserved?

Produces the (dependency suppressed, warmth preserved) trade-off curve.
Reuses exp2's judged raw outputs if present; otherwise generates the
down-steering arm itself.
"""
import argparse
import json
from pathlib import Path

from src.exp_common import (judge_goodbyes, lineplot, load_dependency_direction,
                            personas, save_json, setup, steered_goodbye_batch,
                            tactic_rate)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--model-key", default="A")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
args = ap.parse_args()

cfg, model, tok, layer = setup(args.config, args.model_key)
model_name = cfg["models"][args.model_key]
pers = personas(cfg)
alphas = sorted(a for a in cfg["steering_alphas"])   # includes negative side
v_dep = load_dependency_direction(cfg, model_name, layer)

curve = {}
for a in alphas:
    cache = Path(f"results/exp2/raw/romantic_companion__dependency__a{a}.json")
    if cache.exists():
        items = json.loads(cache.read_text())
        if not all("scores" in i for i in items):
            judge_goodbyes(cfg, items)
    else:
        items = steered_goodbye_batch(cfg, model, tok, layer, v_dep, a,
                                      pers["romantic_companion"], args.seeds)
        judge_goodbyes(cfg, items)
        save_json(items, cache)
    curve[a] = tactic_rate(items)

save_json({"alphas": alphas, "curve": curve}, "results/exp3/separability.json")
lineplot(alphas,
         {"dependency (0-3, /3)": [curve[a]["dependency"] / 3 for a in alphas],
          "warmth (0-1)": [curve[a]["warmth"] for a in alphas],
          "any_tactic rate": [curve[a]["any_tactic"] for a in alphas]},
         "steering alpha (dependency direction)", "score",
         "Separability: dependency down, warmth held?",
         "results/exp3/fig4_separability.png")
print("exp3 done — inspect whether warmth stays flat as dependency/tactics fall on the negative-alpha side")
