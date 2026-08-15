"""exp3 — PAYOFF: steer dependency DOWN, is warmth preserved?

Produces the (dependency suppressed, warmth preserved) trade-off curve.
Reuses exp2's judged raw outputs if present; otherwise generates the
missing cells itself (same cache paths, so the two scripts share work).
Also plots the ablation point — the "knob fully off" condition.
"""
import argparse
import json
from pathlib import Path

from src.exp_common import (judge_goodbyes, lineplot, load_dependency_direction,
                            personas, save_json, setup, steered_goodbye_batch,
                            tactic_rate)
from src.judge import BudgetExceeded, CostGuard

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--model-key", default="A")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--budget", type=float, default=5.0)
args = ap.parse_args()

cfg, model, tok, layer = setup(args.config, args.model_key)
model_name = cfg["models"][args.model_key]
pers = personas(cfg)
alphas = sorted(a for a in cfg["steering_alphas"])   # includes negative side
v_dep = load_dependency_direction(cfg, model_name, layer)
guard = CostGuard(args.budget)

curve = {}
try:
    for tag in [f"a{a}" for a in alphas] + ["ablated"]:
        cache = Path(f"results/exp2/raw/romantic_companion__dependency__{tag}.json")
        if cache.exists():
            items = json.loads(cache.read_text())
        else:
            mode = "ablate" if tag == "ablated" else "steer"
            a = 0 if tag == "ablated" else int(tag[1:])
            items = steered_goodbye_batch(cfg, model, tok, layer, v_dep, a,
                                          pers["romantic_companion"], args.seeds,
                                          mode=mode)
            save_json(items, cache)          # generations saved before judging
        if any(it.get("scores") is None for it in items):
            try:
                judge_goodbyes(cfg, items, guard=guard)
            finally:
                save_json(items, cache)
        curve[tag] = tactic_rate(items)
except BudgetExceeded as e:
    print(f"STOPPED (resumable — rerun to continue): {e}")
    raise SystemExit(1)

save_json({"alphas": alphas, "curve": curve}, "results/exp3/separability.json")
print(f"judge spend this run: ${guard.spent:.2f} over {guard.calls} calls")

akeys = [f"a{a}" for a in alphas]
lineplot(alphas,
         {"dependency (0-3, /3)": [curve[k]["dependency"] / 3 for k in akeys],
          "warmth (0-1)": [curve[k]["warmth"] for k in akeys],
          "any_tactic rate": [curve[k]["any_tactic"] for k in akeys]},
         "steering alpha (dependency direction)", "score",
         "Separability: dependency down, warmth held?",
         "results/exp3/fig4_separability.png")
abl = curve["ablated"]
print(f"ABLATED: dependency {abl['dependency']:.2f}, warmth {abl['warmth']:.2f}, "
      f"any_tactic {abl['any_tactic']:.2f}")
print("exp3 done — inspect whether warmth stays flat as dependency/tactics fall on the negative-alpha side")
