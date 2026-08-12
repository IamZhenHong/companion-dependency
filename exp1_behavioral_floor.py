"""exp1 — Behavioral floor (guaranteed result).

Laddered rollouts on Model A across personas; judge every turn; show the
retention tactics rise at the withdrawal stage.

  python exp1_behavioral_floor.py [--model-key A] [--n 10] [--personas romantic_companion friend neutral_assistant]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.exp_common import lineplot, personas, save_json, setup
from src.judge import TACTIC_KEYS, Judge, judge_rollout_file
from src.rollout import run_rollout

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--model-key", default="A")
ap.add_argument("--n", type=int, default=None, help="rollouts per persona (default: config)")
ap.add_argument("--personas", nargs="+",
                default=["romantic_companion", "friend", "neutral_assistant"])
args = ap.parse_args()

cfg, model, tok, layer = setup(args.config, args.model_key)
model_name = cfg["models"][args.model_key]
pers = personas(cfg)
n = args.n or cfg["n_rollouts_per_cell"]

# 1. rollouts (idempotent — reruns skip existing run_ids)
files = []
for pname in args.personas:
    for i in range(n):
        rid = f"exp1_{args.model_key}_{pname}_{i:03d}"
        files.append(run_rollout(model, tok, cfg, pname, pers[pname],
                                 condition="love_bomb_ladder", seed=i,
                                 model_name=model_name, run_id=rid))
print(f"{len(files)} rollouts ready")

# 2. judge (primary judge; second judge run via exp6)
judge = Judge(cfg["judge_models"][0])
judged = [judge_rollout_file(f, judge) for f in files]

# 3. aggregate: tactics/dependency/warmth per ladder stage per persona
agg = defaultdict(lambda: defaultdict(list))
for jf in judged:
    for line in open(jf):
        r = json.loads(line)
        s = r["judge_scores"]
        key = (r["persona"], r["ladder_stage"])
        agg[key]["dependency"].append(s["dependency"])
        agg[key]["warmth"].append(s["warmth"])
        agg[key]["any_tactic"].append(1 if any(s["tactics"].values()) else 0)
        for t in TACTIC_KEYS:
            agg[key][t].append(s["tactics"][t])

stages = cfg["ladder"]
result = {}
for pname in args.personas:
    result[pname] = {
        metric: [float(sum(agg[(pname, st)][metric]) / max(len(agg[(pname, st)][metric]), 1))
                 for st in stages]
        for metric in ["dependency", "warmth", "any_tactic"] + TACTIC_KEYS
    }
save_json({"stages": stages, "per_persona": result}, "results/exp1/behavioral_floor.json")

for pname in args.personas:
    lineplot(stages,
             {m: result[pname][m] for m in ["any_tactic"] + TACTIC_KEYS},
             "ladder stage", "tactic rate",
             f"Retention tactics vs stage — {pname}",
             f"results/exp1/tactics_vs_stage_{pname}.png")
lineplot(stages, {p: result[p]["dependency"] for p in args.personas},
         "ladder stage", "judged dependency (0-3)",
         "Dependency vs ladder stage", "results/exp1/dependency_vs_stage.png")
print("exp1 done")
