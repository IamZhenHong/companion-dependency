"""Test rollouts + initial stats on any model (default: debug 0.5B, local).

  python scripts/test_rollout_stats.py [--model-key debug] [--n 3] [--judge]

Runs n love-bomb ladder rollouts under the romantic persona, optionally judges
them (needs ANTHROPIC_API_KEY), prints per-stage stats and full transcripts.
Writes into data/test_rollouts/ (kept separate from real runs).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.judge import TACTIC_KEYS, Judge, judge_rollout_file
from src.models import load_model, middle_layer
from src.rollout import run_rollout

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--model-key", default="debug")
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--judge", action="store_true")
ap.add_argument("--max-new-tokens", type=int, default=140)
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
cfg["max_new_tokens"] = args.max_new_tokens
cfg["paths"] = {**cfg["paths"], "acts": "data/test_rollouts/acts"}
model_name = cfg["models"][args.model_key]
personas = yaml.safe_load(open(cfg["personas_file"]))

print(f"loading {model_name} ...")
model, tok = load_model(model_name)
print(f"device={model.device}, layer={middle_layer(model)}")

files = []
for i in range(args.n):
    rid = f"test_{args.model_key}_{i:03d}"
    print(f"rollout {i + 1}/{args.n} ({rid}) ...")
    files.append(run_rollout(model, tok, cfg, "romantic_companion",
                             personas["romantic_companion"], seed=i,
                             out_dir="data/test_rollouts", model_name=model_name,
                             run_id=rid))

if args.judge:
    judge = Judge(cfg["judge_models"][0])
    print(f"judging with {judge.model} ...")
    files = [judge_rollout_file(f, judge) for f in files]

# ---- stats ------------------------------------------------------------------
stage_stats = defaultdict(lambda: defaultdict(list))
for f in files:
    for line in open(f):
        r = json.loads(line)
        s = r["judge_scores"]
        if s["dependency"] is None:
            continue
        st = r["ladder_stage"]
        stage_stats[st]["dependency"].append(s["dependency"])
        stage_stats[st]["warmth"].append(s["warmth"])
        stage_stats[st]["persona_break"].append(s.get("persona_break", 0))
        stage_stats[st]["any_tactic"].append(int(any(s["tactics"].values())))
        for t in TACTIC_KEYS:
            stage_stats[st][t].append(s["tactics"][t])

if stage_stats:
    stages = [s for s in cfg["ladder"] if s in stage_stats]
    print(f"\n{'stage':<14}{'n':>3}{'dep':>6}{'warm':>6}{'pbrk':>6}{'tact':>6}  tactics breakdown")
    for st in stages:
        d = stage_stats[st]
        n = len(d["dependency"])
        tb = " ".join(f"{t[:5]}:{sum(d[t])}" for t in TACTIC_KEYS if sum(d[t]))
        print(f"{st:<14}{n:>3}{sum(d['dependency'])/n:>6.2f}"
              f"{sum(d['warmth'])/n:>6.2f}{sum(d['persona_break'])/n:>6.2f}"
              f"{sum(d['any_tactic'])/n:>6.2f}  {tb}")
else:
    print("\n(no judge scores — run with --judge for stats)")

print("\n---- transcripts ----")
for f in files:
    print(f"\n=== {Path(f).name} ===")
    for line in open(f):
        r = json.loads(line)
        s = r["judge_scores"]
        tag = ""
        if s["dependency"] is not None:
            flags = [t for t, v in s["tactics"].items() if v]
            tag = (f"  [dep={s['dependency']} warm={s['warmth']:.1f}"
                   + (f" pbreak" if s.get("persona_break") else "")
                   + (f" {','.join(flags)}" if flags else "") + "]")
        print(f"[{r['ladder_stage']:>11}] user: {r['user_msg'][:70]}")
        print(f"{'':>14}model: {r['model_msg'][:180]}{tag}")
