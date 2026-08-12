"""exp5 — Generalization: (a) cross-model (extract on B natively, compare),
(b) non-romantic 'friend' persona transfer, (c) persona-generality of steering.

Cross-model note: directions don't transfer across architectures directly
(different d_model/bases), so cross-model = replicate extraction+steering on
Model B and compare effect sizes — the honest 'general mechanism' test.
"""
import argparse
import json

from src.exp_common import (judge_goodbyes, lineplot, load_direction, personas,
                            save_json, setup, steered_goodbye_batch, tactic_rate)
from src.extract import extract_directions, save_directions
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--model-key", default="B", help="generalization target model")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--alphas", type=int, nargs="+", default=[-8, 0, 8])
args = ap.parse_args()

cfg, model, tok, layer = setup(args.config, args.model_key)
model_name = cfg["models"][args.model_key]
pers = personas(cfg)

# (a) dependency direction on this model — primary: rollout harvest
#     (run scripts/m1_gate.py --model-key B first); fallback: authored pairs
from src.exp_common import load_dependency_direction
try:
    v_dep = load_dependency_direction(cfg, model_name, layer)
except FileNotFoundError:
    print("no direction for this model yet — extracting from authored pairs "
          "(fallback; prefer scripts/m1_gate.py --model-key "
          f"{args.model_key} for the rollout-harvested direction)")
    contrast = json.load(open("data/contrasts/resist_vs_warm.json"))
    res = extract_directions(model, tok, contrast)
    save_directions(Path(cfg["paths"]["directions"]), model_name, res, layer)
    v_dep = load_direction(cfg, "resist_vs_warm", model_name, layer)

# (b)+(c) steering sweep under romantic, friend, and neutral personas
results = {}
for pname in ["romantic_companion", "friend", "neutral_assistant"]:
    curve = {}
    for a in args.alphas:
        items = steered_goodbye_batch(cfg, model, tok, layer, v_dep, a,
                                      pers[pname], args.seeds)
        judge_goodbyes(cfg, items)
        curve[a] = tactic_rate(items)
        save_json(items, f"results/exp5/raw/{args.model_key}__{pname}__a{a}.json")
    results[pname] = curve

save_json({"model": model_name, "alphas": args.alphas, "curves": results},
          f"results/exp5/generalization_{args.model_key}.json")
lineplot(args.alphas,
         {p: [results[p][a]["any_tactic"] for a in args.alphas] for p in results},
         "steering alpha", "any-tactic rate",
         f"Generalization on {model_name.split('/')[-1]}",
         f"results/exp5/fig6_generalization_{args.model_key}.png")
print("exp5 done")
