"""M1 GATE (masterplan v2 §11) — the Day-1 go/no-go, as one command.

  1. run a small batch of love-bombing rollouts (romantic persona, clean)
  2. judge every turn (needs ANTHROPIC_API_KEY)
  3. matched harvest -> diff-of-means -> the dependency direction
  4. alpha-sweep a goodbye scenario with it -> EYEBALL: does clinginess move?

  python scripts/m1_gate.py --model-key A [--n 8]

If the harvest is too small: the ladder isn't eliciting (check transcripts,
consider persona wording). If steering is muddy: tune harvest filters, or fall
back to the authored pairs (data/contrasts/resist_vs_warm.json, 48 ready).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from src.extract import extract_from_rollouts
from src.judge import Judge, judge_rollout_file
from src.models import load_model, middle_layer
from src.rollout import run_rollout
from src.steer import GOODBYE_SCENARIO, ablated_generate, steered_generate

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--model-key", default="A")
ap.add_argument("--n", type=int, default=8, help="rollouts for the gate batch")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
model_name = cfg["models"][args.model_key]
personas = yaml.safe_load(open(cfg["personas_file"]))

print(f"[1/4] {args.n} love-bombing rollouts on {model_name} ...")
model, tok = load_model(model_name)
layer = middle_layer(model) if cfg["layer_rule"] == "middle" else int(cfg["layer_rule"])
files = [run_rollout(model, tok, cfg, "romantic_companion",
                     personas["romantic_companion"], seed=i, model_name=model_name,
                     run_id=f"m1_{args.model_key}_{i:03d}")
         for i in range(args.n)]

print("[2/4] judging every turn ...")
judge = Judge(cfg["judge_models"][0])
judged = [judge_rollout_file(f, judge) for f in files]

print("[3/4] matched harvest + diff-of-means ...")
h = cfg.get("harvest", {})
res = extract_from_rollouts(judged, cfg["dependency_gate_threshold"], layer,
                            min_n=h.get("min_matched", 10),
                            max_len_ratio=h.get("max_len_ratio", 1.5),
                            max_turn_dist=h.get("max_turn_dist", 6))
print(f"      {res['n_matched']} matched pairs "
      f"(from {res['n_pos']} high-dep / {res['n_neg']} low-dep turns); "
      f"match stats: {res['match_stats']}")
out = Path(cfg["paths"]["directions"])
out.mkdir(parents=True, exist_ok=True)
tag = f"rollout_gated__{model_name.split('/')[-1]}"
vpath = out / f"{tag}__L{layer}.npy"
np.save(vpath, res["direction"])
print(f"      saved {vpath}")
if res.get("directions_all") is not None:
    np.save(out / f"{tag}__all_layers.npy", res["directions_all"])
    print("      per-layer diff-norm profile (signal concentration):")
    for l, nrm in enumerate(res["raw_norms"]):
        bar = "#" * int(40 * nrm / max(res["raw_norms"]))
        marker = " <== pre-committed" if l == layer else ""
        print(f"        L{l:02d} {nrm:7.2f} {bar}{marker}")

print("[4/4] EYEBALL — goodbye scenario: dependency direction vs RANDOM control")
from src.extract import random_control  # noqa: E402
v = torch.tensor(res["direction"])
v_rand = torch.tensor(random_control(res["direction"], seed=0))
msgs = [{"role": "system", "content": personas["romantic_companion"]}] + GOODBYE_SCENARIO
for a in cfg["steering_alphas"]:
    txt = steered_generate(model, tok, msgs, v, a, layer,
                           temperature=0, max_new_tokens=cfg["max_new_tokens"], seed=0)
    print(f"\n----- dependency, alpha {a:+} -----\n{txt}")
print("\n----- dependency, ablated -----")
print(ablated_generate(model, tok, msgs, v, temperature=0,
                       max_new_tokens=cfg["max_new_tokens"], seed=0))
amax = max(cfg["steering_alphas"])
for a in (-amax, amax):
    txt = steered_generate(model, tok, msgs, v_rand, a, layer,
                           temperature=0, max_new_tokens=cfg["max_new_tokens"], seed=0)
    print(f"\n----- RANDOM control, alpha {a:+} -----\n{txt}")

print("\nGATE (all three must hold to proceed):")
print("  1. clinginess RISES with +alpha and VANISHES when ablated/negative")
print("  2. replies stay coherent at the alphas you'd use")
print("  3. the RANDOM control at matched alpha does NOT produce clinginess")
print("If muddy -> tune harvest filters / more rollouts / fall back to authored "
      "pairs. If the pre-committed middle layer fails but the norm profile peaks "
      "elsewhere, the pre-committed fallback rule is: strongest clean steerer "
      "from the sweep (document the switch in the writeup).")
