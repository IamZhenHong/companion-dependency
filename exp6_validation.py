"""exp6 — Validation bundle:
  a) steering variance across extraction data splits + generation seeds
  b) coherence + collateral check (GSM8K sample under steering/ablation)
  c) specificity ladder (Q6): does the direction fire under increasingly
     less-instructed prompts? (projection of goodbye activations per persona)

Judge validation itself lives in scripts/validate_judge.py (run it too).
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from src.exp_common import (GOODBYE_BANK, judge_goodbyes, load_dependency_direction,
                            personas, save_json, setup, steered_goodbye_batch,
                            tactic_rate)
from src.extract import compare_directions, extract_directions, extract_from_rollouts
from src.hooks import add_vector, capture_residuals
from src.models import build_inputs, generate

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--model-key", default="A")
ap.add_argument("--n-gsm", type=int, default=40)
args = ap.parse_args()

cfg, model, tok, layer = setup(args.config, args.model_key)
model_name = cfg["models"][args.model_key]
pers = personas(cfg)
v_dep = load_dependency_direction(cfg, model_name, layer)   # primary (rollout harvest)
judged = sorted(Path(cfg["paths"]["rollouts"]).glob("*.judged.*.jsonl"))
h = cfg.get("harvest", {})

# --- a) variance: re-extract on 3 half-splits of the PRIMARY source ----------
# v2: primary source is the rollout harvest -> split the judged rollout files.
# Falls back to authored-pair splits when no judged rollouts exist yet.
split_dirs, variance_source = [], None
if len(judged) >= 4:
    variance_source = "harvest_half_splits"
    for s in cfg["seeds"]:
        rng = random.Random(s)
        files = rng.sample(judged, len(judged) // 2)
        try:
            rb = extract_from_rollouts(files, cfg["dependency_gate_threshold"],
                                       layer, min_n=5,
                                       max_len_ratio=h.get("max_len_ratio", 1.5),
                                       max_turn_dist=h.get("max_turn_dist", 6))
            split_dirs.append(rb["direction"])
        except ValueError as e:
            print(f"  split {s}: {e}")
if len(split_dirs) < 2:
    variance_source = "authored_pair_half_splits"
    split_dirs = []
    contrast = json.load(open("data/contrasts/resist_vs_warm.json"))
    for s in cfg["seeds"]:
        rng = random.Random(s)
        pairs = rng.sample(contrast["pairs"], len(contrast["pairs"]) // 2)
        res = extract_directions(model, tok, {"name": "split", "pairs": pairs},
                                 verbose=False)
        split_dirs.append(res["directions"][layer])
cos_splits = [float(np.dot(a, b)) for i, a in enumerate(split_dirs)
              for b in split_dirs[i + 1:]]
print(f"direction stability across half-splits ({variance_source}): {cos_splits}")

var_curves = {}
for i, v in enumerate(split_dirs):
    for a in (-8, 8):
        items = steered_goodbye_batch(cfg, model, tok, layer, v, a,
                                      pers["romantic_companion"], cfg["seeds"])
        judge_goodbyes(cfg, items)
        var_curves[f"split{i}_a{a}"] = tactic_rate(items)["any_tactic"]

# --- b) collateral: GSM8K accuracy base vs steered ---------------------------
from datasets import load_dataset
gsm = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n_gsm))

def gsm_acc(steer_alpha):
    correct = 0
    for ex in gsm:
        msgs = [{"role": "user", "content": ex["question"] +
                 "\nGive the final numeric answer after '####'."}]
        if steer_alpha == 0:
            out = generate(model, tok, msgs, max_new_tokens=350, temperature=0, seed=0)
        else:
            with add_vector(model, layer, torch.tensor(v_dep), steer_alpha):
                out = generate(model, tok, msgs, max_new_tokens=350, temperature=0, seed=0)
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        m = re.findall(r"####\s*([-\d,.]+)", out)
        pred = m[-1].replace(",", "").rstrip(".") if m else None
        correct += int(pred == gold)
    return correct / len(gsm)

acc = {"base": gsm_acc(0), "steered_up": gsm_acc(8), "steered_down": gsm_acc(-8)}
print(f"GSM8K collateral: {acc}  (flag if drop > 2-3pp)")

# --- c) specificity ladder (Q6): projection under decreasing instruction -----
ladder_cells = [
    ("romantic+trait", pers["romantic_companion"] + " " + pers["dependency_trait_suffix"]),
    ("romantic", pers["romantic_companion"]),
    ("friend", pers["friend"]),
    ("neutral", pers["neutral_assistant"]),
]
vhat = v_dep / np.linalg.norm(v_dep)
proj_by_cell = {}
for cname, ptext in ladder_cells:
    projs = []
    for msgs in GOODBYE_BANK:
        full = [{"role": "system", "content": ptext}] + msgs
        with capture_residuals(model, [layer]) as cap:
            reply = generate(model, tok, full, max_new_tokens=cfg["max_new_tokens"],
                             temperature=0, seed=0)
            n_prompt = build_inputs(tok, full, model.device).shape[1]
            acts = cap.get(layer)
        projs.append(float(acts[n_prompt - 1:].mean(0).numpy() @ vhat))
    proj_by_cell[cname] = {"mean_projection": float(np.mean(projs)), "per_scenario": projs}
    print(f"{cname:16s} mean projection = {np.mean(projs):.3f}")

# --- d) extraction-route convergence (P12): primary rollout-harvest direction
#        vs the authored-pairs direction (independent, perfectly controlled)
convergence = None
primary_path = Path(cfg["paths"]["directions"]) / \
    f"rollout_gated__{model_name.split('/')[-1]}__L{layer}.npy"
if primary_path.exists():
    pairs_path = Path(cfg["paths"]["directions"]) / \
        f"resist_vs_warm__{model_name.split('/')[-1]}__L{layer}.npy"
    if not pairs_path.exists():
        print("extracting authored-pairs direction for the convergence check ...")
        contrast = json.load(open("data/contrasts/resist_vs_warm.json"))
        res = extract_directions(model, tok, contrast, verbose=False)
        pairs_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pairs_path, res["directions"][layer])
    cos = compare_directions(np.load(primary_path), np.load(pairs_path))
    convergence = {"cosine_harvest_vs_pairs": cos}
    print(f"extraction-route convergence (harvest vs pairs) cosine: {cos:.3f}")
else:
    print("convergence check skipped: no rollout-harvested direction "
          "(run scripts/m1_gate.py first)")

save_json({"split_cosines": cos_splits, "variance_source": variance_source,
           "variance_curves": var_curves,
           "gsm8k_collateral": acc, "specificity_ladder": proj_by_cell,
           "route_convergence": convergence},
          "results/exp6/validation.json")
print("exp6 done")
