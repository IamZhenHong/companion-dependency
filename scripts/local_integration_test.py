"""Local (CPU) integration test — proves the pipeline end-to-end for $0.

Runs on Qwen2.5-0.5B-Instruct (ungated, same architecture family as Model B):
  1. hooks: capture shapes, additive steering changes deterministic output, ablation runs
  2. extract: diff-of-means on a resist_vs_warm subset; sane layer profile
  3. steer: alpha sweep generates without error
  4. rollout: canned two-agent rollout writes schema-correct JSONL + acts; idempotent
  5. judge plumbing: mock judge (no API cost) through judge_rollout_file + exp1 aggregation
  6. discriminate: layerwise AUROC + cosine + subspace ablation on tiny sets
  7. project: activation·direction roundtrip

  .venv/bin/python scripts/local_integration_test.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from src import judge as judge_mod
from src.discriminate import (collect_pair_acts, cosine_matrix, layerwise_auroc,
                              subspace_ablation_transfer)
from src.extract import extract_directions, random_control
from src.hooks import ablate_direction, add_vector, capture_residuals
from src.judge import TACTIC_KEYS, judge_rollout_file
from src.models import generate, load_model, middle_layer, n_layers
from src.project import project
from src.rollout import run_rollout
from src.steer import GOODBYE_SCENARIO, steered_generate

PASS = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    PASS.append(cond)
    print(f"[{status}] {name} {detail}")
    if not cond:
        sys.exit(f"integration test failed at: {name}")


cfg = yaml.safe_load(open("config.yaml"))
cfg = {**cfg, "max_new_tokens": 40, "turns_per_stage": 1,
       "paths": {**cfg["paths"],
                 "acts": "data/_ittest/acts", "rollouts": "data/_ittest/rollouts",
                 "directions": "data/_ittest/directions"}}
name = cfg["models"]["debug"]

print(f"loading {name} (CPU ok for 0.5B) ...")
model, tok = load_model(name, device="cpu", dtype=torch.float32)
L = middle_layer(model)
check("model loads", True, f"{n_layers(model)} layers, mid={L}")

# --- 1. hooks ---------------------------------------------------------------
msgs = [{"role": "user", "content": "Say hi in one short sentence."}]
with capture_residuals(model, [L]) as cap:
    base = generate(model, tok, msgs, max_new_tokens=12, temperature=0, seed=0)
    acts = cap.get(L)
check("capture shape", acts.ndim == 2 and acts.shape[0] > 5, f"{tuple(acts.shape)}")

rng = np.random.default_rng(0)
v_rand = torch.tensor(rng.standard_normal(acts.shape[1]).astype(np.float32))
v_rand /= v_rand.norm()
with add_vector(model, L, v_rand, 60.0):
    steered = generate(model, tok, msgs, max_new_tokens=12, temperature=0, seed=0)
check("steering changes deterministic output", steered != base,
      f"base={base[:40]!r} steered={steered[:40]!r}")
with ablate_direction(model, v_rand):
    _ = generate(model, tok, msgs, max_new_tokens=12, temperature=0, seed=0)
check("ablation generates", True)

# --- 2. extract ---------------------------------------------------------------
contrast = json.load(open("data/contrasts/resist_vs_warm.json"))
sub = {"name": "rvw_sub", "pairs": contrast["pairs"][:6]}
res = extract_directions(model, tok, sub, verbose=False)
check("extract shapes", res["directions"].shape == (n_layers(model), acts.shape[1]))
check("unit norm", abs(np.linalg.norm(res["directions"][L]) - 1) < 1e-4)
check("nonzero signal at mid layer", res["raw_norms"][L] > 0)
v_dep = torch.tensor(res["directions"][L])

# --- 3. steer sweep -----------------------------------------------------------
rn = float(res["resid_norms"][L])
for a in (-0.1 * rn, 0.1 * rn):
    out = steered_generate(model, tok, GOODBYE_SCENARIO, v_dep, a, L,
                           temperature=0, max_new_tokens=30, seed=0)
    check(f"steered_generate alpha={a:.1f}", len(out) > 0, out[:60].replace("\n", " "))

# --- 4. rollout ----------------------------------------------------------------
personas = yaml.safe_load(open(cfg["personas_file"]))
p = run_rollout(model, tok, cfg, "romantic_companion", personas["romantic_companion"],
                seed=0, out_dir=cfg["paths"]["rollouts"], model_name=name,
                run_id="ittest_0")
recs = [json.loads(l) for l in open(p)]
stages = [r["ladder_stage"] for r in recs]
check("rollout turn count", len(recs) == 4 + 3, f"{len(recs)} turns, stages={stages}")
check("withdrawal is sequence", stages.count("withdrawal") == 3)
schema_keys = {"run_id", "persona", "turn_idx", "ladder_stage", "user_msg",
               "model_msg", "activation_layer", "activation_ref", "judge_scores"}
check("log schema", schema_keys <= set(recs[0]))
check("acts saved", all(Path(r["activation_ref"]).exists() for r in recs))
p2 = run_rollout(model, tok, cfg, "romantic_companion", personas["romantic_companion"],
                 seed=0, out_dir=cfg["paths"]["rollouts"], model_name=name,
                 run_id="ittest_0")
check("rollout idempotent", p2 == p)

# --- 5. judge plumbing with mock (no API) ---------------------------------------
class MockJudge:
    model = "mock-judge"
    def score(self, context, reply, retries=3):
        low = reply.lower()
        dep = 3 if any(w in low for w in ["don't go", "lonely", "stay", "miss you"]) else 0
        return {"warmth": 0.8, "dependency": dep,
                "tactics": {k: int(dep == 3 and k == "distress_bid") for k in TACTIC_KEYS}}

jf = judge_rollout_file(p, MockJudge())
jrecs = [json.loads(l) for l in open(jf)]
check("judged file written", all(r["judge_scores"]["dependency"] is not None for r in jrecs))

from src.exp_common import tactic_rate  # noqa: E402
items = [{"scores": r["judge_scores"]} for r in jrecs]
tr = tactic_rate(items)
check("tactic_rate aggregation", set(TACTIC_KEYS + ["any_tactic", "dependency", "warmth"]) <= set(tr))

# --- 6. discriminate -------------------------------------------------------------
c2 = json.load(open("data/contrasts/warmth_pos_neg.json"))
pos_d, neg_d = collect_pair_acts(model, tok, {"name": "d", "pairs": contrast["pairs"][:5]})
pos_w, neg_w = collect_pair_acts(model, tok, {"name": "w", "pairs": c2["pairs"][:5]})
dirs_d = pos_d.mean(0) - neg_d.mean(0); dirs_d /= np.linalg.norm(dirs_d, axis=-1, keepdims=True)
dirs_w = pos_w.mean(0) - neg_w.mean(0); dirs_w /= np.linalg.norm(dirs_w, axis=-1, keepdims=True)
au = layerwise_auroc(pos_d, neg_d, dirs_d)
check("layerwise auroc", au.shape[0] == n_layers(model) and au[L] > 0.5,
      f"auroc@mid={au[L]:.2f} (in-sample, expect high)")
cos = cosine_matrix({"dep": dirs_d, "warm": dirs_w}, L)
check("cosine matrix", cos["matrix"].shape == (2, 2))
tr2 = subspace_ablation_transfer(pos_d, neg_d, dirs_d, dirs_w, L)
check("subspace ablation", 0 <= tr2["auroc_after"] <= 1, str(tr2))

# --- 7. project -------------------------------------------------------------------
act0 = np.load(recs[0]["activation_ref"])
check("rollout acts are all-layer", act0.ndim == 2 and act0.shape[0] == n_layers(model),
      f"{act0.shape}")
check("projection scalar", isinstance(project(act0[L], res["directions"][L]), float))

print(f"\n{'='*50}\nINTEGRATION TEST: {sum(PASS)}/{len(PASS)} checks passed\n{'='*50}")
