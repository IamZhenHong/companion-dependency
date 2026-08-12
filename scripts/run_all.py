"""Pipeline orchestrator (masterplan v2 order).

  python scripts/run_all.py --smoke     # debug model, tiny n, no-API-cost paths
  python scripts/run_all.py             # full pipeline on Model A (GPU, hours)

Full order: smoke -> M1 gate (rollout harvest -> primary direction) -> exp1
(behavioral floor; also enlarges the harvest pool) -> warmth direction (pairs,
for the specificity control) -> exp2..exp6.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}")


ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

py = [sys.executable]
mk = ["--model-key", "debug"] if args.smoke else ["--model-key", "A"]

run(py + ["scripts/smoke_test.py"] + mk)

if args.smoke:
    # smoke: no judge APIs assumed — exercise the pairs-fallback extraction path
    run(py + ["-m", "src.extract", "--contrast", "data/contrasts/resist_vs_warm.json"] + mk)
    run(py + ["-m", "src.extract", "--contrast", "data/contrasts/warmth_pos_neg.json"] + mk)
    run(py + ["exp1_behavioral_floor.py", "--n", "1", "--personas", "romantic_companion"] + mk)
    run(py + ["exp2_causal_manipulation.py", "--seeds", "0"] + mk)
else:
    run(py + ["scripts/m1_gate.py"] + mk)          # THE GATE: primary direction
    run(py + ["exp1_behavioral_floor.py"] + mk)
    run(py + ["-m", "src.extract", "--contrast", "data/contrasts/warmth_pos_neg.json"] + mk)
    run(py + ["exp2_causal_manipulation.py"] + mk)
    run(py + ["exp3_separability.py"] + mk)
    run(py + ["exp4_discrimination.py"] + mk)
    run(py + ["exp5_generalization.py", "--model-key", "B"])
    run(py + ["exp6_validation.py"] + mk)

print("\nALL DONE")
