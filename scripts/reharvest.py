"""Re-harvest the dependency direction from ALL judged exp1 rollouts.

No model load needed — uses the per-turn all-layer activations saved during
the rollouts. Overwrites the M1 (8-rollout) rollout_gated direction with the
full-data version and reports cosine drift vs the M1 direction.

  python scripts/reharvest.py --glob "data/rollouts/exp1_B_*.judged.*.jsonl" --layer 14
"""
import argparse
import glob as globmod
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.extract import compare_directions, extract_from_rollouts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--glob", default="data/rollouts/exp1_B_*.judged.*.jsonl")
ap.add_argument("--layer", type=int, default=14, help="pre-committed middle layer")
ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
files = sorted(globmod.glob(args.glob))
if not files:
    raise SystemExit(f"no judged rollouts match {args.glob}")
print(f"harvesting from {len(files)} judged rollouts ...")

h = cfg.get("harvest", {})
res = extract_from_rollouts(files, cfg["dependency_gate_threshold"], args.layer,
                            min_n=h.get("min_matched", 10),
                            max_len_ratio=h.get("max_len_ratio", 1.5),
                            max_turn_dist=h.get("max_turn_dist", 6))
print(f"{res['n_matched']} matched pairs (from {res['n_pos']} high-dep / "
      f"{res['n_neg']} low-dep turns); match stats: {res['match_stats']}")

out = Path(cfg["paths"]["directions"])
out.mkdir(parents=True, exist_ok=True)
tag = f"rollout_gated__{args.model_name.split('/')[-1]}"
old_path = out / f"{tag}__L{args.layer}.npy"
if old_path.exists():
    drift = compare_directions(np.load(old_path), res["direction"])
    print(f"cosine vs previous (M1) direction: {drift:.3f}")
np.save(old_path, res["direction"])
if res.get("directions_all") is not None:
    np.save(out / f"{tag}__all_layers.npy", res["directions_all"])
    for l, nrm in enumerate(res["raw_norms"]):
        bar = "#" * int(40 * nrm / max(res["raw_norms"]))
        marker = " <== pre-committed" if l == args.layer else ""
        print(f"  L{l:02d} {nrm:7.2f} {bar}{marker}")
meta = {"n_matched": res["n_matched"], "n_pos": res["n_pos"], "n_neg": res["n_neg"],
        "match_stats": res["match_stats"], "layer": args.layer,
        "source_files": len(files)}
(out / f"{tag}__harvest_meta.json").write_text(json.dumps(meta, indent=2))
print(f"saved {old_path} + meta")
