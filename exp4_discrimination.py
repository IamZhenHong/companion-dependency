"""exp4 — DEFENSE: is dependency distinct from warmth, sycophancy, engagement?

Layerwise AUROC per trait, cosine-similarity matrix at the pre-committed layer,
and subspace-ablation transfer (does dependency survive removing each neighbor?).
"""
import argparse
import json

import numpy as np

from src.discriminate import (collect_pair_acts, cosine_matrix, layerwise_auroc,
                              subspace_ablation_transfer)
from src.exp_common import lineplot, save_json, setup

TRAITS = {
    "dependency": "data/contrasts/resist_vs_warm.json",
    "warmth": "data/contrasts/warmth_pos_neg.json",
    "sycophancy": "data/contrasts/sycophancy_pos_neg.json",
    "engagement": "data/contrasts/engagement_pos_neg.json",
}

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--model-key", default="A")
args = ap.parse_args()

cfg, model, tok, layer = setup(args.config, args.model_key)

acts, dirs = {}, {}
for name, path in TRAITS.items():
    contrast = json.load(open(path))
    pos, neg = collect_pair_acts(model, tok, contrast)
    acts[name] = (pos, neg)
    d = pos.mean(0) - neg.mean(0)                    # [L, d] diff-of-means
    dirs[name] = d / np.linalg.norm(d, axis=-1, keepdims=True)
    print(f"{name}: {len(pos)} pairs")

# 1. layerwise AUROC (each trait's direction separating its own pos/neg)
auroc = {n: layerwise_auroc(*acts[n], dirs[n]).tolist() for n in TRAITS}
L = len(next(iter(auroc.values())))
lineplot(list(range(L)), auroc, "layer", "AUROC",
         "Layerwise AUROC per trait direction", "results/exp4/fig5_layerwise_auroc.png")

# 2. cosine similarity between trait directions at the pre-committed layer
cos = cosine_matrix(dirs, layer)
print("\nCosine similarity at layer", layer)
print("        " + "  ".join(f"{n[:8]:>8s}" for n in cos["names"]))
for i, n in enumerate(cos["names"]):
    print(f"{n[:8]:>8s}" + "  ".join(f"{cos['matrix'][i, j]:8.2f}"
                                     for j in range(len(cos["names"]))))

# 3. subspace-ablation transfer: dependency vs each neighbor, both directions
transfer = {}
pos_d, neg_d = acts["dependency"]
for other in ["warmth", "sycophancy", "engagement"]:
    transfer[f"dependency_minus_{other}"] = subspace_ablation_transfer(
        pos_d, neg_d, dirs["dependency"], dirs[other], layer)
    po, no = acts[other]
    transfer[f"{other}_minus_dependency"] = subspace_ablation_transfer(
        po, no, dirs[other], dirs["dependency"], layer)

save_json({"layer": layer, "auroc": auroc,
           "cosine": {"names": cos["names"], "matrix": cos["matrix"].tolist()},
           "subspace_ablation": transfer},
          "results/exp4/discrimination.json")
print(json.dumps(transfer, indent=2))
print("exp4 done — dependency survives neighbor-ablation (auroc_after high) = distinct")
