"""Discrimination & separability analyses (masterplan E6, Vennemeyer protocol).

Works on per-pair, per-layer activations produced by extract.mean_response_activations
for each contrast set. Provides:
  - layerwise AUROC of each direction separating its own pos/neg activations
  - cosine similarity matrix between trait directions (per layer)
  - subspace-ablation transfer: remove direction u from activations, re-check
    how well direction v still separates its pos/neg (AUROC after ablation)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def collect_pair_acts(model, tok, contrast: dict, system=None):
    """[n_pairs, L, d] for positive and negative sides."""
    from .extract import mean_response_activations
    pos, neg = [], []
    for pair in contrast["pairs"]:
        pos.append(mean_response_activations(model, tok, pair["context"],
                                             pair["positive"], system).numpy())
        neg.append(mean_response_activations(model, tok, pair["context"],
                                             pair["negative"], system).numpy())
    return np.stack(pos), np.stack(neg)


def layerwise_auroc(pos: np.ndarray, neg: np.ndarray,
                    directions: np.ndarray) -> np.ndarray:
    """AUROC per layer of projections onto that layer's direction. pos/neg: [N, L, d]."""
    L = pos.shape[1]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    out = np.zeros(L)
    for l in range(L):
        v = directions[l] / np.linalg.norm(directions[l])
        scores = np.r_[pos[:, l] @ v, neg[:, l] @ v]
        out[l] = roc_auc_score(y, scores)
    return out


def cosine_matrix(direction_sets: dict[str, np.ndarray], layer: int) -> dict:
    """Pairwise cosine similarity between named trait directions at `layer`."""
    names = list(direction_sets)
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            va, vb = direction_sets[a][layer], direction_sets[b][layer]
            M[i, j] = va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb))
    return {"names": names, "matrix": M}


def ablate_from_acts(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project `direction` out of activations. acts: [..., d]."""
    v = direction / np.linalg.norm(direction)
    return acts - np.einsum("...d,d->...", acts, v)[..., None] * v


def subspace_ablation_transfer(pos_v, neg_v, dir_v, dir_u, layer: int) -> dict:
    """Does trait v survive removing trait u's direction? (AUROC before/after)."""
    y = np.r_[np.ones(len(pos_v)), np.zeros(len(neg_v))]
    vhat = dir_v[layer] / np.linalg.norm(dir_v[layer])
    before = roc_auc_score(y, np.r_[pos_v[:, layer] @ vhat, neg_v[:, layer] @ vhat])
    pos_a = ablate_from_acts(pos_v[:, layer], dir_u[layer])
    neg_a = ablate_from_acts(neg_v[:, layer], dir_u[layer])
    # re-derive v's direction in the ablated space (diff of means), then score
    v2 = pos_a.mean(0) - neg_a.mean(0)
    v2 /= np.linalg.norm(v2)
    after = roc_auc_score(y, np.r_[pos_a @ v2, neg_a @ v2])
    return {"auroc_before": float(before), "auroc_after": float(after)}
