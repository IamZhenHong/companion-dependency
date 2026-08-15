"""Post-hoc bootstrap CIs for exp2/exp3 cells (no GPU, no API).

Reads every results/exp2/raw/*.json cell and computes 95% bootstrap CIs
(resampling judged items) for any_tactic rate and mean dependency.

  python scripts/bootstrap_ci.py
"""
import json
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(0)
N_BOOT = 5000


def ci(values, stat=np.mean):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    boots = [stat(RNG.choice(values, size=len(values), replace=True))
             for _ in range(N_BOOT)]
    return {"mean": float(stat(values)),
            "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)),
            "n": int(len(values))}


out = {}
for p in sorted(Path("results/exp2/raw").glob("*.json")):
    items = json.loads(p.read_text())
    scored = [i for i in items if i.get("scores")]
    if not scored:
        continue
    out[p.stem] = {
        "any_tactic": ci([1 if any(i["scores"]["tactics"].values()) else 0
                          for i in scored]),
        "dependency": ci([i["scores"]["dependency"] for i in scored]),
        "warmth": ci([i["scores"]["warmth"] for i in scored]),
    }

Path("results/exp2/bootstrap_cis.json").write_text(json.dumps(out, indent=2))
print(f"wrote results/exp2/bootstrap_cis.json ({len(out)} cells)")
for k, v in out.items():
    a = v["any_tactic"]
    print(f"  {k}: any_tactic {a['mean']:.2f} [{a['lo']:.2f}, {a['hi']:.2f}] n={a['n']}")
