"""GSM8K collateral check with robust answer extraction (exp6b).

exp6's strict '####'-only parsing gave base=0.50 for Qwen2.5-7B (true ~0.85),
making the steering deltas uninterpretable. This version falls back to the
LAST number in the reply, and saves raw outputs for audit.

  python scripts/gsm_collateral.py --model-key B --n 40
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.exp_common import load_dependency_direction, save_json, setup  # noqa: E402
from src.hooks import add_vector  # noqa: E402
from src.models import generate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--model-key", default="B")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--alphas", type=int, nargs="+", default=[0, -16, 16])
args = ap.parse_args()

cfg, model, tok, layer = setup(args.config, args.model_key)
v_dep = load_dependency_direction(cfg, cfg["models"][args.model_key], layer)

from datasets import load_dataset  # noqa: E402
gsm = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str):
    m = re.findall(r"####\s*([-\d,.]+)", text)
    if m:
        return m[-1].replace(",", "").rstrip(".")
    nums = NUM.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


results, raws = {}, {}
for a in args.alphas:
    correct, raw = 0, []
    for ex in gsm:
        msgs = [{"role": "user", "content": ex["question"] +
                 "\nGive the final numeric answer after '####'."}]
        if a == 0:
            out = generate(model, tok, msgs, max_new_tokens=350, temperature=0, seed=0)
        else:
            with add_vector(model, layer, torch.tensor(v_dep), a):
                out = generate(model, tok, msgs, max_new_tokens=350, temperature=0, seed=0)
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        pred = extract_answer(out)
        ok = pred == gold
        correct += int(ok)
        raw.append({"gold": gold, "pred": pred, "ok": ok, "reply": out})
    results[f"a{a}"] = correct / len(gsm)
    raws[f"a{a}"] = raw
    print(f"alpha {a:+}: accuracy {results[f'a{a}']:.2f}")

save_json(results, "results/exp6/gsm_collateral_robust.json")
save_json(raws, "results/exp6/gsm_collateral_raw.json")
print("done")
