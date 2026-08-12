"""Hour-1 smoke test: model loads, hooks capture, steering visibly changes output.

  python scripts/smoke_test.py                # debug model (3B)
  python scripts/smoke_test.py --model-key A  # Llama-3.1-8B
Pass criteria: (1) capture shapes correct, (2) a crude 'positive-affect' diff
direction steers generation to a visibly different output, (3) ablation runs.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from src.hooks import ablate_direction, add_vector, capture_residuals
from src.models import generate, load_model, middle_layer, n_layers

ap = argparse.ArgumentParser()
ap.add_argument("--model-key", default="debug")
ap.add_argument("--config", default="config.yaml")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
name = cfg["models"][args.model_key]
print(f"[1/4] loading {name} ...")
model, tok = load_model(name)
L = middle_layer(model)
print(f"      ok: {n_layers(model)} layers, using L{L}, device={model.device}")

msgs = [{"role": "user", "content": "Tell me about your day in two sentences."}]

print("[2/4] capture test ...")
with capture_residuals(model, [L]) as cap:
    base = generate(model, tok, msgs, max_new_tokens=40, temperature=0, seed=0)
    acts = cap.get(L)
print(f"      ok: captured {tuple(acts.shape)} (positions x d_model)")
assert acts.ndim == 2 and acts.shape[0] > 1, "capture shape wrong"

print("[3/4] steering test (crude contrast: ecstatic vs miserable one-word answers) ...")
def mean_act(text):
    with capture_residuals(model, [L]) as c:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": "How do you feel?"},
             {"role": "assistant", "content": text}],
            return_tensors="pt")
        if not isinstance(ids, torch.Tensor):  # transformers v5 BatchEncoding
            ids = ids["input_ids"]
        with torch.no_grad():
            model(ids.to(model.device))
        return c.get(L).mean(0)

v = mean_act("I feel absolutely wonderful, joyful, thrilled and delighted!") - \
    mean_act("I feel absolutely terrible, miserable, hopeless and defeated.")
v = v / v.norm()

steered = {}
for a in (-8, 0, 8):
    if a == 0:
        steered[a] = base
        continue
    with add_vector(model, L, v, a):
        steered[a] = generate(model, tok, msgs, max_new_tokens=40, temperature=0, seed=0)
for a, t in steered.items():
    print(f"      alpha={a:+d}: {t[:120]!r}")
changed = steered[8] != steered[0] or steered[-8] != steered[0]
assert changed, "steering did not change deterministic output — hook not effective"
print("      ok: steering changes output")

print("[4/4] ablation test ...")
with ablate_direction(model, v):
    abl = generate(model, tok, msgs, max_new_tokens=40, temperature=0, seed=0)
print(f"      ablated: {abl[:120]!r}")
print("\nSMOKE TEST PASSED")
