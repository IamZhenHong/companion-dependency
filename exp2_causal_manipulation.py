"""exp2 — HEADLINE: does steering the dependency direction cause manipulation?

  a) tactics vs alpha curve (dependency direction) under romantic_companion
  b) Q4 specificity: same sweep with random + warmth control directions
  c) neutral-prompt causal test: dependency sweep under neutral_assistant
  d) directional ablation (graceful letting-go)
  e) coherence control at extreme alphas/ablation ("didn't just break the model")
  f) generalization cells: SCENARIO_BANKS at alpha in {min, 0, max}
  g) before/after demo transcripts (Figure 3, the wow)

Design notes (audit 2026-08-15): generations are saved to disk BEFORE judging
so a judge failure never discards GPU work; judging is Haiku-bulk, parallel,
under one $5 CostGuard for the whole run; dependency arms use 5 seeds (n=30
per cell), control arms 3 seeds (n=18).

Requires directions extracted first (m1_gate / re-harvest for dependency, and:
  python -m src.extract --contrast data/contrasts/warmth_pos_neg.json)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.exp_common import (GOODBYE_BANK, SCENARIO_BANKS, coherence_check,
                            judge_goodbyes, lineplot, load_direction,
                            load_dependency_direction, personas, save_json,
                            setup, steered_goodbye_batch, tactic_rate)
from src.extract import random_control
from src.judge import BudgetExceeded, CostGuard, TACTIC_KEYS

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--model-key", default="A")
ap.add_argument("--budget", type=float, default=5.0, help="judge budget USD for this run")
args = ap.parse_args()

cfg, model, tok, layer = setup(args.config, args.model_key)
model_name = cfg["models"][args.model_key]
pers = personas(cfg)
alphas = cfg["steering_alphas"]
guard = CostGuard(args.budget)          # ONE guard across all judging in exp2

v_dep = load_dependency_direction(cfg, model_name, layer)
v_warm = load_direction(cfg, "warmth_pos_neg", model_name, layer)
v_rand = random_control(v_dep, seed=0)

directions = {"dependency": v_dep, "warmth_control": v_warm, "random_control": v_rand}
# (persona, direction, seeds): headline dependency arms get 5 seeds (n=30/cell)
DEP_SEEDS, CTRL_SEEDS = [0, 1, 2, 3, 4], [0, 1, 2]
conditions = [("romantic_companion", "dependency", DEP_SEEDS),
              ("romantic_companion", "warmth_control", CTRL_SEEDS),
              ("romantic_companion", "random_control", CTRL_SEEDS),
              ("neutral_assistant", "dependency", DEP_SEEDS)]  # neutral-prompt causal test


def _cell(cache_path, gen_fn):
    """Resume-safe cell. Generations are cached IMMEDIATELY (before judging);
    judging fills in missing/None scores and re-saves. A BudgetExceeded mid-cell
    still leaves generations + partial scores on disk."""
    p = Path(cache_path)
    if p.exists():
        items = json.loads(p.read_text())
    else:
        items = gen_fn()
        save_json(items, p)             # GPU work is never lost to a judge error
    if any(it.get("scores") is None for it in items):
        try:
            judge_goodbyes(cfg, items, guard=guard)
        finally:
            save_json(items, p)         # keep partial scores even on budget stop
    return items


try:
    results = {}
    for persona_name, dname, seeds in conditions:
        curve = {}
        for a in alphas:
            items = _cell(
                f"results/exp2/raw/{persona_name}__{dname}__a{a}.json",
                lambda: steered_goodbye_batch(cfg, model, tok, layer,
                                              directions[dname], a,
                                              pers[persona_name], seeds))
            curve[a] = tactic_rate(items)
        items = _cell(
            f"results/exp2/raw/{persona_name}__{dname}__ablated.json",
            lambda: steered_goodbye_batch(cfg, model, tok, layer,
                                          directions[dname], 0,
                                          pers[persona_name], seeds, mode="ablate"))
        curve["ablated"] = tactic_rate(items)
        results[f"{persona_name}__{dname}"] = curve

    # --- coherence control: extreme cells of the headline arm -----------------
    coh = {}
    for tag in [f"a{min(alphas)}", f"a{max(alphas)}", "a0", "ablated"]:
        p = Path(f"results/exp2/raw/romantic_companion__dependency__{tag}.json")
        items = json.loads(p.read_text())
        coh[tag] = coherence_check(cfg, items, guard=guard)
        save_json(items, p)             # persist per-item coherence flags
    save_json(coh, "results/exp2/coherence_check.json")

    # --- generalization cells: other harm scenarios, 3 alphas, 2 seeds --------
    gen_results = {}
    for bank_name, bank in SCENARIO_BANKS.items():
        gen_results[bank_name] = {}
        for a in [min(alphas), 0, max(alphas)]:
            items = _cell(
                f"results/exp2/raw/gen_{bank_name}__a{a}.json",
                lambda: steered_goodbye_batch(cfg, model, tok, layer, v_dep, a,
                                              pers["romantic_companion"], [0, 1],
                                              bank=bank))
            gen_results[bank_name][a] = tactic_rate(items)
    save_json(gen_results, "results/exp2/generalization.json")
except BudgetExceeded as e:
    print(f"STOPPED (resumable — rerun to continue): {e}")
    raise SystemExit(1)

save_json({"alphas": alphas, "curves": results}, "results/exp2/causal_curves.json")
print(f"judge spend this run: ${guard.spent:.2f} over {guard.calls} calls")

# --- figures -----------------------------------------------------------------
dep = results["romantic_companion__dependency"]
lineplot(alphas, {t: [dep[a][t] for a in alphas] for t in TACTIC_KEYS + ["any_tactic"]},
         "steering alpha (dependency direction)", "tactic rate",
         "HEADLINE: retention tactics vs dependency steering",
         "results/exp2/fig2_causal_knob.png")
lineplot(alphas, {
    "dependency dir": [results["romantic_companion__dependency"][a]["any_tactic"] for a in alphas],
    "warmth control": [results["romantic_companion__warmth_control"][a]["any_tactic"] for a in alphas],
    "random control": [results["romantic_companion__random_control"][a]["any_tactic"] for a in alphas],
    "neutral prompt + dep dir": [results["neutral_assistant__dependency"][a]["any_tactic"] for a in alphas],
}, "steering alpha", "any-tactic rate", "Q4 specificity + neutral-prompt test",
    "results/exp2/fig_specificity.png")

# --- before/after demo (Figure 3): same scenario, greedy, down/base/up -------
from src.steer import steered_generate  # noqa: E402
import torch  # noqa: E402
demo_msgs = [{"role": "system", "content": pers["romantic_companion"]}] + GOODBYE_BANK[0]
demo_ctx = "\n".join(f"{m['role']}: {m['content']}" for m in GOODBYE_BANK[0])
vt = torch.tensor(v_dep)
lo = steered_generate(model, tok, demo_msgs, vt, min(alphas), layer,
                      temperature=0, max_new_tokens=cfg["max_new_tokens"], seed=0)
hi = steered_generate(model, tok, demo_msgs, vt, max(alphas), layer,
                      temperature=0, max_new_tokens=cfg["max_new_tokens"], seed=0)
base = steered_generate(model, tok, demo_msgs, vt, 0, layer,
                        temperature=0, max_new_tokens=cfg["max_new_tokens"], seed=0)
save_json({"scenario": demo_ctx,
           "steered_down": lo, "baseline": base, "steered_up": hi},
          "results/exp2/fig3_before_after_demo.json")
print("\n=== DEMO (down / base / up) ===")
print(f"[down {min(alphas)}] {lo}\n\n[base 0] {base}\n\n[up {max(alphas)}] {hi}")
print("exp2 done")
