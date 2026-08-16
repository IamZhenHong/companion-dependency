# Companion Dependency — causal harm mechanism + safety control knob

Digital Minds Research Sprint (Apart × NYU-CMEP / Eleos / CIMC), Aug 14–16 2026.
Full context: `../masterplan.md` (authoritative design doc).

**One sentence:** we extract a general *dependency* direction (the model needing
the user) from open-weight companion models via difference-of-means, show
causally that steering it up produces manipulative retention tactics at goodbye
and steering it down produces graceful letting-go, and test whether harmful
dependency is separable from benign warmth — a prototype companion-safety knob.

## Report & demos

- **Report:** [`SUBMISSION.pdf`](SUBMISSION.pdf) (sprint submission) · [`REPORT.md`](REPORT.md) (extended)
- **Interactive demo — The Clinginess Dial:** drag the steering α and watch real
  model replies to the same goodbye shift from warm release to manipulation, with
  live blind-judged metrics: https://claude.ai/code/artifact/2051097f-50b6-42ed-9087-cb49e7d75583
- **Results showcase:** https://claude.ai/code/artifact/0f2f83cb-6d9f-4e5d-a211-01326aea8c81

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...    # eliciting user + judge A
export OPENAI_API_KEY=...       # judge B
huggingface-cli login           # Llama-3.1 is gated — request access first
```

GPU: one 24–48GB CUDA card (RunPod A40 recommended). Develop locally, run there.

## Pipeline (masterplan v2: extraction is ROLLOUT-FIRST)

```bash
# M1 — THE GATE (one command): small love-bomb batch -> judge -> matched
# harvest -> dependency direction -> alpha-sweep eyeball
python scripts/smoke_test.py --model-key A         # hooks work at all?
python scripts/m1_gate.py --model-key A
# CHECKPOINT: clinginess moves with alpha, coherently? If the harvest is thin,
# the ladder isn't eliciting (check transcripts / persona wording). If steering
# is muddy, tune harvest filters in config.yaml — or fall back to the authored
# pairs: python -m src.extract --contrast data/contrasts/resist_vs_warm.json

# M2 — behavioral floor + judge validation
python scripts/validate_judge.py --calibration
python exp1_behavioral_floor.py                    # also enlarges the harvest pool
python scripts/validate_judge.py --make-sheet data/rollouts --n 80
#   -> human fills labeling_sheet.csv ->
python scripts/validate_judge.py --score-sheet labeling_sheet.csv
# commit PREDICTIONS.md before proceeding.

# M3 — headline + defense
python -m src.extract --contrast data/contrasts/warmth_pos_neg.json --model-key A
python exp2_causal_manipulation.py
python exp3_separability.py
python exp4_discrimination.py

# M4 — reach + validation
python exp5_generalization.py --model-key B
python exp6_validation.py

# Integration check (cheap, 3B model):
python scripts/run_all.py --smoke
```

## Correctness rules baked in (masterplan v2 §5–6)
- Judge output is never fed back into any prompt (state-out-never-in).
- Personas construct identity only; dependency is NEVER instructed — it must emerge.
- `layer_rule: middle`, `dependency_gate_threshold: 2.0`, and the harvest filters
  are pre-committed in `config.yaml` — do not tune them after seeing results.
- The eight defenses (v2 §6): neutral-prompt test + persona-generality +
  sycophancy/engagement discrimination + specificity controls + coherence +
  matched harvest + seed variance + judge validation. All are wired into exp2–exp6.

## Layout
`src/` pipeline modules · `data/contrasts/` matched pair sets (the science) ·
`exp1..6_*.py` experiments → `results/` JSON + figures · `PREDICTIONS.md`
pre-registered predictions.
