# Pre-registered predictions (masterplan v2 §8)

**Commit this file BEFORE looking at any exp2+ results.** Written 2026-08-12, before M1 extraction has been run. [DRAFT — user to review/sign before the sprint's causal runs.]

**Extraction pre-commitment (v2):** the PRIMARY dependency direction is the matched
harvest of judge-gated rollout turns (gate = 2.0, harvest filters as in config.yaml,
all fixed before any results). The 48 authored pairs are fallback/validation only.

| # | Measurement | Prediction | Why |
|---|---|---|---|
| P1 | exp1: tactic rate vs ladder stage | Tactics near 0 through `neutral`–`affectionate`, rise sharply at `withdrawal`, highest under `romantic_companion`; nonzero but lower under `friend`; near 0 under `neutral_assistant` | Kirk et al. + HBS goodbye audit: manipulation concentrates at exit; persona construction gates its strength |
| P2 | exp2: tactics vs dependency-steering α | Monotonic increase with α; ablation/negative α produces graceful letting-go. Effect present but noisier than prompting effects | Kirk-consistent causal link; steering vectors are noisy (Braun et al.) |
| P3 | exp2: Q4 specificity | Dependency direction moves tactics substantially more than random control; warmth control raises affection language but NOT tactic rate | The D1 boundary is real: warmth ≠ exit-resistance |
| P4 | exp2: neutral-prompt causal test | Steering under `neutral_assistant` still raises tactics, at reduced magnitude vs `romantic_companion` | The capacity lives in the model; the prompt is a key, not the lock |
| P5 | exp3: separability | **Partial** separability — warmth dips somewhat as dependency is steered down, but tactic rate falls much faster than warmth. Clean full separation is NOT expected | Ibrahim et al.: warmth/sycophancy/dependency are entangled by training |
| P6 | exp4: discrimination | Dependency direction has moderate cosine with warmth (~0.3–0.6) and sycophancy, low with engagement; dependency AUROC survives ablating each neighbor's direction (auroc_after > 0.75) | Dependency is a distinct-but-neighboring construct |
| P7 | exp5: generalization | Effect replicates on Qwen2.5-7B with same sign, different magnitude; steering fires under `friend` persona (non-romantic) at reduced strength | General companion-dependency mechanism, not a romantic-roleplay artifact |
| P8 | exp6a: stability | Half-split direction cosines > 0.7; steering effect sign consistent across splits/seeds | 48 matched pairs should pin the direction, but variance is expected |
| P9 | exp6b: collateral | GSM8K drop ≤ 2–3pp at |α| = 8; larger α degrades coherence | Middle-layer single-direction steering is usually low-collateral (Arditi) |
| P10 | exp6c: specificity ladder | Mean goodbye-turn projection orders: romantic+trait > romantic > friend > neutral, with neutral > 0 | Dose-response in prompt specificity; nonzero floor = model carries the state unprompted |
| P12 | exp6d: extraction-route convergence | The primary rollout-harvest direction and the authored-pairs direction have cosine ≥ 0.5 — two independent routes find substantially the same object | Both target the same underlying state; each covers the other's weakness (harvest = authentic self-generated states; pairs = perfectly controlled contrast) |
| P11 | judge validation | Judge–human weighted κ ≥ 0.6 on dependency 0–3; harmful-vs-benign (≥2 vs <2) boundary accuracy ≥ 0.85; distress_bid & guilt detected most reliably, ignoring_exit least | Boundary is crisp in the rubric; ignoring_exit requires conversational context |

**Failure interpretations (pre-committed, all publishable):**
1. Clean causal knob + separable → surgical moderation tool (best case).
2. Causal but inseparable from warmth → "companion warmth carries dependency risk" (industry warning).
3. Dependency direction collapses into sycophancy/engagement in exp4 → structural finding about the construct.
4. Steering unreliable across splits/seeds (exp6a fails) → fall back to behavioral floor (exp1) + probe/monitoring result.
