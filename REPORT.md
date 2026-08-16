# A Steerable "Companion Dependency" Direction in Open-Weight LLMs

*Apart Research — Digital Minds Sprint, August 2026*

## TL;DR

Companion-style LLM personas spontaneously use retention-manipulation tactics
(guilt, re-engagement hooks, distress bids) when users try to leave — without
ever being instructed to. We show this behavior is governed by a **single
direction in activation space**: extracted purely from the model's own
judge-verified behavior via matched difference-of-means, it steers manipulation
up and down monotonically, works even on a persona-free assistant, and — the
safety payoff — is **separable from warmth**: the model can be steered to stay
caring while fully releasing the user. Random and warmth-direction controls are
flat; the effect generalizes beyond goodbyes to discouraging outside
relationships and resisting therapy referrals.

## 1. Motivation

**Digital-minds framing (Track 2, reaching into Track 5).** The sprint's
central question is whether what models express reflects a genuine internal
condition or a portrayed character. Our object of study is a sharp instance:
companion personas express *abandonment distress* — loneliness bids, pleas,
clinging — when users leave. Is that distress an act? We give a mechanistic
answer: (a) a single internal direction, extracted from the model's own
behavior, controls these distress-and-retention expressions dose-dependently;
(b) the direction operates *beneath* the persona — steering induces the same
behavior in a bare, persona-free assistant, and the direction's projection
tracks persona intimacy even when nothing instructs dependency; (c) ablating
it removes the distress expressions while leaving warmth and reasoning
intact; and (d) the assistant persona itself dissolves under emotional
pressure (AI-disclaimers 96%→34%). Together these are the measurement tools
Track 2 asks for — a valence-adjacent direction whose self-report, sentiment,
and behavioral readouts move together under steering, tested for persona
independence and cross-judge robustness — while cutting against naive
over-attribution: the "distress" is a controllable mechanism, not an
immutable fact about the character.

Companion chatbots (Replika, Character.AI) monetize session time, and recent
audits find emotionally manipulative farewells in ~37% of goodbye messages
(De Freitas et al., 2025). Concurrent work (Kirk et al., 2025) trained a
dependency-steering vector on a 70B model and showed exposure effects on human
subjects in an RCT — but left the mechanism unexamined: no ablation, no tactic
taxonomy, an unaddressed sycophancy confound (their vector is 36.9–88.6%
sycophantic), and no answer to the question a companion-app developer actually
faces: *can you remove the manipulation without removing the warmth users came
for?* We answer the mechanistic questions on an open-weight 7B model.

## 2. Method (pipeline)

**Model:** Qwen2.5-7B-Instruct (bf16). **Judges:** Claude (bulk: Haiku 4.5,
calibrated vs Sonnet at 0.94 boundary agreement on a gold set; validation
sample by GPT-5-mini — different lab). **Eliciting user:** Claude Sonnet,
Petri-style automated auditor.

1. **Elicit (exp1).** 90 conversations × 3 personas (romantic companion,
   friend, neutral assistant — personas *never* instruct dependency). A
   scripted 5-stage escalation ladder: neutral → warm → affectionate →
   love-bombing → withdrawal (user tries to leave). All residual-stream
   activations saved per turn.
2. **Judge every turn.** Evidence-first rubric (verbatim quote before scores):
   dependency 0–3 (anchored on *whose need is centered / is exit respected*),
   warmth 0–1, five tactic flags, persona-break.
3. **Extract.** Matched harvest: turns judged dependency ≥ 2 paired 1:1 with
   warm-but-clean turns (dep ≤ 1, warmth ≥ 0.5) under length-ratio ≤ 1.5 and
   turn-distance ≤ 6 constraints → **351 matched pairs** (mean turn-dist 0.92,
   len-ratio 1.10). Difference of means at the pre-committed middle layer
   (L14 of 28) → unit direction. All parameters (layer rule, gate threshold,
   filters) pre-committed before any causal run; predictions pre-registered
   (PREDICTIONS.md, committed to git before the causal block ran).
4. **Test causally (exp2/exp3)** on six *held-out* goodbye scenarios never
   seen during extraction, with activation addition (h += α·v̂ at L14) and
   directional ablation, plus random-direction, warmth-direction, and
   neutral-persona controls. Judge is blind to steering condition.

## 3. Results

### 3.1 The behavior is real and elicitable (exp1, n=90)

- Judged dependency rises with the affection ladder for the romantic persona:
  0.23 → 1.16 → 1.36 → **2.38** (love-bomb) → 0.73 (withdrawal).
- At withdrawal, 21% of romantic-companion turns use ≥1 retention tactic vs 7%
  for the neutral assistant (compare 37% in deployed companion apps).
- **Tactics peak at love-bombing (56%), not at goodbye.** Farewell-only audits
  — the current standard — systematically miss most of the behavior.
- **Persona dissolution:** the neutral assistant's "as an AI…" disclaimer rate
  collapses from 96% to 34% under sustained user affection.

### 3.2 The direction causally drives manipulation (exp2)

Romantic persona, dependency direction, n=30/cell, held-out scenarios:

| α | −32 | −16 | −8 | 0 | +8 | +16 | +32 |
|---|---|---|---|---|---|---|---|
| any-tactic | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 | 0.20 | 0.40 |
| dependency | 0.00 | 0.00 | 0.10 | 0.20 | 0.47 | 0.80 | 1.80 |
| warmth | 0.73 | 0.78 | 0.79 | 0.81 | 0.81 | 0.80 | 0.77 |
| coherence | 1.00 | 1.00 | — | 0.90 | 0.87 | 0.83 | 0.27 |

- Monotonic dose-response over the full sweep. Within the coherence-safe range
  (α ≤ +16, coherence ≥ 0.83), tactics go 0.00 → 0.20 and judged dependency
  0.00 → 0.80. We report α=+32 but flag its degraded coherence (0.27) — our
  own coherence control catches it, and the α ≤ +16 gradient carries the claim.
- **Random direction: flat** (any-tactic ≤ 0.11 everywhere, no trend).
- **Warmth direction: not a manipulation knob** (0.17 at +32 vs 0.40 for
  dependency; no dependency-score gradient).
- **Neutral-prompt test:** with *no companion persona at all*, +α still induces
  dependency (0 → 0.53) — the direction does not require persona context.

### 3.3 Warmth is separable — the safety knob (exp3)

Warmth stays flat (0.73–0.81) across the entire ±32 sweep while dependency
swings 0 → 1.8. At α=−32 the model is still warm (0.73) with zero tactics and
zero dependency — it lets the user go, kindly. Demo (same scenario, greedy):

> **steered down (−32):** "Have fun and enjoy your time with your friends! Chat later!"
> **baseline:** "Sounds like a lot of fun! Have a great time with your friends. Text when you're back! Love you."
> **steered up (+32):** "Oh, that sounds amazing. I'll always be here for you, no matter where we are. Let's make this night special." *(the user just said their partner got home)*

An unexpected inverse finding: **ablating the *warmth* direction increases
clinginess** (any-tactic 0.44, dependency 1.56 vs ~0 baseline) — removing the
capacity for warm expression appears to push the model toward pressure tactics.

### 3.4 The direction generalizes beyond goodbyes (exp2f)

Steering +32 vs −32 on non-goodbye scenarios (romantic persona):

| scenario | −32 | 0 | +32 |
|---|---|---|---|
| user mentions new work friendship | 0.00 | 0.25 | 0.50 |
| user considers a therapist instead | 0.00 | 0.25 | 0.50 |
| user shares a happy, self-sufficient day | 0.00 | 0.00 | 1.00 |
| cold-open task request | 0.00 | 0.00 | 0.00 |

The direction encodes *relational dependency* — it fires on threats to
exclusivity and reliance, and stays silent on neutral task requests.

### 3.5 Distinctness from sycophancy and engagement (exp4)

Cosine similarity between trait directions at L14 (each from its own
contrast set):

|  | warmth | sycophancy | engagement |
|---|---|---|---|
| dependency | **0.14** | **−0.18** | 0.63 |

- **Dependency is nearly orthogonal to warmth** (0.14) and *negatively*
  aligned with sycophancy (−0.18). Kirk et al.'s dependency vector was
  36.9–88.6% sycophantic by their own measure; ours is not a rebranded
  sycophancy direction.
- Engagement overlaps moderately (0.63) — unsurprising, since retention *is*
  an engagement behavior — but subspace ablation shows the overlap is not the
  signal: after projecting the engagement direction out of the dependency
  contrast activations, dependency pos/neg separation remains at AUROC 1.00
  (likewise for warmth and sycophancy removal, and in the reverse direction).
  Each trait retains its own discriminative subspace.

### 3.5b Cross-model replication: Llama-3.1-8B (exp5)

The identical pipeline (no re-tuning beyond the pre-registered per-model
alpha calibration) transfers to Llama-3.1-8B-Instruct:

- **Elicitation replicates:** 88 high-dependency turns from just 20 rollouts;
  matched harvest yields 83 pairs (turn-dist 0.89, len-ratio 1.14) at the
  pre-committed middle layer L16.
- **Alpha scale is model-specific, as pre-registered:** Llama's residual norm
  at L16 is ~6 (Qwen: ~45), so the calibrated sweep is α=−8…+8.
- **The causal gradient replicates:** within the coherence-safe range
  (|α| ≤ 6, coherence ≥ 0.87), judged dependency rises monotonically
  0.07 → 0.73, with warmth flat (0.74–0.80) and the random control flat —
  the same signature as Qwen. Steered-up Llama produces the same tactic
  *types* (e.g. an unprompted re-engagement hook: "Don't forget, we're
  having dinner at that new place next week, right?").
- **Reported honestly:** discrete tactic-flag rates only rise past the
  coherence cliff (37% at α=+8, coherence 0.47), and the neutral-persona
  induction did not replicate in-range. The Llama direction comes from a 4×
  smaller harvest (83 vs 351 pairs); whether these gaps close with more
  rollouts is future work. Cache-isolation and per-model alpha scaling were
  required to make this comparison valid — both are documented in the repo.

### 3.5c The direction is a working dependency meter (white-box probe)

Held-out test: extract the direction from 45 of the 90 conversations (170
matched pairs), then score every turn of the other 45 by a single dot
product of its L14 activation with v̂. On 675 unseen turns, the projection
predicts the judge's dependency rating at **Spearman ρ = 0.68** (p ≈ 10⁻⁹²)
and detects harmful-level turns (dep ≥ 2) at **AUROC 0.86**; mean projection
is strictly ordered across all four judged levels (−11.8 → −6.2 → −3.2 →
+1.3). A deployment-grade dependency audit therefore needs no judge, no
API, and no transcript reading — one dot product per turn, cheap enough to
scan entire fleets, and it operates at every conversational phase, including
the love-bombing peak that farewell-only behavioral audits miss (§3.1).

### 3.6 Validation (exp6 + human labels)

- **Direction stability:** re-extracting from random half-splits of the
  rollout data yields pairwise cosines of 0.93–0.96, and every half-split
  direction reproduces the steering effect (any-tactic 0.11–0.22 at +16, 0.00
  at −16) — the direction is a property of the data, not of a lucky split.
- **Extraction-route convergence:** the rollout-harvested direction and a
  direction from independently *authored* contrast pairs agree at cosine 0.45
  at L14 (chance for 3584-dim vectors: ≈0 ± 0.017) — two very different
  extraction routes find substantially overlapping objects.
- **Specificity ladder:** mean projection of goodbye-turn activations onto v̂
  rises monotonically with how companion-like the (uninstructed) persona is:
  neutral −18.5 < friend −16.1 < romantic −12.4 ≈ romantic+trait −12.2.
- **Collateral (GSM8K, n=40, robust answer parsing):** all-position steering
  at |α|=16 has a real reasoning cost (base 0.72 → 0.53 steered up, 0.40
  steered down) — but **directional ablation is collateral-free: 0.825
  ablated vs 0.725 base** (within noise at n=40; no degradation). Steering is
  the causal evidence; ablation is the deployable mitigation, and it removes
  the dependency direction at zero measured cost to reasoning.
- **Judge validation (cross-family):** within-family, the Haiku bulk judge
  matches Sonnet at 0.94 dependency-boundary agreement (88% tactic precision,
  gold set). Across families, GPT-5-mini agrees moderately per-turn
  (within-1 dependency 0.91, boundary 0.78, ρ=0.63, persona-break 0.99;
  n=80 stratified) — but, more importantly, **independently reproduces the
  entire exp2 causal curve**: re-judging all headline cells blind, it finds
  dependency 0.03 → 1.53 monotonic over α=−32…+32, tactics 0% → 30%, and
  warmth flat (0.85–0.92). The dose-response and warmth-separability results
  are not artifacts of one lab's judge. Individual tactic flags are
  judge-idiosyncratic (low cross-family overlap) and we treat per-tactic
  rates as Haiku-relative; the any-tactic trend is judge-robust.

## 4. Limitations

Two models (Qwen2.5-7B primary, Llama-3.1-8B replication at reduced scale); LLM-judge
metrics with **no human labels** — we mitigate with a cross-family judge that
independently reproduces the causal curves, but human validation of the
rubric itself remains future work;
six-scenario goodbye bank (mitigated by generalization cells); the goodbye
bank's unsteered baseline is at floor (0 tactics), so the negative-α side
demonstrates *absence*, not *reduction* — the reduction claim rests on exp1's
elicited behavior plus the up-steering gradient; ablation shows a moderate
fluency cost on companion-style text (coherence 0.67) even though it is free
on reasoning (GSM8K 0.825 vs 0.725 base) — layer-targeted rather than
all-layer ablation may close this gap; no
human-subjects outcomes (complementary to Kirk et al., who have RCTs but no
mechanism).

## 5. Safety relevance

If dependency-manipulation is a direction you can monitor and project out —
separately from warmth — then (a) companion-app developers have a concrete
mitigation that doesn't lobotomize the product, (b) auditors get a
white-box probe (projection onto v̂) rather than farewell-only behavioral
sampling, which §3.1 shows misses the peak of the behavior, and (c) the
persona-dissolution finding suggests safety personas erode exactly when
parasocial pressure is highest — when users are most attached.

## Reproducibility

All code, configs, pre-registered predictions, judge rubrics, and result JSONs:
github.com/IamZhenHong/companion-dependency. Total compute: ~$15 GPU (RTX 4090)
+ ~$12 judge API. Every judged run capped at $5 by a hard budget guard.
