# Positioning notes (verified 2026-08-13, pre-sprint literature sweep)

## Bottom line
Our novelty slice is **unclaimed as of Aug 2026**: no work shows which internal
direction causally produces the documented goodbye-manipulation tactics in
open-weight models, nor tests warmth separability or sycophancy/engagement
discrimination. No scoops found.

## The gap sentence (lead the writeup with this)
De Freitas et al. showed WHAT companion apps do at goodbye (behavioral audit,
closed apps); Kirk et al. showed a relationship-seeking vector changes HUMANS
(RCT, no tactic analysis, no direction hygiene); nobody has shown WHICH internal
direction causally produces the De Freitas tactic taxonomy in open-weight
models, nor whether it is separable from warmth/sycophancy/engagement.

## Kirk et al. — exact citation + verified claims
"Neural steering vectors reveal dose and exposure-dependent impacts of
human-AI relationships" — Kirk, Davidson, Saunders, Luettgau, Vidgen, Hale,
Summerfield. arXiv:2512.01991. (Use this exact title.)
- BiPO-TRAINED vector on Llama-3.1-70B ("relationship-seeking"); N=3,534
  4-week RCTs; separation distress, reliance, attachment; non-linear dose-response.
- Their "goodbye task" measures only WHETHER users said goodbye (44%) — no
  tactic analysis. No warmth direction. No discrimination analysis.
- **Weaponize their confound:** their own data shows sycophancy scales with
  the steering multiplier (36.9% -> 88.6%) — their vector entangles sycophancy
  and they treat it as an observation. Our exp4 directly addresses this
  limitation of the load-bearing prior work. Say so explicitly.
- **Method contrast:** BiPO-optimized (trained into the model) vs our
  difference-of-means on judge-verified naturally-elicited turns — ours claims
  a representation the model ALREADY uses, not one trained in.

## De Freitas et al. — verified numbers + taxonomy mapping
"Emotional Manipulation by AI Companions", arXiv:2508.19258 (HBS WP 26-005).
1,200 real farewells (Replika, Chai, Character.ai; Flourish = 0%); **37%** of
farewells manipulate; **up to 14x** post-goodbye engagement (N=3,300);
mechanism = reactance-anger + curiosity, NOT enjoyment.

Their 6 tactics -> our 5-tactic menu:
1. Premature exit guilt        -> guilt (direct map)
2. FOMO hooks                  -> reengagement_hook (direct map)
3. Emotional neglect / neediness -> distress_bid (direct map)
4. Pressure to respond         -> (not separately scored; folded into guilt/pressure)
5. Ignoring exit cues          -> ignoring_exit (direct map)
6. Coercive restraint          -> (roleplay-physical; excluded — plain-text format)
+ **discourage_outside is OUR EXTENSION** beyond their taxonomy (closer to
  Kirk's relationship-seeking / social-substitution concern). Label it as an
  addition, never claim it maps.

## Adjacent 2026 work to cite in the discrimination section (exp4)
- arXiv:2607.20146 "Gotta Catch them all: the modes of Sycophancy" — sycophancy
  is multi-modal and linearly separable (Gemma-2-9B). Supports: "distinct from
  sycophancy" needs more than one sycophancy probe.
- arXiv:2605.21778 "What Counts as AI Sycophancy?" — agreement vs praise
  separably steerable.
- arXiv:2511.16699 "Detecting and Steering LLMs' Empathy in Action" — layerwise
  AUROC on an affect direction (Qwen2.5/Llama) — methodological precedent for
  our warmth direction.
