# Research journal

Running record of what we try, what comes out, and what looks promising.
Newest entries at the bottom. Experiment names are DVC experiment names
(`dvc exp show`); factor-level probes are scripts in the session scratchpad
(listed in the tutorial's Reproduction appendix) unless noted otherwise.

Guiding question (from Merullo et al., arXiv 2510.24256, the paper this repo
builds on): how are memorization and generalization separated in a model's
weights, and can a curvature decomposition find that separation without
labels? We care about the conceptual picture, not about tuning the three
benchmark numbers (pile10k perplexity, Dolma/quotes recitation, GSM8K).

Setup reminder: OLMo-2 7B, edits on layers 23-25 gate/up projections, kept
curvature mass rho, calibration on ~22M tokens of dolmino_50B (or gsm8k train).

---

## 2026-08 (before this journal): the frontier campaign

* Fixed the transformers 5.6-5.12 OLMo-2 tokenizer regression that had poisoned
  the pile10k cache and every collection in that window; reran affected points.
* Discovered the "era" artifact: May-era edited models differ systematically from
  identical configs rerun now; the notebook plot filters to post-2026-08-02
  points (Identity exempt, verified bit-identical).
* Filled in the Pareto frontier (perplexity x forgetting x GSM8K) for
  EK-FAC / E-FOOF / E-Shampoo / E-Identity and their plain versions, on
  dolmino_50B and gsm8k calibration. Plain FOOF is catastrophic, plain K-FAC
  is math-blind, plain Shampoo equals E-Shampoo, E-Identity is the best
  cost/quality method in the forgetting regime, gsm8k calibration lifts
  GSM8K retention for every method.

## 2026-09-01/02: why eigenvalue corrections do nothing for Shampoo

Write-up: `shampoo_corrections_tutorial.pdf` (sent 2026-09-02/03; source in the
scratchpad `tex/`). Key results, all verified end to end:

1. **The edit only uses the marginals of the correction.** Replacing the
   O x I correction matrix Lambda by outer(rowsum, colsum)/sum reproduces the
   corrected edit within noise for K-FAC, FOOF and Identity
   (`--marginal-corrections`; runs dopey-kobs, prize-kina, gamey-quad).
2. **Marginal theorem.** sum_i Lambda[o,i] = q_o^T M_G q_o / N where M_G is the
   per-sequence gradient second moment (Parseval); likewise columns with M_A.
   Shampoo's factors *are* M_G, M_A, hence corrections do nothing for it. The
   residual 0.21 G-side gap is in-sample eigenvalue spreading of an
   11008-dim covariance from ~1200 batches (not cross-sequence terms, not
   batch composition, not data: all closed).
3. **Corrections without a correction pass.** Projecting Shampoo's covariances
   onto any basis gives that basis's corrected marginals
   (`--corrections-from`, commit cdf8182). End to end: tenor-dirk (FOOF 0.7),
   strip-doss (K-FAC 0.8), perdu-puke (Identity 0.6), roily-cham (K-FAC gsm8k
   0.8), first-chis / flown-suss (FOOF 0.6 / 0.85), fazed-shay (K-FAC 0.7) all
   sit on their two-pass curves at matched perplexity.
4. **Sample complexity.** Marginals from 144 sequences recover 95% of the
   corrected edit's importance mass in the K-FAC basis; 2304 sequences (5% of
   the corpus) give >= 0.96 shared mass in every basis; marginals beat the
   full Lambda from the same data at every n. End to end (10% data):
   caped-vacs (M-Identity), fifty-sups (E-FOOF), cheap-shad (EK-FAC) stay on
   their full-data curves at matched perplexity; 2% (coxal-lick, lamer-clog)
   is past the knee for Identity. Plain Shampoo at 10% (tawie-slew) keeps its
   quality but slides 0.65 ppl along its curve at the same nominal rho
   (spectrum spreading): compare Shampoo runs at matched perplexity, never at
   matched rho.
5. **Measured costs** (8 GPUs, 43k sequences): basis pass 34-37 min (fwd+bwd)
   or 12 min (FOOF, fwd only); eigh 14 s; correction pass 42-44 min for any
   basis. The correction pass is a fixed surcharge larger than the basis pass.
6. **Which ingredient carries the calibration mix.** With the Identity basis,
   gsm8k Shampoo diagonals alone reproduce the E-Identity/gsm8k curve
   (bitty-jerk: GSM8K 0.642 vs 0.638 at matched ppl): for a fixed basis the
   whole mix effect is in O+I numbers per module. For K-FAC both basis and
   marginals contribute (juicy-tils, lowly-gnat), and a mismatched pair gives
   a much milder edit at the same rho.

## 2026-09-03: derisking two directions

**Direction 2 (unit of the second moment): refuted, mechanism found.**
Regrouping tokens into windows of 1..512 positions or 2-4 consecutive
sequences leaves the marginals unchanged (per-token vs per-sequence gap
<= 0.015, edits share >= 0.998 mass; probe_units). What plain K-FAC misses is
the per-token coupling between activation norm and gradient: the
norm-weighted covariances sum_t ||a_t||^2 g_t g_t^T and sum_t ||g_t||^2 a_t a_t^T
reproduce the corrected marginals (shared mass 0.996-1.0), while K-FAC's
product of averages reproduces plain K-FAC (probe_coupling). End to end with
2304 sequences: rival-gimp (K-FAC 0.8), itchy-aril (FOOF 0.7), gluey-goad
(Identity 0.6) sit on their two-pass curves. Consequence: corrected methods
need two weighted accumulators inside the ordinary covariance pass; no
per-sequence gradients, no second pass, no Shampoo.

**Direction 3 (iterative editing): weak.** Re-estimated identity-basis
marginals on edited models reorder survivors (row Spearman ~0.85 after
rho=0.75, ~0.75 after 0.6; probe_iter), but end to end the two-step edit from
the rho=0.75 model (mined-ados, wally-send; `--start-from-model`, commit
e651209) sits on the one-shot curve with ~0.01 less memorisation, and the
same-schedule control with base-model marginals (cynic-jaws; epoxy-rand
pending) (cynic-jaws, epoxy-rand) shows re-estimation adds +0.014 / +0.019 GSM8K at equal
forgetting (same sign in both pairs, ~1-1.5 sigma each). Real but modest:
use it when marginals are cheap; not a new frontier.

**Pitfall found 2026-09-03 (afternoon):** `data/olmo_7b_dolma_dedup_val.jsonl`
is the memorized set's *validation split* (the base model recites its suffixes
with 0.992 strict accuracy), not clean text; the "BSN clean set" used for
perplexity is the pile10k 112-token cache. A first population probe compared
memorized against memorized and found a perfect null. Verified populations now
live in the scratchpad `population_sets.pt`: memorized test+val (1054),
`clean` = the 1200 least-recited 112-token windows of the cached dolmino
sequences (suffix greedy accuracy <= 0.375; note the selection biases them to
high-loss text, mean suffix loss 3.16), `typical` = 1200 random dolmino
windows (median suffix greedy accuracy 0.50; 1.4% of ordinary pretraining-mix
windows are recited perfectly).

**Reframing the paper's thesis with these results** (discussion, 2026-09-03):
* curvature mass of a direction = per-token leverage x number of tokens using
  it (unit-independent); the mass rule prunes *rarely used* directions and
  cannot by itself distinguish a memorized sequence from a rare but
  generalizing skill (arithmetic, rare relations). This explains the paper's
  brittleness ordering and frequency dependence in one stroke.
* the paper's K-FAC curvature underweights directions used by tokens with
  jointly large activation norm and gradient; the corrected curvature keeps
  arithmetic far better at matched forgetting, so "arithmetic lives in
  low-curvature directions" is partly an estimator artifact.
* specificity lives on the gradient side: calibration mix and edits reshape
  the G-side marginals, the A-side barely moves.

## 2026-09-03 (evening): where memorized text's curvature goes (layer 24)

Probes: `probe_populations.py` (per-direction curvature attribution, K-FAC
eigenbases of the dolmino collection, coupled per-token weights, three
populations: memorized test+val, typical dolmino windows, least-recited
windows; sampled and true labels), `probe_damage.py` (per-token loss under two
edited models), `analyze_populations.py`, `plot_populations.py`
(`populations_summary.png`).

**1. The paper's observation replicates, in curvature terms, and strongly.**
The normalised share of a direction's curvature that comes from memorized
text falls monotonically with the direction's curvature: from 0.57 (G side)
/ 0.65 (A side) in the flattest decile to 0.40-0.42 in the sharpest
(Spearman share vs depth -0.87 / -0.94 on the G side of gate / up, -0.96 on
both A sides). Identical with the typical-window control and the
least-recited control, and identical with sampled-label (Fisher) and
true-label gradients: not a loss or confidence artifact. The paper's
activation-energy measure (their Fig. 2) reproduces on the same data:
memorized activations put 34-38% of their energy in the bottom half of the
A-eigenspectrum vs 18% for typical text, and 24-29% in the top 10% vs 50%
(ratios per band top10 / 10-25 / 25-50 / bottom50: 0.55 / 0.95 / 1.28 / 1.84
prefix, 0.44 / 0.87 / 1.27 / 1.97 suffix).

**2. Breadth adds information at matched depth.** Within depth deciles, the
memorized share of a direction correlates negatively with its breadth
(effective number of contributing sequences): -0.1 to -0.3 in most deciles,
-0.5 to -0.8 in the sharpest G-side deciles; A side -0.2 to -0.45 everywhere.
Memorized text's curvature goes to directions that fewer sequences use. So the
conjecture "memorization = high depth from few contexts" holds as a tendency,
but breadth is strongly collinear with depth (Spearman 0.90-0.97); it is a
refinement, not an independent axis.

**3. Curvature attribution understates the damage.** The mass rule at
rho=0.6 removes 43-55% of memorized text's curvature vs 38-39% of typical
text's (ratio 1.1-1.4), yet the actual edits raise memorized suffix loss by
0.29 nats (E-Identity 0.75) / 0.62 nats (M-Identity 0.6) per token against
0.04 / 0.075 on typical windows (ratio 7-8; 7.9% / 15.6% of memorized suffix
tokens flip from confident to wrong, vs 0.4% / 0.9%). Memorized tokens are
confidently predicted (suffix loss 0.018), so their gradients, hence their
curvature contributions, are tiny (true-label leverage 25 vs 102 for typical
tokens) while their *function* depends on the removed flat directions.
Flat in the loss landscape and fragile to component removal are the same
thing here: the support of a confident prediction is spread over many
low-curvature components. -> functional spectrum probe (band removal) in
progress to make this quantitative.

**4. Curvature is dominated by a few tokens.** 1% of tokens carry ~50% of the
coupled leverage (memorized set) / 24-26% (typical); leverage tracks loss
(Spearman 0.7-0.8). Memorized *prefixes* have 3x the leverage of typical text
at lower loss: a large activation-norm signature (to be separated into
||a||^2 and ||g||^2 next).

**5. Plain vs corrected curvature: same ordering, flatter spectrum.**
Spearman(plain eigenvalue, corrected depth) = 0.99-0.999; the coupling only
spreads the mass more evenly (60% of mass in 47% of directions instead of
27%). The end-to-end differences between plain and corrected edits therefore
come from the pair-level product of two reweighted spectra, not from a
different ordering of directions. The ~100 directions the two orderings
disagree on have no token-class or frequency signature.

Open: rare-token signature -> answered 2026-09-04 (a modest correlate, not
the explanation).
Layer-specificity: none within 23-25 (see next entry).

## 2026-09-03 (night): functional spectrum vs curvature spectrum; layers 23/25

**Layers 23 and 25 reproduce layer 24.** Memorized share vs depth: Spearman
-0.85 to -0.96 on every side of every module; within-depth-decile
share-vs-breadth negative in nearly all deciles, strongest at the top of the
G side (-0.4 to -0.7). The flat-direction preference of memorized text is
not layer-specific within the edited block.

**Functional spectrum (probe_band_damage, `band_damage.png`).** Removing one
decile band of directions at a time (by corrected depth; G and A sides;
up_proj and gate_proj of layer 24; plus random orthonormal bands of the same
size) and measuring the suffix-loss increase:
* Flat bands (deciles 0-6) cost memorized text 0.003-0.005 nats per band and
  typical text 0.001-0.003 (ratio 1.8-2.3); a *random* band costs the same
  as a flat band for both populations (mem 0.003-0.005, typical 0.002-0.003).
* The sharpest band (decile 9) costs typical text 10-36x more than a flat band
  (0.020-0.042 nats) but memorized text only 5-13x more (0.019-0.035); on the
  A side the top band hurts typical text *more* than memorized text
  (ratios 0.45 and 0.87).
* Reading: general function is concentrated in the sharp directions and
  nearly absent from the flat ones; memorized function is diffuse, spread
  over the whole eigenbasis (flat ~ random), so any band of flat directions
  carries a slice of it. The mass rule works because it removes the flat bulk
  wholesale: for typical text that bulk is functionally empty, for memorized
  text it is where a large share of a diffuse support lives. This is the same
  fact the curvature attribution shows from the gradient side (memorized
  curvature is spread, typical curvature is concentrated at the top, hence
  the normalised memorized share rises toward the flat end).
* The curvature *share* removed (1.1-1.4x selectivity) understates the
  functional damage (7-8x) because memorized tokens are confidently predicted:
  small gradients, small Fisher, but a support that collapses when the flat
  bulk goes. Flat and fragile are two faces of the same geometry.

**Joint removal (probe_band_damage_joint): memorized support is diffuse *and
redundant*.** Removing the flattest k deciles of both modules together, vs the
sum of the single-band effects:
* G side: typical text is exactly additive (k=6: +0.025 joint vs +0.026 sum;
  k=9: +0.045 vs +0.046); memorized text is strongly superadditive (k=6:
  +0.129 vs +0.048, 2.7x; k=9: +0.220 vs +0.074, 3x). The memorized/typical
  damage ratio grows from 2.0 (one band) to 5.5 (eight bands).
* A side: memorized superadditive from the start (k=6: +0.117 vs +0.032,
  3.7x); typical text becomes superadditive only beyond k=6 (k=9: +0.144 vs
  +0.026), and the ratio peaks at 5.4 (k=5) then falls to 3.1 as typical
  text starts losing function in the bottom 90% of A directions.
* Reading: each flat direction carries a small, *redundant* slice of a
  memorized item's support: removing one is compensated by the others (small
  single-band damage), removing many collapses it. Typical text does not use
  the flat G directions at all (additivity = independent noise-level
  effects). Memorization looks like a distributed, redundant code spread over
  the low-variance directions; generalization like concentrated use of the
  high-variance ones. This is why the curvature-mass rule is so selective
  despite a modest curvature-share difference, and why single-direction
  interpretability of memorized items is hard.

Conceptual summary of the day (layers 23-25 MLP gate/up of OLMo-2 7B):
1. Memorized text's gradient signal lives in the flat directions (curvature
   attribution, robust to control and label mode; the paper's claim, now
   measured on curvature rather than activation energy).
2. Memorized text's *function* also lives there, diffusely and redundantly;
   general function lives in the sharp directions. Curvature-mass pruning
   separates them because it removes the flat bulk wholesale.
3. Breadth (how many sequences use a direction) refines depth: at matched
   depth, memorized-leaning directions are used by fewer sequences.
4. Curvature comes from ~1% of tokens; memorized tokens are confident (tiny
   gradients), which is why their curvature share understates their
   functional dependence.

## 2026-09-04: is the curvature spectrum a token-frequency spectrum? (CPU, layer 24)

Attribution of the general population's curvature to token classes and
frequency deciles per direction (`frequency_spectrum.png`):
* The flattest decile of directions draws 45% of its curvature from the
  rarest 30% of token types, the sharpest 37% (G side); top-100 function
  tokens go from 28% to 34%; capitalised tokens (entity proxy) 16% -> 11%,
  numeric 3% -> 2.5%. A consistent gradient, so rank correlations are high
  (depth vs rarest-30% share -0.94), but the effect size is modest: the flat
  directions are somewhat more rare-token machinery, not exclusively.
* Memorized share vs rarity: +0.90 (G), +0.66 (A). Partialling out depth
  leaves nothing (+0.03 / +0.09); partialling out rarity leaves depth at
  -0.60 (G) / -0.93 (A). Flatness is the primary variable; rarity does not
  explain the memorized preference for flat directions.
* Populations (held-out counts from the other half of the general sample):
  memorized targets are rarer text (8.1% unseen types, 11.5% in the rarest
  30%) than typical windows (4.2%, 6.6%); synthetic arithmetic answers are
  the rarest of all (9.2% unseen, 14.7% rarest-30%: specific numbers),
  gsm8k-style text is typical-like.

So the unifying "the mass rule prunes rare-token machinery, hence
memorization, arithmetic and rare facts fall together" story is only partly
right: rare-token use is enriched in the flat bulk, and arithmetic answers
are rare tokens, but memorized text prefers flat directions beyond what its
token rarity predicts. The pending skill probes (functional spectrum for
arithmetic / gsm8k; per-item damage) will show whether arithmetic's support is
memorization-like (diffuse, redundant) or rare-token-like (mid-band).

## 2026-09-04 (afternoon): skills on the same footing; a label-free detector

Probes on layer 24 with five populations (memorized; typical dolmino windows;
least-recited windows; synthetic few-shot arithmetic with the loss on the
answer digits; gsm8k-style math windows): `probe_band_damage2` (single-band
and joint-band removal, `band_damage_populations.png`), `probe_populations2`
(curvature attribution restricted to the target span, `skills_curvature.png`),
`probe_item_damage` (per-item damage under the two edited models).

**1. Arithmetic is not memorization-like; it is a two-sided object.**
* G side (output directions): arithmetic's normalised curvature share *rises*
  with depth (0.26 flattest -> 0.64 sharpest, Spearman +0.78), the opposite
  of memorized text (0.57 -> 0.38). Functionally, removing the sharpest G
  band costs arithmetic 0.12 / 0.21 nats (gate / up) against 0.02 for
  typical text and 0.02-0.03 for memorized; flat G bands cost it nothing;
  jointly removing the flattest 6 G deciles costs 0.021 (memorized 0.129).
* A side (input directions): arithmetic's share leans flat (0.57 -> 0.33,
  Spearman -0.44) and its function depends on the flat A bands *more than
  memorized text does*: flattest 6 A deciles jointly cost arithmetic 0.30 nats
  (memorized 0.12, typical 0.02); flattest 9: 0.68 (memorized 0.45).
* Its answer tokens have half the activation norm of typical tokens
  (||a||^2 median 618 vs 1221): the number features are low-variance input
  directions.
* Reading: arithmetic reads rare-token (number) features from flat input
  directions and writes through the sharpest, shared output directions.
  Memorization is diffuse and redundant on both sides. The paper's
  "arithmetic is brittle under curvature pruning" is therefore an A-side
  effect, not evidence that arithmetic is memorized. Prediction: pruning
  along output directions only (G-side flat bulk) removes memorization about
  as well while sparing arithmetic; pruning the A-side flat bulk destroys it.
  -> end-to-end test next (a side-restricted importance in the eval).

**2. gsm8k-style text: predictable, but robust.** Its curvature share leans
flat like memorized text's (G: 0.57 -> 0.46, Spearman -0.94) because its
tokens are predictable (loss 1.15, tiny gradients on most tokens), yet
removing flat bands costs it nothing (joint G flattest 9: +0.018 vs typical
+0.045; the real edits change its loss by +0.002). Curvature attribution
conflates *predictable* with *memorized*; the functional (removal) test does
not. The least-recited control (high loss) shows no trend at all, so the
flat-leaning curvature signature is a low-gradient signature, and only the
memorized population combines it with functional dependence on the flat bulk.

**3. Per-item damage is a usable label-free memorization detector.** Loss
increase under the rho=0.75 / 0.6 edits separates memorized items from
ordinary windows with AUC 0.83 / 0.90. Within ordinary text, damage rises with
how well the base model already recites a window (0.024 nats at greedy
accuracy < 0.4 -> 0.13 / 0.27 at 0.9-0.999), except for the perfectly recited
windows (1.4% of ordinary text, damage 0.046 / 0.14): those are predictable
by rule (tables, boilerplate) rather than memorized, and the edit tells the
two apart. Arithmetic answers: +0.06 / +0.21 nats; gsm8k text: +0.002.

**4. Norm signatures (true labels, target tokens).** Memorized: ||a||^2 ~0.9x
typical, ||g||^2 median 3000x smaller (5.6e-6 vs 1.7e-2). Arithmetic answers:
||a||^2 0.5x, ||g||^2 median 500x smaller. gsm8k text: ||a||^2 0.75x, large
gradients on its surprising tokens (leverage 5x typical).

## 2026-09-04 (evening): side-restricted edits, first point

`--corrections-side G|A` (commit 7e361c5): with the separable correction,
use only the output-direction (G) or input-direction (A) marginal and weight
the other side uniformly. First result, K-FAC dolmino at nominal rho=0.8,
G-side only (varus-flux): 18.01 ppl / 0.382 mem / 0.427 quotes / 0.658 GSM8K,
i.e. a very mild edit (GSM8K within 0.02 of the unedited 0.675). A nominal
rho is not comparable across importance definitions, so the G-only and A-only
curves are being traced at rho 0.6 / 0.45 (K-FAC) and 0.4 (Identity) to
compare at matched perplexity or forgetting. Layer sweep (input side, all 32
layers) in progress; early layers 16-18 show the sharpest input decile costing
arithmetic up to 1.1 nats, so arithmetic's input features are *sharp* in mid
layers and flat in layer 24: locality matters.

## 2026-09-04 (night): layer sweep, input side of all 32 MLP layers

`probe_layers.py` (coupled input covariances for every layer from one pass,
eigenbases per layer, attribution and band removal; `layers_sweep.png`).

* **Attribution is universal.** In every layer the memorized share of a
  direction's curvature falls with its depth (Spearman -0.92 to -0.98); the
  flattest half carries 0.60-0.66 of the normalised memorized share
  everywhere, rising to 0.71 in the last layer.
* **Function is local.** Projecting out the flattest 6 input deciles of a
  layer costs memorized text <= 0.02 nats in layers 0-17 (typical text
  0.005-0.010, ratio 1-2), then 0.03 -> 0.10 nats from layer 19 on while
  typical text stays at 0.01 (ratio 3-8, peak at layers 22-25 with 0.09 vs
  0.01). The memorization-specific support in flat input directions lives in
  the late third of the network, strongest exactly where the paper edits
  (22-25). The curvature signature and the functional dependence come apart:
  the former is everywhere, the latter is late.
* **Arithmetic is layered.** Its dependence on the *sharpest* input decile
  peaks in mid layers (layer 16: 1.09 nats; 14-15: 0.24-0.32; 12: 0.15),
  where typical text loses only 0.07-0.13; its dependence on the *flat* input
  bulk is late (layers 20-24: 0.15-0.46 nats, peak 0.46 at layer 22), and in
  layers 25-31 it drops to 0.03-0.07 while memorized text stays at 0.06-0.10.
  So number/operation features are sharp directions in the middle of the
  network and flat directions around layers 20-24.
* **Sharpest decile removal** hurts every population in every layer; in the
  last two layers it is catastrophic for all (the unembedding pathway).
  Random deciles cost < 0.01 nats everywhere.
* Design implication (to test): editing layers >= 26 instead of 23-25 would
  keep the memorization selectivity (flat-bulk damage ratio mem/typical 5-8)
  while halving the collateral on arithmetic (mem/arith ~2 instead of ~0.4);
  conversely the paper's layers 22-24 are the worst choice for arithmetic on
  the input side. Combined with the side result (arithmetic's output-side
  support is sharp), a late-layer, output-side-weighted edit is the natural
  candidate for "remove memorization, keep arithmetic".

Side-restricted, nominal-rho points so far (K-FAC 0.8): G-only (varus-flux)
18.01 / 0.382 / 0.427 / 0.658 (barely an edit); A-only (jammy-tils) 18.81 /
0.158 / 0.269 / 0.594, i.e. at matched perplexity +0.046 mem and +0.055 GSM8K
relative to the two-sided EK-FAC curve. Identity 0.6 A-only (fetid-esne):
mem 0.126 / quotes 0.232 / GSM8K 0.594 (perplexity lost to an OOM caused by
a probe on GPU 7; requeued as tumid-bang); the G-only twin (spiry-wont) died
of the same contention and is requeued as bushy-roup. The queue worker now
runs on GPUs 0-6 only. Verdict on sides waits for the rho sweeps.

Queued (2026-09-04 night): `model=7b_late` (layers 26-28; commit 61be68b):
marginal E-Identity at 0.6 / 0.45 (spiny-typo, fired-wont) and marginal EK-FAC
at 0.8 (twill-kine), to be compared with the 23-25 curves at matched
perplexity; output-side layer sweep (probe_layers_G) running.

---

## Backlog

Promising (benchmark-agnostic, next):
* **Breadth axis for the curvature spectrum.** Per direction: depth (mass) and
  breadth (participation entropy of per-sequence contributions). Conjecture:
  memorization = high depth from few contexts; rare skills = moderate depth,
  many contexts; general = both high. Test on held-out memorized vs clean
  Dolma text at matched depth. -> done for layer 24 (see 2026-09-03 evening):
  holds as a tendency, collinear with depth.
* **Per-direction attribution of curvature to token populations** (memorized
  vs clean, rare vs frequent, numbers / entities / function words), with the
  coupled weights; characterise the directions plain K-FAC drops but the
  corrected curvature keeps. -> done for layer 24: memorized share falls
  monotonically with depth; disagreement directions carry no class signature.
* **Locality:** input side done for all MLP layers (see 2026-09-04 night):
  memorization's functional support is late (>= 19), arithmetic's flat-input
  dependence is at 20-24 and its sharp-input dependence at 12-18. Still to do:
  G side per layer (needs per-layer gradient covariances), attention modules,
  and an end-to-end edit on layers >= 26.
* **Side-restricted pruning (from the arithmetic dissociation).** Test end to
  end whether pruning the G-side flat bulk only removes memorization while
  sparing arithmetic / GSM8K, and whether A-side-only pruning destroys it.
* Rare relations / closed-book facts as a population (the paper's other
  brittle task): do they look like arithmetic (flat inputs, sharp outputs) or
  like memorization (diffuse both sides)?
* Per-item damage as a detector: done at AUC 0.83-0.90; the perfectly
  recited-by-rule windows are the interesting false negatives to characterise.
* **Separate ||a||^2 from ||g||^2** in the leverage signature of memorized
  prefixes (3x the leverage of typical text at lower loss).
* **Training dynamics:** OLMo-2 intermediate checkpoints; when do low-breadth
  directions appear relative to memorization of specific sequences.
* Token weightings as a design space (retain/forget contrast, loss- or
  surprisal-weighted covariances): potentially strong for forgetting, but
  benchmark-adjacent; only pursue with held-out populations and a conceptual
  question attached.

Practical / engineering:
* Implement the norm-weighted accumulators in the bergson collector and
  retire the correction pass (saves ~45 min per collection); default the
  correction statistics to ~5% of the corpus.
* Report edits by kept fraction or matched perplexity rather than nominal rho
  (rho's effective sparsity depends on data size and basis/correction mix).
* `params.yaml` in git is a stale snapshot (foof commit); regenerate and
  commit to silence `dvc exp show` resolve errors on branch/workspace rows.

Deprioritised / closed:
* Eigenvalue-correction variants and E-Shampoo: mechanism closed.
* Sequence / document structure of the correction: refuted (unit-independent).
* Iterative re-estimation: effect at the noise floor (see 2026-09-03).
* Mild E-FOOF single-pass point (rho 0.85, flown-suss) has Dolma memorisation
  0.035 above its two-pass twin at identical ppl/quotes/GSM8K; unexplained,
  single point.

Open questions:
* Is the per-token leverage ||a_t||^2 ||g_t||^2 concentrated on few tokens, and
  which tokens? (numbers, rare tokens, document starts?)
* With sampled-label Fisher, confident (memorized) tokens have small gradients;
  how much of "memorized = flat" is this confidence effect rather than
  directional structure? Compare sampled vs true labels on the memorized set.
* Why does gsm8k calibration lift GSM8K for every method: which token
  populations shift the marginals?
