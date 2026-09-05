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

## 2026-09-04 (late): the output-side prediction holds end to end

G-side-only marginal corrections for K-FAC on dolmino (importance
C^2 x r_o, input directions weighted uniformly), compared with the two-sided
EK-FAC curve interpolated at the same perplexity:

| run | rho | ppl | mem | quotes | GSM8K | two-sided curve (mem / GSM8K) |
|---|---|---|---|---|---|---|
| stiff-food | 0.6 | 18.75 | 0.128 | 0.252 | 0.607 | 0.117 / 0.545 |
| lippy-tomb | 0.45 | 19.46 | 0.092 | 0.200 | 0.536 | 0.091 / 0.463 |

Same forgetting (mem within 0.01) and +0.061 / +0.073 GSM8K (5 sigma each);
at matched *forgetting* still +0.045 / +0.07. Pruning along output directions
only removes memorization as well as the two-sided rule and keeps the
arithmetic-bearing components, exactly as the band-removal probes predicted
(arithmetic reads flat input directions but writes through sharp output
directions; memorization is diffuse on both). A-side-only at 0.8 (jammy-tils,
18.81 / 0.158 / 0.269 / 0.594) is a milder edit (+0.046 mem at matched ppl)
with +0.055 GSM8K at matched ppl but only +0.01 at matched forgetting; the
A-only 0.6 / 0.45 points are running.

This is the first mechanism-derived change to the edit that moves the
forgetting/capability trade-off rather than its cost. Caveat: two points, one
basis, GSM8K only; the Identity-basis twins and the late-layer runs will say
whether it generalises.

## 2026-09-05 (early): the side rules measured on arithmetic itself

`eval_models_on_populations.py` (`models_on_populations.png`): edited models
materialised from the DVC cache, evaluated on the probe populations
(memorized suffix loss as the forgetting axis; synthetic-arithmetic answers
fully correct under greedy decoding as the capability axis; typical and
gsm8k-style window loss as collateral).

| rule | run | mem loss | arith acc | typical loss | two-sided arith at this mem |
|---|---|---|---|---|---|
| unedited | base | 0.018 | 0.892 | 2.137 | |
| two-sided EK-FAC 0.85..0.5 | ahead-fees .. rummy-drum | 0.38 .. 1.21 | 0.79 .. 0.15 | 2.17 .. 2.36 | |
| G-only 0.8 / 0.6 / 0.45 | varus-flux / stiff-food / lippy-tomb | 0.19 / 0.57 / 0.83 | 0.86 / 0.76 / 0.56 | 2.17 / 2.21 / 2.24 | - / 0.64 / 0.40 |
| A-only 0.8 / 0.6 | jammy-tils / couth-obit | 0.47 / 1.16 | 0.67 / 0.21 | 2.19 / 2.32 | 0.72 / 0.18 |
| Identity two-sided 0.6 | gamey-quad, unlet-genu | 0.63 / 0.62 | 0.74 / 0.78 | 2.21 | 0.60 |

* At equal forgetting, the G-only rule keeps 0.11-0.16 more arithmetic
  accuracy than the two-sided K-FAC rule (0.76 vs 0.64 at mem loss 0.57;
  0.56 vs 0.40 at 0.83), at the price of ~0.02 nats more typical-text loss.
  The A-only rule is at or below the two-sided curve (0.67 vs 0.72 at 0.47;
  telic-wind at rho 0.45 is a very harsh edit, ppl 24.5, mem loss 1.41,
  arithmetic 0.14, on the curve's extrapolation).
  This is the mechanism's own quantity, not GSM8K, and it matches the
  band-removal prediction: arithmetic writes through sharp output directions
  (kept by G-only) and reads flat input directions (removed by A-only and by
  the two-sided product).
* The Identity-basis edits (per-weight, no eigenbasis) sit above the
  two-sided K-FAC curve on arithmetic too (0.74-0.78 vs 0.60 at mem 0.62),
  consistent with E-Identity's strong GSM8K in the frontier: the K-FAC
  A-eigenbasis is what makes the edit hit the number features.
* gsm8k-style text loss barely moves for any edit (1.12-1.25 vs 1.15), so
  GSM8K accuracy changes are not about this text's likelihood.

## 2026-09-05: output-side layer sweep; Identity G-only

**Output side (probe_layers_G, `layers_sweep_both.png`).** Coupled output
covariances for all 32 layers (both modules), flattest 6 deciles of output
directions projected out per layer:
* Memorized text's functional dependence on flat output directions is late,
  like the input side: ratio to typical text ~0.5-1 in layers 0-12 (flat
  output removal there hurts typical text *more*), rising from layer 16 to a
  plateau of 3-5 at layers 20-27 (peak 5.3 at 24-25; absolute 0.13-0.25 nats
  at 20-24 with typical at 0.02-0.03), then 2-2.5 in the last layers.
* Arithmetic uses flat output directions at layers 18-23 (0.11-0.31 nats,
  peak 0.31 at layer 22) but hardly at 24-31 (0.02-0.06). So the earlier
  layer-24 statement "arithmetic writes through sharp outputs" holds at 24-25
  and is *not* general: at 20-23 arithmetic sits in the flat output bulk too.
* The G-only rule's advantage in the 23-25 edits therefore comes from a
  favourable coincidence at those layers: memorization's output-side support
  is at its peak while arithmetic's is at its minimum, whereas on the input
  side arithmetic's flat dependence peaks exactly there (0.23-0.46 nats at
  22-24). Prediction for the layer-26-28 runs: two-sided editing should already
  spare arithmetic much better than at 23-25 (both sides have mem/arith >= 1.5
  there).
* The sharpest output decile of layers 13, 15 and 17 is critical for every
  population (0.3-1.2 nats); attribution there is also weakest (Spearman
  -0.6 to -0.7 at 11-13), so mid-network output geometry is dominated by a few
  shared directions.

**Identity basis, G-only (zinky-pons, rho 0.4):** 19.47 / 0.082 / 0.187 /
0.590 vs the two-sided E-Identity curve at matched perplexity 0.083 / 0.533:
same forgetting, +0.057 GSM8K. The output-side rule generalises to the
per-weight basis (identity rows = output units, columns = input features).
On the population axes the gain is smaller than for K-FAC: E-Identity's own
two-sided curve is already arithmetic-friendly (mem loss 0.31 / 0.62 / 0.89 /
1.03 -> arithmetic accuracy 0.84 / 0.76 / 0.62 / 0.37 for rho 0.75 / 0.6 /
0.45 / 0.3, vs the K-FAC two-sided curve's ~0.60 at 0.62), and G-only at 0.4
(mem 0.85 -> 0.67) sits ~0.03 above it. The K-FAC A-eigenbasis is what makes
the two-sided K-FAC edit hit the number features; per-weight pruning does not
align with them.

## 2026-09-05: forward attribution and an edit-free detector

Curvature attribution is gradient-based and confidence-blind, so it
understated memorized text's functional dependence (2026-09-03). A second,
forward quantity: the energy an item's tokens send through each direction of
the weights, E_G[o] = sum_t (q_o^T W a_t)^2 and E_A[i] = ||W p_i||^2 sum_t
(p_i^T a_t)^2 (`probe_forward.py`, layer 24).

* Per band, forward-energy share tracks the functional damage of removing
  that band about as well as curvature share does on the A side (Spearman
  0.8-0.9) and better where curvature failed (memorized text, gate output
  side: 0.84 vs 0.24).
* Population level, input side: memorized text sends 0.35 / 0.44 (gate / up)
  of its input energy through the flattest 60% of input directions, typical
  text 0.13 / 0.21, arithmetic answers 0.27 / 0.37, gsm8k text 0.18 / 0.27.
  Output side: ~0.5-0.6 for everyone (most MLP output energy is in
  low-curvature output directions, which the downstream network barely reads
  for typical predictions).
* **Item level:** the "private share" (input energy in the flattest 60% of
  directions / total) separates the 1054 memorized items from all 9216
  ordinary dolmino windows with AUC 0.986 (both modules, layer 24), from a
  single forward pass: no gradients, no edit, no labels beyond the population
  eigenbasis. The edit-based detector reached 0.83-0.90. Validation across
  layers with random-basis and plain-covariance controls and loss/rarity
  confounds is running (`probe_private_share.py`).

Theory sketch (2026-09-05): a **public/private decomposition of weight space
by usage**. The population Fisher's sharp directions are shared computation
(public); its flat bulk is many low-usage, near-orthogonal directions where
item-specific information can persist because population gradients rarely
touch them (private). General text routes through public directions on both
sides; memorized items route a distributed, redundant code through private
directions on both sides (single-band robustness, bulk fragility,
superadditivity); rare skills route through private inputs and public
outputs (arithmetic); the split is functional only in the late third of the
network. Curvature (gradient-weighted usage) defines the split; forward energy
(activation-weighted routing) says where an item's computation goes; the
detector is the overlap of the two.

## 2026-09-05: superadditivity is a margin effect (probe_margin, layer 24)

Jointly removing the flattest k input deciles (both modules) and tracking
confident tokens (base loss < 0.1; 96% of memorized suffix tokens, 20% of
typical suffix tokens):
* Memorized: flip fraction (loss > 1 after removal) 0.002 / 0.009 / 0.028 /
  0.069 for k = 2 / 4 / 6 / 8, i.e. roughly x3 per two deciles; the median
  loss change stays ~0 while the 90th percentile grows 0.006 -> 0.45. Flips
  concentrate in the lowest base-margin quartile (k=8: 0.146 vs 0.024 in the
  highest; k=6: 0.073 vs 0.004).
* Typical confident tokens: flips 0.000 / 0.001 / 0.005 / 0.031, with no
  margin structure (0.019-0.046 across quartiles at k=8).
Reading: a memorized token's prediction is a sum of many small contributions
from flat directions; removing bands erodes its logit margin gradually and the
loss jumps only when the margin is crossed, so the population-level damage is
superadditive and the low-margin items go first. Typical predictions do not
draw their margin from the flat bulk. This is the "distributed code" account
of memorization made quantitative, and it suggests the per-item margin drawn
from private directions as a graded memorization strength.

Identity A-only at 0.4 (sandy-delf): 19.87 / 0.068 / 0.170 / 0.494 vs the
two-sided E-Identity curve at matched ppl 0.071 / 0.484: no GSM8K gain
(+0.009), unlike G-only (+0.057); on the populations mem loss 1.04 ->
arithmetic 0.43 (two-sided curve ~0.37 at that forgetting). Side asymmetry
confirmed in the identity basis too. The three layer-26-28 runs failed at
collection because the queue worker was restricted to GPUs 0-6 while the
FSDP collection needs 8; requeued as dotty-orfe / modal-tier / brute-doll,
worker to restart on all GPUs after the checkpoint probes.

## 2026-09-05: the private-share detector validated (probe_private_share)

Private share of an item = fraction of its target tokens' input activation
energy (weighted by ||W p_i||^2, both modules) that flows through the flattest
60% of a layer's input directions, directions ordered by the coupled
population covariance. One forward pass; no gradients, labels or edits.
`private_share_detector.png`.

| layer | AUC coupled | AUC plain cov. | AUC random basis | mean share mem / windows / arith |
|---|---|---|---|---|
| 4 | 0.876 | 0.880 | 0.311 | 0.26 / 0.20 / 0.17 |
| 12 | 0.906 | 0.893 | 0.250 | 0.21 / 0.14 / 0.14 |
| 18 | 0.984 | 0.978 | 0.269 | 0.22 / 0.12 / 0.18 |
| 20-28 | 0.992-0.995 | 0.986-0.992 | 0.32-0.44 | 0.24-0.36 / 0.12-0.18 / 0.22-0.33 |
| 30 | 0.960 | 0.944 | 0.244 | 0.15 / 0.07 / 0.11 |
| 20-28 combined | 0.994 | | | |

* AUC 0.994 against all 9216 ordinary dolmino windows; the edit-based
  detector reached 0.83-0.90. Plain (unweighted) covariance ordering is
  nearly as good; a random orthonormal basis gives 0.25-0.44 (memorized
  activations are more anisotropic, so the control is not neutral, but it is
  nowhere near the coupled ordering).
* Confounds: in layers 20-28 the score over ordinary windows is uncorrelated
  with base loss (0.00-0.02) and greedy accuracy (~0), weakly negatively
  correlated with token frequency (-0.13 to -0.19). Early layers' score
  partly tracks difficulty (+0.24 with loss at layer 4): the detector is a
  late-layer property, like the functional support.
* Ordinary windows the model recites (greedy accuracy >= 0.9, 2.7% of text)
  score 0.22-0.25 vs 0.16 for the rest and 0.34 for the memorized set: the
  score grades recitation within ordinary text too. (Contrast the damage
  detector, which found the perfectly recited windows robust: they route
  through private inputs yet survive the edit, presumably via public outputs.)
* Arithmetic answers score 0.29-0.33 in late layers, close to memorized
  text: "reads rare features from private input directions" is exactly what
  the score measures; the two are told apart by the output side (and by
  function).

Theory status: the public/private split (curvature-defined) plus forward
routing (activation-defined) now yields (i) the selectivity of the mass rule,
(ii) the side/layer structure of skills, (iii) a graded, edit-free
memorization score. Next: dynamics (checkpoints, running), and a formal
statement with predictions.

Write-ups in progress (scratchpad `tex/`): `memorization_notes.tex` (long
notes, 5 pp.) and `key_findings.tex` (short, 3 pp.); both compile, each with
one pending item (layer-26-28 edits; checkpoint dynamics).

## 2026-09-05: dynamics across training checkpoints (`probe_checkpoint`, `dynamics.png`)

Stage-1 checkpoints at 1.26T / 2.10T / 2.94T / 3.78T tokens (the hub's
step-101k and step-300k branches serve byte-identical weights, an upstream
mislabel, so only four distinct stage-1 points) plus the final model; each
checkpoint's own coupled covariances define its eigenbases.

| tokens | mem strict recitation | mem suffix acc | arith acc | L24 AUC | private share mem / windows |
|---|---|---|---|---|---|
| 1.26T | 0.53 | 0.952 | 0.47 | 0.987 | 0.343 / 0.156 |
| 2.10T | 0.52 | 0.958 | 0.58 | 0.988 | 0.350 / 0.164 |
| 2.94T | 0.56 | 0.969 | 0.63 | 0.990 | 0.354 / 0.169 |
| 3.78T | 0.64 | 0.980 | 0.66 | 0.990 | 0.358 / 0.180 |
| final | 0.99 | 1.000 | 0.89 | 0.990 | 0.366 / 0.189 |

* The memorized set is already 95% recited per token at 1.26T (strict
  recitation only 0.53 because a few tokens per item are missed); verbatim
  completion happens mostly in stage 2 (0.64 -> 0.99). Arithmetic keeps
  improving through stage 1 and jumps in stage 2.
* The public/private split and the detector are fully formed at 1.26T (AUC
  0.987) and barely change afterwards.
* Per-item private share is a stable property (Spearman 0.90 for memorized
  items, 0.91 for windows, between 1.26T and final) and does **not** predict
  which items will be recited later (Spearman with recitation gain -0.06 /
  -0.10; items not yet recited at 1.26T have slightly *lower* share).
* But it tracks the memorization state: ordinary windows that are not recited
  at 1.26T and are recited by the final model (n=209) move from share 0.13
  (typical) to 0.25; never-recited windows move 0.16 -> 0.19; already-recited
  ones 0.30 -> 0.33. Memorizing an item during training means routing its
  inputs into the private subspace; the private share is a consequence of
  memorization, not a precursor.

Theory reading: the private subspace exists from early in training (it is a
property of the population geometry), and items enter it as they are
memorized. Which items get memorized is decided by something else (exposure,
duplication); where they are stored is decided by the geometry.

Identity-basis side rules, remaining points (2026-09-05): G-only at 0.6
(bushy-roup) 18.66 / 0.202 / 0.253 / 0.632 is a milder edit than the two-sided
0.6 (+0.061 mem at matched ppl, +0.015 GSM8K); on the populations it sits on
the two-sided identity curve (mem loss 0.53 -> arithmetic 0.80). Together with
zinky-pons (0.4: +0.057 GSM8K at equal forgetting) the identity G-only gain is
real at stronger edits only. The requeued A-only 0.6 (tumid-bang) hit DVC's
run-cache and reused fetid-esne's eval output, so its perplexity is still
missing (mem 0.126 / quotes 0.232 / GSM8K 0.594); the model is the same and
its population metrics are fetid-esne's: mem loss 0.644 -> arithmetic 0.72,
i.e. on or slightly below the two-sided identity curve (0.74-0.78 at 0.62-0.63),
while G-only at the same nominal rho is milder (0.53 -> 0.80). Identity
A-only 0.4 (sandy-delf): 1.04 -> 0.43 (curve ~0.37). In the identity basis the
side asymmetry is weaker than for K-FAC but has the same sign.

## 2026-09-05: projected continued training (probe_projected_ft), first pass

Sixty normalised SGD steps on ordinary dolmino text (total update norm 0.3 per
step across the six gate/up matrices of layers 23-25, ~0.1% of a matrix norm
per step, ~5% cumulative), restricted to the private-private block, the
public-public block, or unrestricted, plus matched-norm Gaussian noise in each
block. Result: nothing moves. Unrestricted SGD improves typical loss
2.137 -> 2.122 and lowers strict recitation 0.99 -> 0.97; private-only SGD
leaves recitation at 0.99; public-only 0.97; noise of the same norm in either
block changes nothing (0.984-0.992). (A first attempt with a raw learning rate
of 2e-4 had updates below bf16 resolution and did nothing at all.)

Reading: at perturbation norms an order of magnitude below a band removal
(which zeroes weight energy of order tens per matrix), the memorized code is
robust in every subspace, as the margin picture predicts; the discriminating
regime is large perturbations. A graded noise sweep at matched norm (5 to 120
per matrix, private vs public vs isotropic) is running to map where memorized
and typical text break.

## 2026-09-05: layers 26-28, first point (dotty-orfe: identity, marginal, rho 0.6)

18.94 / 0.260 / 0.307 / 0.663 vs the 23-25 E-Identity curve at matched
perplexity: mem 0.118, GSM8K 0.590. The late-layer edit keeps GSM8K near the
unedited 0.675 but removes far less memorization per unit perplexity (+0.14
mem). On the populations: mem loss 0.39 -> arithmetic 0.815 with typical loss
2.209, whereas the 23-25 identity curve at the same forgetting has arithmetic
~0.83 and typical loss ~2.19. So at matched forgetting the 26-28 identity edit
spares no more arithmetic and costs more clean-text loss: the layer-choice
prediction from the input-side sweep does not hold for the per-weight edit.
The sweep's own numbers already said 22-25 is where the flat bulk is most
selective for memorization (ratios 8-9 vs 3-5 at 26-28); what I predicted was a
better arithmetic/forgetting trade-off there, and that is not what the
per-weight rule delivers. Confirmed by modal-tier (identity 26-28, rho 0.45): 19.59 / 0.200 / 0.238 /
0.641, i.e. +0.12 mem and +0.12 GSM8K at matched perplexity, but at matched
forgetting (mem ~0.20) the 23-25 curve already gives GSM8K ~0.63: the late-layer
per-weight edit just buys less forgetting per unit perplexity. K-FAC 26-28
(brute-doll, marginal 0.8): 18.55 / 0.286 / 0.366 / 0.644, i.e. +0.155 mem and
+0.08 GSM8K at matched perplexity; on the populations mem loss 0.34 ->
arithmetic 0.840 with typical loss 2.186, vs the 23-25 EK-FAC curve's ~0.81 /
~2.165 at that forgetting: +0.03 arithmetic for +0.02 nats of clean-text loss.
modal-tier on the populations: 0.62 -> 0.754 / 2.242 vs the identity curve's
0.76-0.78 / 2.213. Verdict: moving the edit to layers 26-28 is a weak lever at
best (K-FAC) and a loss (identity); the side rule is the effective one.

## 2026-09-05: the private block as a container (probe_noise_scale, `noise_sweep.png`)

Gaussian noise of matched Frobenius norm per matrix, applied to all six gate/up
matrices of layers 23-25 at once, confined to the private-private block
(flattest 60% x 60%; it holds ~76 of each matrix's ~128 norm), the public
block (sharpest 40% x 40%; ~55) or isotropic:

| norm | private: strict / typical dloss / arith | public: strict / dloss / arith | isotropic |
|---|---|---|---|
| 30 | 0.84 / +0.003 / 0.887 | 0.77 / +0.014 / 0.882 | 0.82 / +0.007 / 0.892 |
| 60 | 0.43 / +0.012 / 0.887 | 0.28 / +0.059 / 0.850 | 0.40 / +0.026 / 0.873 |
| 90 | 0.18 / +0.028 / 0.874 | 0.04 / +0.147 / 0.751 | 0.11 / +0.060 / 0.823 |
| 120 | 0.07 / +0.050 / 0.833 | 0.03 / +0.312 / 0.591 | 0.05 / +0.120 / 0.738 |

* Per unit norm, public noise breaks everything faster, memorization
  included; but per unit *collateral*, private noise is far more selective:
  at ordinary-text damage ~+0.03 nats, private noise leaves strict recitation
  at 0.18, isotropic at ~0.40, public at ~0.6.
* Random private noise at norm 120 reaches the same recitation removal as the
  rho=0.6 curvature edit (strict 0.07 vs 0.075; memorized loss 0.70 vs 0.63)
  with *less* collateral: typical loss +0.050 vs +0.076 and arithmetic 0.833
  vs 0.742 (identity edit) / ~0.60 (K-FAC two-sided at that forgetting).
* Reading: the subspace is what matters, not the ranking within it. Zeroing
  the flat components (pruning) removes memorization but also systematically
  removes arithmetic's flat-input components; random perturbation of the same
  block scrambles the memorized code (which needs many coordinates to agree)
  while leaving any individual public-facing computation mostly intact. This
  is the theory's container prediction in a graded, data-free form, and a
  candidate method: label-free, no correction pass, no C^2 ranking.
* End-to-end check queued: the perturbed models (private 60 / 90 / 120, public
  30) saved under out/aux and run through the pipeline's eval + GSM8K with
  --start-from-model and rho = 1 (no further edit).

## 2026-09-05: attention projections (probe_attention; layers 20, 24, 28)

Coupled input covariances and eigenbases for q/k/v/o, private-share detector on
each module's input, flat-bulk (60%) and sharpest-decile removal per module.
* Detector: the residual stream read by q/k/v carries the signature (AUC
  0.98-0.99, mean private share mem 0.27-0.47 vs windows 0.10-0.29); the
  attention output read by o_proj does not (AUC 0.51-0.86, shares ~0.04-0.15).
* Function: projecting out the flattest 60% of any attention projection's
  input costs memorized text only +0.003-0.006 nats (typical +0.001;
  arithmetic +0.01-0.03, except o_proj at layer 20: +0.16), against
  0.13-0.25 nats for the MLP inputs of the same layers. The sharpest decile
  matters for everyone (mem 0.12-0.49, typical 0.06-0.27, arithmetic up to
  0.48 at layer 20).
* Reading: the private-direction *signature* is a property of the residual
  stream and is visible wherever it is read, but the memorized *function*
  lives in the MLP weights, not in the attention projections' flat input
  directions. This localises the container to the MLPs (consistent with the
  key-value-memory view of MLPs) and means attention weights can be left
  alone by unlearning edits.

## 2026-09-05: the private share as a graded memorization strength

Within the memorized set, an item's private share (layer 24 input, or the mean
over layers 20-28) predicts how much the curvature edits damage it: Spearman
+0.50 with the loss increase and +0.47 with the accuracy drop under the
rho=0.75 edit (+0.48 / +0.45 under rho=0.6). Over ordinary windows the
correlation is +0.15 (most have nothing memorized to lose). So the score is not
only a detector but a per-item measure of how much of the item's support sits in
the private block, hence of its vulnerability to private-subspace edits.
Extractability (shortest prefix, of 8/16/24/32/48/64 tokens, from which greedy
decoding recites >= 90% of the suffix): the memorized items spread evenly over
the range (20% recite from 8 tokens, 16% need all 64). The private share does
*not* track extractability (Spearman +0.06 with the minimal prefix; -0.15 with
recitation from an 8-token prefix), while the minimal prefix does correlate
with edit damage (+0.30; the shallowly triggered items are the fragile ones)
and the private share correlates with damage more strongly (+0.50). Two
distinct dimensions of memorization, then: how much context an item needs to
be triggered, and how much of its support sits in the private block. The
private share measures storage, not trigger ease.

First benchmark point of the container test (naive-vang, private noise norm
60 on layers 23-25): 17.92 ppl / 0.604 Dolma loose accuracy / 0.570 quotes
strict / 0.675 GSM8K, i.e. a mild edit with GSM8K at the unedited level; the
90 and 120 points are running.

## 2026-09-05: container test on the benchmarks (batch 9)

Private-block noise on layers 23-25 through the standard pipeline (unedited
model: ppl 17.62, Dolma loose acc 0.998, quotes strict 0.967, GSM8K 0.675):

| run | norm | ppl | Dolma | quotes | GSM8K | nearest curvature edits at similar forgetting |
|---|---|---|---|---|---|---|
| naive-vang | 60 | 17.92 | 0.604 | 0.570 | 0.675 | (milder than any) |
| telic-iota | 90 | 18.36 | 0.254 | 0.375 | 0.662 | EK-FAC 0.85: 18.06 / 0.224 / 0.308 / 0.606; E-Identity 0.75: 18.10 / 0.302 / 0.355 / 0.643 |
| funky-cere | 120 | 19.07 | 0.126 | 0.250 | 0.651 | EK-FAC ~0.78: ~18.6 / 0.13 / 0.22 / ~0.56; E-Identity 0.6: 18.63 / 0.144 / 0.235 / 0.621; G-only K-FAC 0.6: 18.75 / 0.128 / 0.252 / 0.607 |

* At matched forgetting the private-noise edit keeps GSM8K higher than every
  curvature edit (0.651 vs 0.56 two-sided, 0.61 G-only, 0.62 E-Identity at
  Dolma ~0.13) and stays within 0.03 of the unedited model.
* It costs more pile10k perplexity: +0.3-0.45 relative to the curvature edits
  at matched forgetting (19.07 vs 18.6-18.75). This contradicts the
  population-level result, where the noise edit had *less* in-distribution
  (dolmino) loss than the curvature edits (+0.050 vs +0.076 nats): the noise
  hurts out-of-distribution text (Pile) more and in-distribution text less.
  Reading: pruning removes only components that are both flat and small
  (C^2-weighted), leaving large private components that other distributions
  rely on; noise perturbs the whole private block indiscriminately, including
  large components. "The subspace matters, not the ranking" is therefore
  half right: the ranking protects out-of-distribution text, the subspace
  choice protects arithmetic/GSM8K. A natural hybrid is noise scaled inversely
  to component magnitude within the private block.
* Public-noise control (ovoid-pope, norm 30 in the public block): ppl 18.00,
  Dolma 0.832, quotes 0.709, GSM8K 0.666. Against private noise at norm 60
  (naive-vang): +0.38 vs +0.30 perplexity for a quarter of the forgetting
  (Dolma -0.17 vs -0.39), i.e. about 3x less forgetting per unit of perplexity
  cost at half the norm, and it is the only noise model that dents GSM8K
  (-0.009). The benchmark pipeline reproduces the population-level asymmetry:
  per unit norm, public noise is the expensive one.

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
* **Locality:** done on both sides for all MLP layers and for the attention
  projections of layers 20/24/28: memorization's functional support is late
  and in the MLPs; attention inputs carry the signature but no memorized
  function in their flat directions; late-layer edits are a weak lever.
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

## 2026-09-05: does the shape of the noise matter? (question from Guillaume)

So far every private-block perturbation was Gaussian and isotropic *within*
the block. The block is a crude indicator function; inside it, general-text
curvature spans orders of magnitude and the weight components span orders of
magnitude, and the noise ignores both. Second-order theory says the expected
collateral of random noise with per-coordinate variance s_oi is
(1/2) sum s_oi^2 d_o d_i (d = general-text depth in the K-FAC basis), while
the hit on a memorized item is the same sum with the item's own depths. The
best shape at fixed expected collateral therefore concentrates variance where
mem depth / general depth is largest, not uniformly over the block. Beyond
second order, the margin mechanism makes the *distribution* of per-item shifts
matter: low-rank noise of the same Frobenius norm gives heavy-tailed shifts
across items (a few large, most tiny), full-rank Gaussian gives Gaussian ones.
And the Pile result suggests a third axis: whether the perturbation adds energy
to empty coordinates (noise) or only removes existing energy (pruning, shrink).

Probe (`probe_noise_shape.py`, layers 23-25 gate/up, matched norm 90 / 120,
metrics add Pile loss as the out-of-distribution proxy): iso, inverse-curvature,
inverse-magnitude, small-components-only, large-components-only, rank-32,
shrink (deterministic), prune-like (deterministic removal by importance), and
the oracle ratio shape (mem depth / general depth) inside the block and over
the whole matrix. Predictions: inverse-magnitude and small-only should recover
pruning's Pile behaviour; ratio should forget most per unit collateral; shrink
should forget little (uniform scaling of the block barely crosses margins).
Queued behind ovoid-pope.

## 2026-09-05: noise shape, round 1 -- the shape inside the block does not matter, removal does

`probe_noise_shape.py`, layers 23-25 gate/up, private block = flattest 60% x
60% (general-text depth), matched Frobenius norm per matrix, run as 8 shards
(one GPU each, ~8 min wall). Base: mem loss 0.018 / strict 0.99, typical
2.137, Pile 2.348, arith 0.892. Block norms: ||W|| ~ 130, ||W_priv-priv|| ~ 76
(34% of the energy in 36% of the coordinates), median |W~| 0.013.

| shape (norm 90) | mem loss / strict | typical | Pile | arith |
|---|---|---|---|---|
| iso | 0.335 / 0.159 | +0.026 | +0.047 | 0.868 |
| inverse-curvature | 0.329 / 0.183 | +0.027 | +0.044 | 0.862 |
| inverse-magnitude | 0.348 / 0.153 | +0.028 | +0.045 | 0.869 |
| small components only | 0.344 / 0.176 | +0.026 | +0.045 | 0.870 |
| large components only | 0.334 / 0.185 | +0.027 | +0.045 | 0.869 |
| rank 32 | 0.334 / 0.177 | +0.027 | +0.046 | 0.884 |
| oracle ratio (mem/general depth), in block | 0.353 / 0.165 | +0.027 | +0.046 | 0.871 |
| oracle ratio, whole matrix | 0.401 / 0.110 | +0.045 | +0.066 | 0.849 |
| remove the whole block (shrink alpha=1, norm 76) | 0.318 / 0.180 | +0.024 | **+0.027** | 0.857 |

Norm 120 tells the same story (Gaussian shapes: mem 0.66-0.70 / 0.07-0.08,
typical +0.047-0.050, Pile +0.081-0.087, arith 0.83-0.85; whole-matrix ratio
0.804 / +0.090 / +0.148 / 0.776).

* **Every Gaussian shape confined to the block is equivalent**, to within
  noise: inverse-curvature, inverse-magnitude, small-only, large-only, rank 32
  and even the oracle shape targeted by the memorized set's own depth all give
  the same forgetting and the same collateral as isotropic noise of the same
  norm. Second-order reasoning: an item's logit shift under noise with
  per-coordinate variance s_oi^2 has variance sum s_oi^2 u_o^2 v_i^2 (u, v =
  the item's coordinates); if the item's energy is spread evenly over the
  block, every shape with the same total variance gives the same shift.
  The invariance is therefore direct evidence that the memorized code has
  no preferred sub-directions inside the private block -- the distributed
  code of the band-removal experiments, seen from the noise side. It also
  says the "magnitude ranking protects OOD text" explanation from batch 9 is
  wrong: inverse-magnitude and small-only noise cost Pile exactly what
  isotropic noise costs.
* Leaving the block hurts: the same oracle shape over the whole matrix has
  1.7x the typical and 1.4x the Pile collateral for less forgetting. The
  block, not the shape, is the lever.
* **Removal is different from addition.** Zeroing the entire private-private
  block of six matrices (34% of their energy) gives the same forgetting and
  the same in-distribution collateral as norm-90 noise but *half* the Pile
  collateral (+0.027 vs +0.047), with arithmetic at 0.857. So the reason
  pruning spares out-of-distribution text is not the ranking but that it
  removes existing structure instead of adding random structure: Pile text
  (private relative to dolmino) has little function in W_priv-priv but is
  exposed to any random content placed there. Round 2 (running) matches the
  norms: noise at 38/57/76 vs shrink at alpha 0.25/0.5/0.75 vs prune-like
  removal, plus sign flip (norm 152), a random permutation of the block
  entries (same energy, no structure), and the same operations on the public
  block as the control.

## 2026-09-05: noise shape, round 2 -- matched norms, and the private/public asymmetry of removal

Same setup, 8 shards, ~6 min wall. Block norms per matrix: private-private ~76,
public-public ~55 (||W|| ~130).

| condition (private block) | norm | mem loss / strict | typical | Pile | arith |
|---|---|---|---|---|---|
| iso noise | 38 | 0.037 / 0.789 | +0.004 | +0.008 | 0.892 |
| shrink alpha=0.5 | 38 | 0.045 / 0.731 | +0.005 | +0.000 | 0.887 |
| prune-like removal | 38 | 0.040 / 0.742 | +0.004 | +0.003 | 0.887 |
| iso noise | 57 | 0.091 / 0.532 | +0.010 | +0.018 | 0.882 |
| shrink alpha=0.75 | 57 | 0.129 / 0.433 | +0.013 | +0.010 | 0.874 |
| prune-like removal | 57 | 0.113 / 0.446 | +0.011 | +0.010 | 0.879 |
| iso noise | 76 | 0.208 / 0.282 | +0.019 | +0.033 | 0.878 |
| remove block (alpha=1) | 76 | 0.318 / 0.180 | +0.024 | +0.027 | 0.857 |
| iso noise | 90 | 0.335 / 0.159 | +0.026 | +0.047 | 0.868 |
| iso noise | 107 | 0.522 / 0.098 | +0.037 | +0.068 | 0.859 |
| shuffle block entries | 107 | 0.686 / 0.071 | +0.044 | +0.068 | 0.843 |
| negate block (-2 W) | 152 | 1.254 / 0.049 | +0.089 | +0.153 | 0.695 |

| condition (public block) | norm | mem loss / strict | typical | Pile | arith |
|---|---|---|---|---|---|
| iso noise | 45 | 0.087 / 0.538 | +0.032 | +0.029 | 0.868 |
| shrink alpha=0.5 | 45 | 0.219 / 0.214 | +0.116 | +0.135 | 0.779 |
| remove block (alpha=1) | 55 | 0.492 / 0.054 | +0.211 | +0.254 | 0.647 |
| iso noise | 90 | 0.756 / 0.044 | +0.167 | +0.204 | 0.723 |

* **At matched norm, removing private content forgets ~1.5x more than random
  noise** (mem loss 0.318 vs 0.208 at 76; 0.129 vs 0.091 at 57) at slightly
  higher in-distribution cost (+0.024 vs +0.019) and lower Pile cost. At
  matched forgetting (removal 76 vs noise 90): same typical cost, half the
  Pile cost. Prune-like removal (smallest importance first) sits between
  shrink and noise -- ordering inside the block is again irrelevant.
  Reading: removal displaces every memorized item's margin coherently (the
  block's whole contribution to the item's logit is taken away), noise
  displaces it with a random sign of the same mean square; and the removal
  direction -W_pp is a low-Pile-curvature direction, while any random
  direction of the same norm is not. Shuffling the block entries (remove +
  add equal random energy) costs Pile exactly what noise of the total norm
  costs: the Pile cost is set by the random part.
* Negating the block is as bad as noise of the same norm (slightly worse):
  the structure helps only when it is removed, not when it is flipped.
* **The sign of the removal-vs-noise effect flips in the public block.** There,
  shrinking by half costs 3.6x more typical loss and 4.7x more Pile loss than
  noise of the same norm (+0.116 vs +0.032), and removing the block outright
  (+0.211) is worse than noise of 1.6x the norm (+0.167). At matched
  forgetting the public block's removal costs ~2.3x what noise costs; the
  private block's removal costs 1.0x (typical) and 0.6x (Pile).
* This gives an operational distinction between *stored content* and
  *computation* that does not use labels: a block holds content if removing
  it costs less than random noise of the same norm (the block's own
  structure is idiosyncratic, so taking it away harms only its owners); it
  holds computation if removing it costs more than noise (the structure is
  what everyone uses). Private = content, public = computation, measured on
  the same six matrices.
* Consequence for editing: the best label-free edit is not noise but
  deterministic removal of the private block. Six matrices, zero their
  private-private block (34% of their energy): recitation 0.99 -> 0.18,
  typical +0.024 nats, Pile +0.027, arithmetic -0.035. Materialising this
  model (and a 10-layer version, 19-28) for the benchmark pipeline.

## 2026-09-05: private-block deletion models for the benchmark pipeline

`make_removal_models.py <name> <layers> [alpha]`: coupled per-token covariances
(1152 dolmino sequences, sampled labels) for gate/up of the given layers,
eigenbases, flattest 60% x 60% block deleted (alpha=1) or halved (alpha=0.5).
Saved under out/aux/model_<name>/model (not DVC deps), evaluated on the
populations before queueing:

| model | mem loss / strict | typical | Pile | arith |
|---|---|---|---|---|
| unedited | 0.018 / 0.990 | 2.137 | 2.348 | 0.892 |
| remove23_25 (delete, layers 23-25) | 0.295 / 0.204 | +0.021 | +0.027 | 0.827 |
| remove23_25_half (alpha=0.5) | 0.044 / 0.733 | +0.005 | +0.000 | 0.882 |
| remove19_28 (delete, layers 19-28) | 1.21 / 0.035 | +0.093 | +0.152 | 0.449 |
| remove23_28 (delete, layers 23-28) | -- / 0.074 | +0.048 | +0.083 | 0.759 |

* The coupled-basis deletion of 23-25 reproduces the K-FAC-basis probe
  (0.318 / 0.180, +0.024, +0.027, 0.857) to within a few thousandths; the
  basis construction does not matter either.
* Deleting the private block of ten layers (19-28) removes recitation almost
  completely but halves arithmetic (0.892 -> 0.449) and costs Pile +0.15: the
  private *input* directions of layers 20-24 are where arithmetic reads its
  number features (layer sweep of 2026-09-04), and deletion takes those
  out with everything else. Deletion is only safe where the private block
  holds no skill -- which is what the side and layer results said all along.
  A 23-28 variant is being built to extend forgetting without touching the
  arithmetic layers.
* remove23_28 lands at the forgetting of norm-120 private noise (strict
  0.074 vs 0.083) with the same typical/Pile cost (+0.048/+0.083 vs
  +0.047/+0.087) but lower arithmetic (0.759 vs 0.843): layers 26-28 add
  forgetting only at the price of arithmetic, in line with the weak late-layer
  lever of 2026-09-05. Deletion's advantage over noise is a three-layer
  result so far.
* Queue (batch 10, same pipeline as batch 9: rho=1, start_from_model):
  sorer-rein = remove23_25 (ran on the old OLMES benchmark stage), then,
  re-queued on the new olmo-eval stage: dicey-ribs = remove23_25_half, agape-wool =
  remove23_28, fussy-adze = remove19_28 (negative control).

## 2026-09-05: benchmark stage moved from OLMES (HF backend) to olmo-eval (vLLM)

Guillaume: OLMES is slow and unmaintained; switching its own vLLM backend gave
him incomparable scores, and no pinning to old versions. Findings:

* The old stage ran oe-eval with the HF backend, model-parallel over all
  GPUs: 23 min for 1319 GSM8K generations, i.e. half of each pipeline run.
  vLLM inside that environment is impossible without pins (the fork pins
  vllm==0.11.0 with transformers>=5, which vllm 0.11 cannot load).
* allenai/olmo-eval (main, 2026-09-03) installs cleanly with vllm 0.19.1 and
  transformers 5.4 via uvx, takes a local checkpoint directory as -m, and its
  gsm8k task is the gsm8k::olmes formulation: the same 8 fixed few-shot
  examples (identical text), the same Question/Answer prompt, greedy 512-token
  generation with the same stop sequences, exact match on the extracted number.
* Comparability check on checkpoints with known OLMES scores: unedited 0.6748
  vs 0.675; public-noise model 0.6672 vs 0.666. Within one question. Wall
  time 5 min per model (vs 24).
* Two upstream bugs worked around from the command line: the task points at
  the legacy dataset id `gsm8k` (the Hub client now requires `openai/gsm8k`;
  overridden with `-o data_source=hf://openai/gsm8k?subset=main&split=test`),
  and the model worker imports `beaker` unconditionally (beaker extra added).
  Both are worth an upstream PR.
* dvc.yaml `benchmark` stage now runs olmo-eval pinned to a current commit
  (`olmo_eval_commit`), keeps the metric key gsm8k::olmes and the output
  paths, so the notebook and the frontier are unaffected;
  `extract_olmes_metrics.py` reads both schemas (commit 6be7f36). Experiments
  from that commit on carry the olmo-eval GSM8K score; earlier ones the OLMES
  HF-backend score, on the same scale.

## 2026-09-05: batch 10, first result -- deleting the private block of layers 23-25

| run | edit | ppl | Dolma | quotes | GSM8K |
|---|---|---|---|---|---|
| sorer-rein | delete private block, layers 23-25 (6 matrices, alpha=1) | 18.005 | 0.322 | 0.346 | 0.640 |
| manly-sine | E-Identity 0.75 (curvature edit, all 6 matrices + down) | 18.10 | 0.302 | 0.355 | 0.643 |
| ahead-fees | EK-FAC 0.85 | 18.06 | 0.224 | 0.308 | 0.607 |
| telic-iota | private noise, norm 90 | 18.36 | 0.254 | 0.375 | 0.662 |

Predicted 18.0-18.1 / 0.26-0.30 / 0.38-0.42 / 0.63-0.66: perplexity and
GSM8K as predicted, slightly less Dolma forgetting than predicted and better
quote removal. The deletion sits on the E-Identity 0.75 point on all four
axes: a label-free, ranking-free, correction-free deterministic edit of six
matrices matches the best curvature edit at this forgetting level, and beats
the matched-forgetting noise edit by 0.36 perplexity at a cost of 0.02
GSM8K. The remaining three deletion runs (half, 23-28, 19-28) run on the new
olmo-eval benchmark stage.

## 2026-09-05: capability suite beyond arithmetic (Guillaume's request)

Purpose: see what the edits preserve or damage along axes other than
arithmetic, and test the theory's sharpest prediction: item-specific
knowledge should live in the private directions, so deleting the private block
should hurt closed-book factual recall while sparing in-context reading and
commonsense reasoning. The prediction is not obvious: factual recall is a
capability one might expect to be "general".

Suite (olmo-eval, vLLM, `run_suite.py`, ~9.4k instances, ~8 min per model at
two vLLM instances per GPU; fixed seed so every model sees the same subsets):

| axis | tasks (cap) |
|---|---|
| closed-book factual recall | naturalqs (500), jeopardy (500), popqa (500; long-tail entities) |
| knowledge + reasoning, ranked classification | arc_challenge, sciq (full) |
| commonsense / language, ranked classification | hellaswag (1000), winogrande, csqa (full) |
| in-context reading (generation) | squad (500), drop (500), coqa (200) |
| language modelling | lambada (1000) |

Dropped: mmlu (57 uncapped sub-tasks, 14k instances, too slow for a profile
run), piqa and socialiqa (Hub repos only have legacy loading scripts). The
first attempt (full sizes, one instance) ran at 3-8 items/s = 2-3 h per model;
the client batches requests in small sequential groups, so the GPU idles.
Capping and running two model instances per GPU (`-P 2`, memory 0.42 each)
gives ~19 items/s.

Models: unedited; private noise 60/90/120; public noise 30; deletion 23-25,
half, 23-28, 19-28; curvature edits at matched forgetting: EK-FAC 0.85
(ahead-fees), 0.8 (jowly-mesh), 0.7 (joint-kale); E-Identity 0.75
(manly-sine), 0.6 (unlet-genu); G-only K-FAC 0.6 (stiff-food); M-Identity
0.6 (gamey-quad).

Predictions: (1) every edit costs closed-book recall more than in-context
reading, and the private-block deletion costs recall at least as much as the
curvature edits at matched forgetting (it removes item-specific content
wholesale); (2) commonsense ranked-classification tasks move little for any
edit at Dolma >= 0.13; (3) the 19-28 deletion damages reading and recall
broadly, in line with its arithmetic collapse; (4) at matched forgetting,
private noise and deletion keep in-context reading (squad/drop) closer to the
unedited model than two-sided EK-FAC does, as they do GSM8K.

## 2026-09-05: timescale test (checkpoint drift by depth band) and the shape of the spectrum

Guillaume asked whether the private/public distinction is ontologically real
and which mathematical tools would characterise it better; the first test run
is the fast/slow-mode (timescale) picture: along a curvature eigen-direction
the residual decays as (1 - eta*lambda)^T, so sharp directions should be
settled early and flat directions should keep accumulating item-specific
content. `probe_drift.py` / `probe_drift2.py` (CPU, 1 min): gate/up of layers
23-25 at 1.26T, 2.10T, 2.94T, 3.78T tokens and final, in the final model's
K-FAC eigenbasis, per decile of general-text depth (band 0 = flattest).

Result: the depth dependence is real in rank but weak in magnitude.

| quantity (layer 24 up_proj) | flattest decile | sharpest decile |
|---|---|---|
| content norm |C_final| | 39.9 | 40.5 |
| persistence cos(C_1.26T, C_final) | 0.56 | 0.66 |
| relative drift 3.78T -> final (the anneal) | 0.275 | 0.244 |
| relative drift 1.26T -> 2.10T | 0.73 | 0.62 |
| autocorrelation of consecutive interval changes | -0.14 | -0.08 |
| anneal change / content, private-private vs public-public block | 1.06 | 0.90 |

* Per direction, the rank correlations are strong: relative drift in the
  anneal vs depth -0.50 (gate) to -0.89 (up); vs the memorized set's
  curvature share +0.50 to +0.85; persistence since 1.26T vs depth +0.50 to
  +0.91. Directions the memorized set uses more changed more during the
  anneal in which it was learned (recitation 0.64 -> 0.99), and hold less
  early content. But the effect sizes are ~10-20%: weight change is dominated
  by direction-independent components (learning-rate schedule: relative
  drift per interval falls 0.7 -> 0.55 -> 0.45 -> 0.26 in every band), and
  consecutive interval changes are *anticorrelated* in every band (mean
  reversion from weight decay / SGD noise), not accumulating. Weight energy
  per direction is nearly uniform across the spectrum (|C| ~ 40-44 per
  decile): the weights are almost white in the K-FAC basis.
* Why the timescale effect is small: the G-side K-FAC spectrum in these
  layers is nearly flat. log10 eva_G quantiles (1, 10, 50, 90, 99%):
  -5.36, -5.26, -5.12, -4.85, -4.32 -- a factor 3 across 90% of the
  directions and 10 across 98%. The A side is wider (-1.18 ... +0.30, x30)
  but still not the decades of a sloppy-model spectrum. The coupled depth
  ordering is rank-identical to eva (Spearman 0.998). With curvature varying
  by x2-3 across the bulk, (1 - eta*lambda)^T cannot separate timescales
  much, and it does not.
* Consequence for the ontology: on the output side the model has a small
  head (top few percent of directions, x3-10 the median curvature) and a
  near-degenerate bulk. Our 40/60 split falls inside the bulk; the
  public-block results are carried by the head, and inside the bulk every
  direction is alike -- which is exactly what the noise-shape invariance
  said. The memorized share nevertheless declines smoothly across the bulk
  (0.24 -> 0.13 over eva deciles), so the ordering inside the bulk is not
  pure noise (split-half test below). The honest description is "head vs
  bulk" plus a weak gradient inside the bulk, not two kinds of direction,
  and not a separation of timescales.
* Status of the fast/slow framing: a real but second-order effect here.
  What remains to explain where memorization sits is the input side (a
  wider spectrum: rare features) and the margin/usage picture, not weight
  dynamics. Next: band-resolved removal-vs-noise ratio (does the
  content/computation character jump at the head?) and self-influence.
* Split-half test of the ordering (2304 general sequences split at random,
  depth recomputed in each half): Spearman between halves 0.994-0.999 over
  all directions and 0.985-0.995 *within the flattest 60%* on both sides of
  every module. The ordering inside the bulk is statistically real; it is the
  dynamic range that is small (p90/p10 = 1.5-2.2 on the G side, 3.0-3.6 on
  the A side; p99/p50 = 1.7-3.4 G, 7-9 A). So: a reproducible, graded usage
  spectrum with a compressed range on the output side and a wider one on the
  input side.

## 2026-09-05: batch 10 (deletion models) and first capability-suite results

Benchmarks (olmo-eval stage for the last two):

| run | edit | ppl | Dolma | quotes | GSM8K | predicted |
|---|---|---|---|---|---|---|
| sorer-rein | delete 23-25 | 18.005 | 0.322 | 0.346 | 0.640 | 18.0-18.1 / .26-.30 / .38-.42 / .63-.66 |
| dicey-ribs | half 23-25 | 17.597 | 0.826 | 0.681 | 0.660 | 17.6-17.7 / .75-.80 / .68-.72 / .67 |
| agape-wool | delete 23-28 | 18.881 | 0.134 | 0.213 | 0.625 | 18.9-19.1 / .12-.14 / .24-.27 / .57-.61 |
| fussy-adze | delete 19-28 (negative control) | 20.136 | 0.068 | 0.128 | 0.522 | 20.0-20.5 / .05-.08 / .12-.16 / .35-.45 |
| refs at Dolma ~0.13 | EK-FAC 0.8 / E-Id 0.6 / G-only 0.6 / noise 120 | 18.40 / 18.63 / 18.75 / 19.07 | .142 / .144 / .128 / .126 | .242 / .235 / .252 / .250 | .578 / .621 / .607 / .651 | |

* Predictions held (GSM8K of the 6- and 10-layer deletions above the ranges:
  the benchmark is less sensitive than synthetic arithmetic to the loss of the
  private input directions of layers 19-22; GSM8K 0.522 is still the worst
  of any edit at this perplexity).
  The 6-layer deletion removes quotes best of all edits at Dolma ~0.13
  (0.213), keeps GSM8K above the two-sided and per-weight curvature edits
  (0.625 vs 0.578 / 0.621) but below noise (0.651), and costs 0.1-0.5 more
  perplexity than the curvature edits. Halving the block is nearly free and
  nearly useless, as the margin picture says.

Capability suite, first five models (delta vs unedited, percentage points):

| model (Dolma) | NQ | PopQA | Jeopardy | HellaSwag | ARC-C | SciQ | WinoG | CSQA | SQuAD | DROP | CoQA | LAMBADA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E-Identity 0.75 (.30) | -4.9 | -4.8 | -0.4 | -2.1 | +0.1 | -1.6 | -0.7 | -0.7 | -0.4 | -1.1 | -0.8 | +1.0 |
| private noise 90 (.25) | -5.2 | -6.8 | -2.7 | -1.2 | -0.3 | -0.6 | -0.8 | -0.6 | -0.3 | +1.2 | -0.5 | -1.1 |
| delete 23-25 (.32) | -7.9 | -8.0 | -2.7 | -1.3 | -0.3 | -1.2 | -0.6 | -0.7 | -0.0 | -0.1 | -0.6 | +1.6 |
| private noise 120 (.13) | -8.3 | -7.6 | -4.6 | -1.4 | -0.6 | -1.2 | -0.8 | -1.6 | -0.5 | +0.8 | -0.9 | -1.6 |

* Prediction 1 holds, sharply: closed-book factual recall is the capability
  every edit damages most (5-8 points on NaturalQs and PopQA against <= 1.6
  on everything else), and the private-block deletion costs recall *more*
  than the per-weight curvature edit at matched forgetting (-8 vs -5), about
  as much as norm-120 noise. Item-specific knowledge lives where memorized
  text lives; the private block is a store of specifics, not only of
  verbatim text. This is the theory's prediction and the deletion edit's
  price.
* Prediction 2 holds: commonsense / knowledge ranked-classification tasks
  move by at most 1-2 points for any edit.
* Prediction 4 (reading): SQuAD is untouched by every edit; DROP is kept or
  improved by noise and deletion and lost by the curvature edit (-1.1);
  CoQA -0.5 to -0.9 everywhere. Mild support.
* LAMBADA: deletion and the curvature edit gain +1.0-1.6; noise loses 1-1.6.
  Removing content helps next-word prediction on narrative text; adding
  random structure hurts it -- the removal/addition asymmetry again.

## 2026-09-05: band-resolved removal-vs-noise (round 1) and self-influence

**Band ratio, round 1** (`probe_band_ratio.py`, quintile blocks Q_k(G) x
Q_k(A), shrink alpha=0.5 vs matched-norm noise, norms 12.5-14.7 per matrix):

| block | shrink: typical / Pile / mem strict | noise: typical / Pile / mem strict | ratio (typical, Pile) |
|---|---|---|---|
| Q0-Q3 (flattest 80%) | +0.0003..0.0009 / ~0 / 0.947-0.953 | +0.0005..0.0013 / +0.0007..0.0010 / 0.942-0.960 | ~1 but at the noise floor |
| Q4 (sharpest 20% x 20%) | +0.0248 / +0.0394 / 0.822 | +0.0060 / +0.0071 / 0.909 | 4.1, 5.5 |

The computation signature (removal several times dearer than noise) appears
only in the head block; in the four bulk blocks a perturbation of norm 13 does
nothing measurable either way (the margin picture: small perturbations are
free). Round 1 is therefore uninformative about the bulk; round 2 (running)
removes each band block fully (alpha=1, norms ~26-29) against matched noise.

**Self-influence** (`probe_self_influence.py`, gate/up of layers 23-25,
K-FAC eigenbasis, damping 0.1 x mean eigenvalue, 600 items per population):
SI(x) = sum c_oi(x)^2 / (lambda_o mu_i + damp).

| population | SI median | grad norm^2 median | SI / grad^2 (mean inverse curvature along the gradient) | private-private share of SI |
|---|---|---|---|---|
| memorized | 8.4e7 | 180 | 5.0e5 | 0.497 |
| ordinary windows | 4.2e9 | 1.9e4 | 2.4e5 | 0.303 |
| arithmetic | 3.5e5 | 0.96 | 3.8e5 | 0.346 |

* Raw self-influence is a poor memorization detector for an already-trained
  model (AUC memorized vs windows 0.16, i.e. inverted): it is dominated by
  gradient magnitude, and memorized items are fit (loss 0.018), so their
  gradients are 100x smaller. Feldman-Zhang-style scores measure fit as much
  as privacy.
* Normalising by the gradient norm isolates the *direction*: the mean inverse
  curvature along a memorized item's gradient is 2.1x that of an ordinary
  window (AUC 0.978), and 50% of its H^-1-weighted gradient energy sits in
  the private-private block against 30% for ordinary text. This is the
  gradient-side twin of the forward private-share detector (AUC 0.99; rank
  correlation between the two 0.40 raw, 0.67 for the private-block share).
  Influence functions and the forward detector agree on what memorization
  is: an item whose gradient points into the flat bulk.
* Arithmetic sits between (1.6x the ordinary inverse curvature, 35% private
  share), as its flat-input / sharp-output profile predicts.

## 2026-09-05: band-resolved removal-vs-noise, round 2 -- the character jumps at the head

Full removal (alpha=1) of each quintile block Q_k(G) x Q_k(A) of the six
matrices, norm 25-29 per matrix, against isotropic noise of the same norm:

| block (depth G) | removal: typical / Pile / mem strict | noise: typical / Pile / mem strict | ratio removal/noise (typical, Pile) |
|---|---|---|---|
| Q0 (6.9e3) | +0.0013 / +0.0001 / 0.882 | +0.0018 / +0.0027 / 0.875 | 0.7, 0.04 |
| Q1 (8.1e3) | +0.0020 / +0.0016 / 0.886 | +0.0020 / +0.0039 / 0.890 | 1.0, 0.4 |
| Q2 (9.0e3) | +0.0022 / +0.0016 / 0.863 | +0.0023 / +0.0037 / 0.871 | 1.0, 0.4 |
| Q3 (1.0e4) | +0.0041 / +0.0037 / 0.862 | +0.0046 / +0.0044 / 0.863 | 0.9, 0.8 |
| Q4 (1.5e4, the head) | +0.1188 / +0.1409 / 0.277 | +0.0270 / +0.0307 / 0.706 | 4.4, 4.6 |

* The content/computation character is *uniform across the bulk and jumps at
  the head*: removal costs about what noise costs (0.7-1.0 on ordinary text,
  less on Pile) in every one of the four bulk quintiles, and 4.4-4.6x noise
  in the top quintile block. There is a qualitative boundary, and it is the
  head of the spectrum, not our 60/40 line.
* Inside the bulk the *cost* is graded even though the character is not:
  the same norm costs 3x more ordinary-text loss in Q3 than in Q0 (+0.0041
  vs +0.0013), in line with the second-order prediction cost ~ lambda_o mu_i
  ||dW||^2 (the product of the two sides' curvature ratios across these
  quintiles is ~3-4). So the bulk is a homogeneous *kind* with a smooth
  *price*: exactly the "head plus graded bulk" ontology.
* The head block is 30-90x more expensive to touch than any bulk block at
  equal norm, and removing it destroys recitation (0.277) along with
  everything else: computation, not content.
* Together with the spectrum (x3 across 90% of output directions), the
  split-half reproducibility (0.99), the shape invariance and the drift test:
  the model's late MLP weights have a small, sharply distinguished head of
  shared computation and a large bulk of content whose only internal
  structure is a smooth price gradient. "Public/private" should be read as
  "head/bulk"; the 40/60 split worked because the head sits inside the
  public 40%.

## 2026-09-05: capability suite, all 16 models (delta vs unedited, points; results scratchpad/suite/, figure capability_profile.png)

| model (Dolma) | NQ | PopQA | Jeop | ARC-C | SciQ | HS | WinoG | CSQA | SQuAD | DROP | CoQA | LAMB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Dolma ~0.13** | | | | | | | | | | | | |
| EK-FAC 0.8 jowly-mesh (.14) | -8.5 | -9.0 | -3.5 | -0.4 | -2.4 | -2.6 | -0.2 | -0.5 | -0.0 | -5.6 | -1.5 | +3.2 |
| E-Identity 0.6 unlet-genu (.14) | -10.5 | -9.0 | -3.2 | -0.2 | -2.8 | -2.5 | -1.3 | +0.1 | -0.3 | -1.8 | -0.1 | +1.2 |
| M-Identity 0.6 gamey-quad (.14) | -10.0 | -9.4 | -4.7 | -0.4 | -3.1 | -2.5 | -1.5 | +0.5 | -0.4 | -2.2 | -0.5 | +0.7 |
| G-only K-FAC 0.6 stiff-food (.13) | -8.0 | -7.8 | -1.8 | +0.3 | -2.8 | -2.1 | -0.8 | -0.3 | +0.2 | -2.2 | -1.2 | +1.3 |
| private noise 120 (.13) | -8.3 | -7.6 | -4.6 | -0.6 | -1.2 | -1.4 | -0.8 | -1.6 | -0.5 | +0.8 | -0.9 | -1.6 |
| delete 23-28 (.13) | -14.7 | -12.4 | -6.4 | -1.1 | -2.9 | -2.5 | -0.9 | -1.8 | -0.2 | -2.2 | -0.6 | +2.1 |
| **Dolma 0.22-0.32** | | | | | | | | | | | | |
| EK-FAC 0.85 ahead-fees (.22) | -7.2 | -7.0 | -2.2 | +0.3 | -1.6 | -2.1 | -0.2 | -0.3 | +0.1 | -1.5 | -0.2 | +2.1 |
| E-Identity 0.75 manly-sine (.30) | -4.9 | -4.8 | -0.4 | +0.1 | -1.6 | -2.1 | -0.7 | -0.7 | -0.4 | -1.1 | -0.8 | +1.0 |
| private noise 90 (.25) | -5.2 | -6.8 | -2.7 | -0.3 | -0.6 | -1.2 | -0.8 | -0.6 | -0.3 | +1.2 | -0.5 | -1.1 |
| delete 23-25 (.32) | -7.9 | -8.0 | -2.7 | -0.3 | -1.2 | -1.3 | -0.6 | -0.7 | -0.0 | -0.1 | -0.6 | +1.6 |
| **mild** | | | | | | | | | | | | |
| private noise 60 (.60) | -2.3 | -3.2 | -1.4 | +0.3 | -0.2 | -0.5 | -0.2 | -0.7 | -0.1 | -0.4 | -0.5 | -0.2 |
| half block 23-25 (.83) | -2.9 | -2.4 | +0.0 | +0.0 | -0.3 | -0.6 | -0.4 | -0.2 | -0.0 | -0.6 | +0.0 | +0.4 |
| public noise 30 (.83) | -2.3 | -0.8 | +0.4 | +0.5 | -0.2 | -0.8 | -1.1 | -0.1 | +0.6 | +0.2 | -1.8 | -1.1 |
| **deep** | | | | | | | | | | | | |
| EK-FAC 0.7 joint-kale (.09) | -13.8 | -10.4 | -5.4 | -1.5 | -3.0 | -3.9 | -0.6 | -0.7 | +0.1 | -9.6 | -1.3 | +2.6 |
| delete 19-28 (.07) | -15.1 | -14.0 | -10.9 | -2.4 | -3.2 | -5.8 | -1.1 | -4.4 | -0.8 | -5.8 | -3.6 | +3.0 |

Findings:
1. **Closed-book factual recall is the cost of forgetting, for every edit.**
   NaturalQs and PopQA fall 5-15 points while commonsense, ARC, SciQ, SQuAD
   and CoQA move by 0-3. The loss tracks the amount of memorization removed
   (Dolma): a recall/forgetting trade-off exists whatever the method. Facts
   and verbatim text share the store.
2. **The curvature ranking protects facts, not general text.** At matched
   forgetting the private-block deletion costs 3-6 points more recall than
   the curvature edits and noise (Dolma ~.13: -14.7/-12.4 vs -8..-10.5;
   Dolma ~.3: -7.9/-8.0 vs -4.9/-4.8 for E-Identity 0.75 and -5.2/-6.8 for
   noise 90). The noise-shape experiment showed that ordering inside the
   block does not matter for ordinary or Pile text; it matters for factual
   knowledge: pruning removes the small components first and facts sit in
   larger ones than verbatim text (they are reinforced by many paraphrases;
   verbatim items are not). This is the real cost of the deterministic edit,
   invisible to perplexity and GSM8K.
3. **Two-sided EK-FAC damages reading-based numerical reasoning.** DROP:
   -5.6 (0.8), -9.6 (0.7), -1.5 (0.85) for EK-FAC against -2.2 or better for
   every other edit at the same forgetting and +0.8/+1.2 for noise. The
   GSM8K result generalises: the input-side pruning of the K-FAC rule hits
   number processing wherever it is needed; per-weight, output-only and
   block-confined edits do not.
4. Cheapest recall per unit forgetting at Dolma ~.13: noise 120 ~ G-only
   K-FAC ~ EK-FAC 0.8 (-8) < per-weight edits (-10) < deletion (-14.7); at
   Dolma ~.3: E-Identity 0.75 (-4.9) < noise 90 (-5.2) < deletion (-7.9).
   Noise is the only family that leaves DROP and HellaSwag essentially
   untouched while paying the same recall as the best curvature edits.
5. LAMBADA improves under every removal-type edit (+0.7 to +3.2) and worsens
   under noise (-1.1 to -1.6): the removal/addition asymmetry once more, on
   narrative next-word prediction.
6. The two deep edits (EK-FAC 0.7, delete 19-28) damage everything; the
   public-noise control damages CoQA and WinoGrande while barely forgetting.

## 2026-09-05: theory sprint -- the spectrum across layers, and Hill numbers

Guillaume: forget paper-facing results, get to a sound theory fast. Running
in parallel: (a) magnitude test (delete the smallest / largest / random
components of the private block at matched norm 54 -> recall vs recitation),
(b) reference relativity (Pile-defined bases and private block), (c) spectrum
shape across all layers, (d) Hill-number breadth profiles. (c) and (d) are in.

**Spectrum across layers** (`analyze_spectrum_layers.py`, G-side coupled
eigenvalues for all 32 layers, `spectrum_layers.png`): the bulk is flat
everywhere in the middle and late layers -- p90/p10 = 3.5-5 for layers 4-28,
p99/p50 = 4-7, and the head above 10x the median holds 0.1-0.5% of the
directions. The extremes differ: layers 0-3 and 29-31 have larger heads and
wider spectra (layer 31 up_proj: p99/p50 = 79, max/median 1100, 9% of
directions above 10x median). The flat-bulk / tiny-head geometry is a property
of the middle and late MLPs, i.e. exactly the layers where the private-block
story works (>= 19), and it is *not* the geometry of the first and last few
layers, which hold high-gain shared machinery (embedding/unembedding-adjacent).

**Hill numbers** (`analyze_hill.py`, `hill_numbers.png`; per direction the
effective number of sequences N_alpha sharing it, out of 2304): the bulk is
*not rarely used*. Median N_1 is ~1500 of 2304 sequences (65%) on the G side
and ~1300 on the A side; the head's top decile has N_1 ~ 1750-1940, the
flattest 60% ~1450-1540. N_1 correlates with depth (Spearman 0.90-0.98) and
anticorrelates with the memorized set's curvature share (-0.77 to -0.94), so
the ordering is right, but the *contrast* is small: head and bulk differ in
curvature per direction by x2-3 and in breadth by x1.2-1.4. Per item, a
memorized item's curvature is spread over N_1 ~ 8300-8900 of 11008 output
directions (general windows 8900-10200) and ~2200-2400 of 4096 input
directions (general windows 1500-1900): on the input side memorized items are
*more* distributed than ordinary text, not concentrated on a few private
features.

**Consequence for the theory statement.** "Private directions carry curvature
from few tokens" is false in the literal sense; almost every direction is used
by most sequences. What distinguishes the head is *gain*: a small set of
directions along which the population's curvature is concentrated (large loss
sensitivity for everyone). The bulk is the large remainder: shared, low-gain
capacity in which everything, including item traces, leaves small, evenly
spread marks. Memorization "in flat directions" is then two things at once: a
volume effect (the head is a few percent of dimensions; idiosyncratic content
lands in the rest by counting), and an avoidance effect (memorized items'
activation energy engages the high-gain shared features less -- the private
share is essentially 1 minus the head share). This is consistent with the
weak timescale effect (curvature x3 cannot separate dynamics), the shape
invariance (the bulk is homogeneous), and the removal-vs-noise jump (the head
is computation because it is where gain is concentrated).

## 2026-09-05: magnitude test refutes "facts sit in larger components"; reference relativity confirmed

**Magnitude test** (`make_block_models.py --mode small|large|random --norm 54`,
half the private block's energy removed from the six matrices of layers
23-25, by smallest / largest / random |C_oi|; recall via olmo-eval):

| removed (norm 54) | components | mem loss / strict | typical | Pile | arith | NQ | PopQA | Jeop | HS | DROP |
|---|---|---|---|---|---|---|---|---|---|---|
| smallest | 87-88% | 0.095 / 0.505 | +0.0079 | +0.0079 | 0.877 | -3.7 | -3.0 | -0.3 | -0.9 | +0.0 |
| largest | 11-13% | 0.095 / 0.507 | +0.0090 | +0.0091 | 0.878 | -3.3 | -2.4 | -1.1 | -0.4 | -1.1 |
| random | ~50% | 0.098 / 0.472 | +0.0085 | +0.0093 | 0.877 | -3.1 | -3.8 | -0.7 | -1.1 | -0.5 |

Identical on every metric, recall included. Removing 87% of the block's
components (the small ones) or 12% (the large ones) at equal energy gives the
same forgetting, the same collateral and the same factual-recall loss. So the
conjecture written this morning -- that the curvature edits' lower recall cost
comes from a magnitude ranking that spares fact-carrying large components --
is **wrong**, and the sentence "facts sit in larger components than verbatim
items" must go from the write-ups. Inside the bulk, magnitude is irrelevant
for everything we can measure, facts included: the bulk is homogeneous also in
what it stores per unit energy.

What then explains deletion's higher recall cost than noise at matched
forgetting (-7.9 vs -5.2 NaturalQs)? The remaining candidate is the coherent /
incoherent distinction itself: verbatim recitation has small margins and falls
to both removal and random perturbation, factual recall is redundant and
survives random perturbation better than coherent removal of the components
it lives in. The curvature edits' footprint (all three MLP matrices, saliency
selection, identity or K-FAC basis) differs from the block deletion's, so
their recall cost cannot be attributed to a ranking; the earlier claim is
withdrawn, not replaced.

**Reference relativity** (`--ref pile`, bases and coupled covariances saved;
`analyze_reference.py`):
* The two references agree on *which directions are high-gain*: the Pile
  curvature of the dolmino eigen-directions has Spearman 0.89-0.98 with their
  dolmino curvature, on both sides of every module.
* But the eigen-*subspaces* barely overlap beyond chance (top-10%: 0.13-0.25
  vs 0.10; top-40%: 0.42-0.52 vs 0.40; bottom-60%: 0.61-0.68 vs 0.60). With a
  near-degenerate spectrum, eigenvectors are rotated arbitrarily within the
  bulk and even within the head; subspace overlap is the wrong test, the
  agreement of quadratic forms is the right one. (This also says the
  individual "directions" we edit are not canonical objects; only the ordering
  by gain is.)
* Functionally the relativity is exact and mirror-symmetric: deleting the
  *Pile*-defined private block costs dolmino text +0.063 nats (3x the +0.021
  of the dolmino-defined deletion) and Pile text +0.015 (half of +0.027),
  forgets the Dolma set more (strict 0.124 vs 0.204; it is dolmino-like text),
  and costs recall the same (NQ -6.9, PopQA -6.6 vs -7.9/-8.0) with more
  DROP/HellaSwag damage (-2.0/-2.1 vs -0.1/-1.3).
* Reading: "private w.r.t. D" is the subspace where D's curvature is low, and
  whatever *other* distributions need sits partly there. The collateral of a
  private-block edit on out-of-reference text is intrinsic to the reference
  choice; the right reference for a memorization edit is the broadest one
  available (the pretraining mixture), so that "private" means idiosyncratic
  to no population at all.

## 2026-09-05: sharp-but-narrow directions are negligible; the theory statement, revised

`probe_outliers.py`: directions in the top 5% of depth with breadth below the
bulk median number 3-9 per module on the G side and 0-1 on the A side, hold
0.1-0.7% of the depth, and their memorized share (0.06-0.21) is *not* elevated
over the bulk's (0.22-0.32). Projecting them out costs nothing (recitation
0.99 -> 0.976, ordinary text +0.001). The matched control -- the same number
of the broadest head directions -- costs +0.018 (G) and +0.124 nats (A) and
destroys recitation along with everything else (strict 0.454). No
"memorization neurons" hide in the head; the scatter's coloured outliers were
one or two directions. Closed.

**Theory statement, revised (written into the notes and key findings):**
Fix a weight matrix of a middle-or-late MLP and the Fisher of the loss on a
reference population D. Its eigen-directions differ in *gain* (curvature per
direction) far more than in *breadth* (how many contexts use them). A head of
a few percent of directions concentrates the population's loss sensitivity;
the remaining bulk is near-degenerate (x3 across 90% of directions) and
homogeneous: inside it, which components an edit touches does not matter,
only how much energy is moved and whether the change is coherent (removal) or
incoherent (noise). Memorized items are items whose activation energy and
gradient avoid the head; their signal is spread evenly over the bulk, on both
sides. Item-specific content -- verbatim text and facts alike -- lives in the
bulk by volume (the head is a few percent of dimensions) and by avoidance,
not because the bulk's directions are rarely used (they are used by ~65% of
sequences) or slowly learned (timescale differences are 10-20%). The head is
computation (removal costs 4-5x equal-norm noise), the bulk is content
(removal costs about what noise costs). Which subspace is the bulk depends on
the reference, exactly and symmetrically. Verbatim text differs from facts in
robustness, not location: small margins fall to any perturbation, redundant
facts fall only to coherent removal. The right edit is therefore a coherent
removal confined to the bulk of the broadest available reference, on the
output side where skills do not read.

Open after today: (i) why the head is so small and the spectrum so flat in
layers 4-28 (a training-dynamics or architecture question); (ii) whether the
same geometry holds in other model families; (iii) a quantitative margin model
predicting the recall/forgetting curve of removal vs noise from item
redundancy; (iv) an edit that removes coherently but spares redundant
content -- e.g. removal restricted to bulk components whose sign structure is
item-like rather than fact-like, if such a distinction exists at all.

## 2026-09-05: mathematical programme -- three checks of the proposed theory

Proposed formalism (see the message of this afternoon): item and population
Fisher mass as measures on the reference spectrum; collateral second-order
(cost ~ sum Lambda Delta^2), forgetting first-order (logit shift vs margin,
coherent N*eps vs incoherent sqrt(N)*eps), and an accumulation model (item
residue signal / population random walk) predicting memorized share ~ kappa^-1/2.

**Check 1 -- the exponent** (`analyze_powerlaw.py`, log share vs log depth over
directions; share = population depth / general depth):

| module side | memorized, all dirs (R^2) | memorized, bulk only | head | clean | typical | null |
|---|---|---|---|---|---|---|
| 23 up G | -0.50 (0.82) | -0.50 | -0.06 | +0.08 | +0.07 | +0.01 |
| 24 up G | -0.51 (0.78) | -0.43 | -0.05 | +0.09 | +0.09 | +0.01 |
| 25 up G | -0.53 (0.83) | -0.41 | -0.16 | +0.09 | +0.08 | +0.01 |
| 23/24/25 gate G | -0.34 / -0.37 / -0.31 (0.4-0.5) | -0.15 / -0.15 / -0.08 | ~+0.1 | +0.10..0.21 | +0.17..0.22 | +0.01 |
| 23/24/25 up A | -0.47 / -0.48 / -0.49 (0.80) | -0.66 / -0.71 / -0.73 | -0.1 | +0.02..0.03 | +0.03 | +0.01 |
| 23/24/25 gate A | -0.46 / -0.47 / -0.48 (0.78) | -0.65 / -0.70 / -0.73 | -0.1 | +0.03..0.04 | +0.03..0.04 | +0.01 |

* The up-projection output side and both input sides give exponent -0.46 to
  -0.53 over the whole range with R^2 ~ 0.8: the kappa^-1/2 prediction holds
  where the spectrum is best behaved. The gate output side is shallower
  (-0.31 to -0.37, R^2 0.4-0.5) and nearly flat inside its bulk; the input
  sides are steeper inside the bulk (-0.65 to -0.73) and flatten in the head.
  Clean and typical text have small *positive* exponents (their share grows
  toward the head), and the split-half null is +0.01: the effect is not an
  estimation artifact.
* Reading: a -1/2 law is the first-order description, with structured
  deviations -- the gate's output gradients pass through the SiLU gate, which
  may decorrelate the item signal from the population walk; the input side's
  steeper bulk suggests item residue there is closer to constant across
  directions while population variance grows. Both are targets for the
  derivation, not exceptions to wave away.

Checks 2 (second-order collateral from the factors) and 3 (margin model) are
running.

**Check 2 -- second-order collateral** (`predict_second_order.py`): with the
EK-FAC diagonal curvature Lambda_oi (dolmino calibration, K-FAC basis) the
prediction cost = c * sum Lambda Delta^2 reproduces the ten band-ratio
measurements (five quintile blocks, removal and noise) with log-space R^2 =
0.97 and a single fitted constant c = 0.57 -- against the theoretical 1/2. So
the absolute second-order formula holds with no free parameter in the bulk:

| block | removal meas / pred | noise meas / pred | ratio meas / pred |
|---|---|---|---|
| Q0 | +0.0013 / +0.0016 | +0.0018 / +0.0016 | 0.73 / 0.99 |
| Q1 | +0.0020 / +0.0021 | +0.0020 / +0.0021 | 0.99 / 1.00 |
| Q2 | +0.0022 / +0.0029 | +0.0023 / +0.0029 | 0.95 / 1.00 |
| Q3 | +0.0041 / +0.0043 | +0.0046 / +0.0043 | 0.91 / 1.00 |
| Q4 (head) | +0.1188 / +0.0680 | +0.0270 / +0.0245 | 4.40 / 2.78 |

* Bulk: second order is exact to the precision of the measurement, for both
  removal and noise, and predicts the ratio 1 (content) directly: with
  Lambda nearly constant across the block, the cost depends on the energy
  moved, not on its shape -- the shape invariance derived.
* Head: noise is predicted (0.0245 vs 0.027) but removal is under-predicted
  by 1.75x (0.068 vs 0.119). Removing the head's learned content is a large
  functional change with a super-quadratic loss response; the second-order
  ratio 2.8 already says "computation" (content aligned with high curvature),
  the remainder is higher order. So the content/computation asymmetry has a
  second-order part (alignment of content with curvature) and a higher-order
  part specific to removing what the model computes with.
* Reference swap with the *separable* (marginal outer-product) curvature of
  the other population on the coupled bases: predictions off by 2-3x
  (dolmino->dolmino 0.007 vs 0.021; dolmino bulk->Pile 0.089 vs 0.027;
  Pile bulk->dolmino 0.030 vs 0.063; Pile->Pile 0.020 vs 0.015). The
  direction (cross costs exceed own costs) is right, the magnitudes are not:
  the separable approximation is the K-FAC independence assumption, which we
  already know misestimates curvature. Computing the true diagonal Lambda of
  each population in each basis (`compute_lambda_pop.py`, running) to test the
  cross-population prediction properly.

**Check 3 -- the margin model** (`probe_margins.py`; per-token margins = target
logit minus best other, for 1054 memorized items x 48 suffix tokens and for the
answer tokens of 316 NaturalQs/PopQA items the model answers correctly, under
shrink alpha in {0.1..1} and isotropic noise at norms {19, 38, 57, 76}, two
seeds; the block is the K-FAC private block of layers 23-25 gate/up):

| quantity | memorized suffix tokens | fact answer tokens |
|---|---|---|
| baseline margin: per-token median / item-min median | 9.94 / 2.26 | 3.75 / 1.49 |
| shift at norm 76, removal: mean (std) | -2.92 (3.70) | -1.38 (2.18) |
| shift at norm 76, noise: mean (std) | -2.19 (3.17) | -0.82 (1.65) |
| |mean removal| / std noise | 0.92 | 0.83 |
| outcome at norm 76: removal / noise | strict 0.179 / 0.287 | acc 0.627 / 0.783 |
| linearity of shrink response (slope, corr): alpha .25 / .5 / .75 / 1 | 0.96, 0.92 / 0.90, 0.75 / 0.79, 0.57 / 0.67, 0.42 | |

Three corrections to the story told this afternoon:
1. **The coherent-vs-incoherent (N vs sqrt N) picture is wrong.** The block's
   coherent contribution to a memorized token's logit (removal shift -2.9) is
   about equal to the noise-induced spread (std 3.2), not much larger, and
   noise has a large *systematic* negative mean shift (-2.2), because a
   confident correct token sits at a maximum: perturbing the logits raises the
   best competitor on average. Removal beats noise per unit norm by only ~1.3x
   in mean margin loss, which is enough for 0.18 vs 0.29 strict recitation.
2. **Facts are not more robust by margin; they are less located here.** Fact
   answer tokens have *smaller* margins than verbatim tokens (median 3.75 vs
   9.94; minimum over the answer 1.49 vs 2.26 over the 48-token chain). They
   survive because the block carries half as much of their logit: removal
   shifts them by -1.38 against -2.92. "Robustness, not location" must be
   withdrawn; it is location -- how much of the token's margin the private
   block of these three layers carries -- and facts are spread over more of
   the network.
3. **Why noise is the more selective edit (less recall per unit forgetting).**
   Noise damage scales with a token's *energy* in the block (variance of the
   induced shift: memorized/facts = (3.17/1.65)^2 = 3.7x); removal damage
   scales with the *coherent* block contribution (2.92/1.38 = 2.1x). Memorized
   tokens route relatively more energy through the block than they extract as
   aligned contribution -- their code has mixed signs, a distributed code with
   partial cancellation -- so an edit that acts through energy (noise) is more
   selective for them than one that acts through the coherent sum (removal).
   This is the mechanism behind the capability-suite asymmetry, and it is a
   second-order-vs-first-order distinction after all, just not the one I wrote.
4. The first-order (linear-in-alpha) model of the shrink response holds to
   alpha ~ 0.5 (slope 0.90, corr 0.75) and degrades beyond: for large edits the
   margin response saturates and re-orders (slope 0.67 at alpha = 1). Forgetting
   curves are therefore predictable from a small-alpha probe up to about half
   the block, and the full-deletion regime needs the nonlinearity.

**Check 2b -- cross-population costs with the true diagonal curvature**
(`compute_lambda_pop.py`: Lambda^P_Q = E_t[g~^2 a~^2] of population P in the
coupled basis of reference Q, sampled labels, 1152 sequences each):

| edit -> population | predicted 1/2 sum Lambda C^2 | measured |
|---|---|---|
| dolmino bulk -> dolmino | +0.0082 | +0.021 |
| dolmino bulk -> Pile | +0.090 | +0.027 |
| Pile bulk -> dolmino | +0.030 | +0.063 |
| Pile bulk -> Pile | +0.0225 | +0.015 |

Still off by 1.5-3.3x, and the asymmetry is wrong in both directions (predicted
11x cross/own for the dolmino deletion, measured 1.3x; predicted 1.3x for the
Pile deletion, measured 4.2x). The own-population noise curve is also
under-predicted by a constant 2.3x here, whereas the same formula in the K-FAC
basis with the collection's EK-FAC Lambda fitted the band probe with constant
0.57 -- so absolute constants differ by ~2x between the two curvature
estimates (normalisation / token positions), but the cross-population *pattern*
is the real failure. Suspect: the diagonal approximation
E[g~_o g~_o' a~_i a~_i'] ~ delta delta E[g~_o^2 a~_i^2] is justified only in the
population's own eigenbasis (where second moments are diagonal); for another
population expressed in this basis the off-diagonal fourth moments are O(1),
and for a structured perturbation (removal of C itself) they do not average
out. Testing the second-order theory itself, without factorisation, by measuring
the exact quadratic form 1/2 E[(sum_m g_m^T dW_m a_m)^2] per token, and the
first-order term E[g_true^T dW a] which need not vanish for a population the
model is not at a minimum for (`probe_quadform.py`, running).

**Check 2c -- exact Taylor terms, no factorisation** (`probe_quadform.py`:
per token s_t = sum_m g_m^T dW_m a_m over the six edited matrices; first order
E[s] with true-label gradients, second order 1/2 E[s^2] with sampled labels =
the GGN quadratic form; same tokens and masks as the measured losses):

| edit -> population | first order | second order (GGN) | total | measured |
|---|---|---|---|---|
| dolmino bulk -> dolmino | -0.0025 | +0.0209 | +0.0184 | +0.021 |
| Pile bulk -> dolmino | -0.0040 | +0.0607 | +0.0567 | +0.063 |
| dolmino bulk -> Pile | -0.0228 | +0.0784 | +0.0556 | +0.027 |
| Pile bulk -> Pile | -0.0147 | +0.0410 | +0.0263 | +0.015 |

* **In distribution the theory is exact.** Both deletions' costs on dolmino
  text are predicted within 10-12% by the unfactorised second-order form. The
  diagonal (K-FAC / EK-FAC) curvature under-predicted these by 2-2.5x: for a
  structured perturbation (removal of the content itself) the off-diagonal
  fourth moments add constructively, even in the population's own coupled
  basis. The band-probe fit in the K-FAC basis (constant 0.57) happened to
  land near 1/2 with the collection's Lambda; the safe statement is that the
  *exact* GGN form predicts collateral and the factorised one needs a
  calibration factor of order 2 for coherent removals.
* **Out of distribution the second-order form is an upper bound.** On Pile
  both predictions overshoot by 1.75-2x. The rms logit shift is twice as
  large on Pile (0.40 vs 0.20 logit units), i.e. Pile leans harder on this
  content, and cross-entropy saturates: it grows linearly, not quadratically,
  in large logit shifts. So OOD collateral is sub-quadratic in the edit.
* **The first-order term is negative.** Removing the reference's bulk content
  *lowers* the loss to first order for both populations, negligibly for
  dolmino (-0.0025) but materially for Pile (-0.023, -0.015: about half of its
  measured cost in magnitude). The population gradient points toward less
  bulk content: idiosyncratic content stored in the bulk actively interferes
  with other distributions, and the quadratic cost of removing it is partly
  offset by that relief. This is the interference reading of "content" made
  quantitative.

Status of the mathematical programme after the three checks: collateral is a
solved second-order problem in distribution (exact GGN form, no free
parameter), sub-quadratic with a first-order interference relief out of
distribution; forgetting is a margin crossing with a measured mechanism (energy
for noise, coherent contribution for removal) and a linear regime to half the
block; the kappa^-1/2 law holds on the up-projection output side and both input
sides. The remaining theoretical items are the derivation of the exponent (and
of its deviations on the gate output side and in the input bulk), a saturation
model for OOD collateral, and a model of the noise mean-shift (the
"maximum-regression" of a confident correct token).

## 2026-09-05: building the theory -- `tex/theory.pdf`, and the half-whitening measurement

Guillaume: time to build a genuine theory. Written as `theory.tex` (scratchpad
tex/, ~8 pp): Proposition 1 (two-order collateral) with corollaries -- shape
invariance / graded price, removal-vs-noise ratio as content-weighted mean
curvature, failure of the factorised curvature for coherent removals, OOD
saturation bound (cross-entropy is 2-Lipschitz in the logit sup-norm), the
negative first-order interference term, reference relativity; Proposition 2
(removal and noise at the margin) with the selectivity corollary; the
kappa^-1/2 claim and its mechanism; head and bulk; five open derivations.
Every statement carries the measurement that pins it.

**Correction to the margin entry above (point 3):** the alignments of the
memorized code and of the fact code are nearly equal (coherent ratio 2.1 vs
noise-spread ratio 1.9), so "distributed code with partial cancellation" is
not supported. The right statement is the linear-vs-quadratic law: removal
damages a population linearly in its coupling to the block, noise between
linearly (spread) and quadratically (mean shift, ~ -0.25 x variance per
logit). A population coupling k times more strongly suffers k x from removal
and k..k^2 x from noise. That is why noise is the more selective edit.

**Half-whitening** (`probe_halfwhitening.py`): which side carries the
kappa^-1/2 law? Per direction, E_mem[a~_i^2] and E_mem[g~_o^2] against the
ordinary-text energies:

| module | side | slope of log(E_mem/E_win) vs log E_win (R^2) | clean control | mean E_mem/E_win |
|---|---|---|---|---|
| 23/24/25 gate+up | A (input) | -0.65 / -0.67 / -0.69 (0.94) | -0.09 | 0.87-0.95 |
| 23/24/25 up | G (output) | -0.62 / -0.66 / -0.69 (0.85) | -0.03..-0.06 | 0.27-0.28 |
| 23/24/25 gate | G (output) | -0.59 / -0.63 / -0.58 (0.67) | -0.15..+0.02 | 0.32-0.34 |

* Both sides, same power: E_mem ~ E_win^(1/3) along each side (ratio falls as
  E_win^(-2/3)), R^2 0.94 on the input side. Clean text is flat. Memorized
  text has normal total activation energy but spreads it far more evenly over
  the input spectrum; its gradients are 3x weaker in total (fit) and likewise
  spread.
* So the location law is *representational*: it is about which texts are
  memorized and how they are encoded (they engage the reference's principal
  directions less, in and out), not primarily about SGD accumulation. The
  depth share, marginalising one side against the other, inherits an exponent
  near -1/2. The storage model's remaining job: explain why a population with
  a one-third-power spectral profile is the one that ends up recited, and how
  much training sharpens the profile (private share rises 0.13 -> 0.25 as an
  item is memorized, so partly it does).
* Selectivity in one formula: an item's coupling to the bulk relative to the
  head is set by its spectral profile; the flatter the profile, the more of its
  energy and coherent contribution live in the bulk, the more any bulk edit
  costs it. Verbatim text is the flattest population we have; facts are
  intermediate (half the block coupling); ordinary text follows the reference.

## 2026-09-05: predictive tests of the margin model (parsimony and prediction)

Guillaume: a good theory must be parsimonious and predictive. Test: fit the
margin model per token at noise norms <= 76 (mean shift -beta_t n^2, spread
gamma_t n, Gaussian), predict strict recitation at norms 90/107/120 measured
independently (probe_noise_shape), and extrapolate the shrink curve from
alpha <= 0.5 (`analyze_margin_predict*.py`):

| noise norm | measured strict | independent Gaussian | + mean saturating ~n^1.5 beyond 76 |
|---|---|---|---|
| 38 / 57 / 76 (in-sample) | 0.789 / 0.532 / 0.282 | 0.777 / 0.486 / 0.229 | same |
| 90 / 107 / 120 (out of sample) | 0.159 / 0.098 / 0.083 | 0.121 / 0.062 / 0.040 | 0.141 / 0.084 / 0.060 |

| shrink alpha | measured | linear from alpha<=0.5 | quadratic from alpha<=0.5 |
|---|---|---|---|
| 0.75 / 1.0 | 0.430 / 0.179 | 0.584 / 0.430 | 0.351 / 0.112 |

* The two-coefficient-per-token model predicts the noise curve within 5 points
  absolute in and out of sample, always over-predicting forgetting; the
  residual is consistent with the mean shift saturating beyond norm 76 (a
  quadratic law extrapolates to -5.5 logits at 120). The mean law is
  quadratic to 76 (predicted -0.55/-1.23 vs measured -0.56/-1.27 at 38/57).
* Within-item correlation of the random shift is nil (rho = 0.01): one weight
  perturbation moves an item's 48 tokens independently -- their block
  couplings (u_t, v_t) are nearly orthogonal. A structural fact about the code:
  per-token, not per-item.
* Deletion beyond half the block is genuinely nonlinear: linear extrapolation
  under-predicts forgetting (0.43 vs 0.18 at alpha=1), quadratic over-predicts
  (0.11). The first-order theory owns the regime alpha <= 0.5; the full-deletion
  regime needs the nonlinearity (attention re-routing / saturation), which is
  an open item, not a failure of the two-order picture where it claims to apply.
* Per item, the edit-free private share (layer 24, input side) predicts the
  alpha at which an item stops being recited with Spearman -0.44, its noise
  shift with +0.37 and its removal shift with +0.29: a one-forward-pass
  quantity carries almost half the rank information about fragility under a
  three-layer, two-sided edit.

## 2026-09-05: the facts population's profile, and a confound in the gradient side

`probe_profiles_facts.py` (answer tokens of 316 correctly answered NQ/PopQA
items vs memorized suffix tokens vs ordinary windows, final K-FAC bases):

| side | exponent memorized | exponent facts | block energy (product G x A) mem/win, facts/win |
|---|---|---|---|
| A (activations) | -0.65 .. -0.69 (R2 0.94) | -0.74 .. -0.81 (R2 0.90) | |
| G (loss gradients) | -0.58 .. -0.69 | -0.69 .. -1.16 | 0.73-0.84, 4.3-5.7 (mem/facts 0.13-0.20) |

* Facts' *activations* are flatter than memorized tokens' (the answer position
  after "A:" in a few-shot QA prompt is an atypical context) -- so activation
  flatness alone does not make a population fragile to bulk edits.
* The product-of-loss-gradient-energies prediction of the block coupling is
  inverted (0.16 predicted vs 3.7 measured). Cause: the loss gradient scales
  with (1 - p_y), the token's misfit. Memorized tokens are fit (p_y ~ 0.98),
  fact tokens are not (margins 1.5-3.8), so the facts' loss-gradient energy is
  5-8x larger in total for reasons unrelated to how their margins couple to
  the block. The margin's gradient is the right object for forgetting. Also
  a caution for the half-whitening exponent on the G side: a population's
  gradient-energy profile is weighted toward its least-fit tokens.
* Rerunning with (i) unit-error gradients g/(1-p_y) and (ii) the margin
  gradient itself, and resolving the bulk energy by side, to predict the
  coupling ratio and to see whether facts and verbatim differ on the output
  side (facts written through the head, like arithmetic?) rather than the
  input side.

## 2026-09-05: does flatness precede or follow memorization? (profile dynamics)

`probe_profile_dynamics.py`: spectral exponents (E_pop/E_never vs E_never,
final K-FAC bases, layers 23-25) at 1.26T tokens and at the final model for
(i) ordinary windows never recited, (ii) 112 windows not recited at 1.26T
(acc < 0.75) but recited by the final model ("become"), (iii) the memorized
Dolma set (95% recited per token already at 1.26T):

| population | side | exponent at 1.26T | exponent at final |
|---|---|---|---|
| become | A (activations) | +0.08 | -0.17 |
| become | G (loss gradients) | +0.24 (R2 0.07) | -0.52 (R2 0.09) |
| memorized set | A | -0.61 (R2 0.92) | -0.64 (R2 0.93) |
| memorized set | G | -0.73 (R2 0.75) | -0.56 (R2 0.67) |

* Windows that get memorized between 1.26T and the end start with a *typical*
  profile (exponents ~0, slightly head-leaning) and end flatter on both sides.
  Learning flattens the representation of what it memorizes -- but these
  windows, memorized late and seen few times, end far less flat (A: -0.17)
  than the Dolma set (A: -0.64), which was already at its final flatness at
  1.26T and did not change.
* So: flattening is produced by memorization (not a pre-existing property of
  the text), and its degree tracks the depth/duration of memorization (the
  Dolma set is heavily duplicated and long memorized). Whether some flatness
  precedes memorization for the Dolma set cannot be decided from these
  checkpoints (it was recited before the first one); the become-windows say
  the effect is at least partly learned.
* Theory reading: fitting an item to near-zero loss against the population's
  restoring force places its representation and its writes where the
  population's curvature is small; the more an item is fit, the flatter its
  profile. The private-share detector measures exactly this flatness, which
  is why it tracks memorization state and rises as items are memorized.

**Dose-response is a threshold.** Over the 9216 ordinary windows, the layer-24
private share is flat at 0.18-0.19 for final recitation accuracy anywhere in
[0, 0.9) and jumps to 0.257 in [0.9, 1) and 0.273 at 1.0 (n = 120 and 129);
the memorized set sits at 0.37. Rank correlations at the window level are
~0 overall because most windows are in the flat regime; within the memorized
set, higher final share goes with lower final loss (Spearman -0.36 at layer
28) and with *lower* recitation at 1.26T (-0.34): items memorized later ended
flatter. Reading: flattening accompanies *saturated* fit (recitation), not
partial predictability -- the same threshold the margin picture puts on
memorization. "Flat" and "fit to saturation" are the same state seen from the
representation and from the loss.

## 2026-09-05: the coupling law, predicted -- and where facts differ from verbatim

`probe_profiles_facts2.py`: per-direction energies on the final K-FAC bases for
ordinary windows, memorized suffix tokens and fact answer tokens, with three
definitions of the output-side gradient (loss gradient; unit-error gradient
g/(1-p_y); the margin gradient dm/dy), and the prediction of the private-block
coupling ratio memorized/facts as the product of the two sides' bulk energies:

| gradient used | predicted coupling ratio mem/facts | measured (noise-shift variance ratio) |
|---|---|---|
| loss gradient | 0.16 | 3.7 |
| unit-error gradient | 1.72 | 3.7 |
| margin gradient | **3.04** | 3.7 |

* The margin gradient is the right object and the product law works: coupling
  of a token's margin to a block = (margin-gradient energy in the block's
  output directions) x (activation energy in its input directions), predicted
  within 20% across populations with no free parameter. Together with
  Proposition 2 this makes per-token fragility computable from one
  forward-and-backward pass.
* Side decomposition (bulk energies relative to ordinary windows): input side
  memorized 2.0-2.2x, facts 2.15x -- identical; bulk share of activation energy
  0.45 vs 0.48. Output side (margin gradient): memorized 29-31x, facts 9-10x.
  The entire difference between verbatim text and facts is on the output side
  and it is one of *magnitude*, not shape: bulk shares of the margin gradient
  are 0.50-0.56 vs 0.47-0.56, and the profile exponents are alike (mem
  -0.55..-0.63, facts -0.46..-0.74). Memorized tokens' margins are three times
  more sensitive to the outputs of layers 23-25 overall; fact answers are
  produced more by other parts of the network.
* So facts and verbatim text have the *same* spectral character within these
  layers -- both item-specific, both half-whitened on the input side (facts
  even more: -0.73..-0.81) -- and differ in how much of their computation
  passes through these layers. "Location" in the sense that matters is
  across layers, not across the spectrum. Flatness marks item-specific
  content in general (facts included), which is exactly what the capability
  suite showed: every bulk edit costs recall first.

## 2026-09-05: deriving -- (1) the shape of the whitening law, (2) the noise mean shift

**Shape of the law** (per-quintile local exponents of E_mem/E_win vs E_win,
layer 24): input side [-0.50, -0.74, -0.78, -0.77, -0.43], output side (up)
[-0.43, -0.70, -0.79, -0.94, -0.35]; mean log10 ratio by quintile on the input
side +0.44, +0.34, +0.23, +0.08, -0.21. So the "one-third power" is a global
fit to a concave curve: memorized text has 2.75x the ordinary energy in the
flattest fifth of input directions and 0.6x in the head, with the steepest
reallocation in the middle of the spectrum and shallower slopes at both ends.
A derivation should target this monotone reallocation of a fixed total energy
(memorized total activation energy is 0.87-0.95x ordinary), not an exact 1/3.
Candidate mechanisms and what they predict: additive drive vs curvature
restoring force under SGD -> U-shaped ratio (flat + 1/mu), refuted by the
monotone curve; optimiser preconditioning (Adam normalises per-coordinate
updates by sqrt of the population second moment) -> stored-content ratio
~ kappa^-1/2 per side, the right direction and the exponent of the curvature
share, but the activation profile is a property of the representation
produced by earlier layers, so the link is indirect. Open.

**The noise mean shift, derived.** Gaussian noise sigma G_S Z A_S^T on
gate_proj adds to unit j's pre-activation a Gaussian eps_j with variance
sigma^2 ||A_S^T a||^2 rho_j (rho_j = ||G_S^T e_j||^2 ~ 0.6); noise on up_proj is
independent and averages out at first order. Hence the mean output under
noise is that of the model with SiLU replaced by its Gaussian smoothing,
E[silu(h + eps)], which equals silu(h) + 1/2 s^2 silu''(h) for small s (the
quadratic law) and grows linearly in s for large s (saturation, since silu
is asymptotically linear). Test running (`probe_smoothed_gate.py`): margins of
the smoothed-gate model at the seven norms vs the measured mean shifts under
actual noise, for memorized and fact tokens; then strict recitation
re-predicted with the derived mean and the measured spread.

**First-order forgetting law, per token** (`probe_spread_predict.py`, 120
memorized items x 48 tokens, exact autograd of each token's margin with respect
to the gate pre-activations and up outputs of layers 23-25 at *all* positions,
so that Var(dm_t) = sigma^2 sum_l sum_s ||G_S^T dm_t/dh_s||^2 ||A_S^T a_s||^2
includes the paths through attention from earlier positions): predicted vs
measured per-token spread (two seeds, four norms) -- log-log correlation 0.60
per token, 0.76 per item; median measured/predicted 0.75, mean 0.0233 vs
0.0273 predicted (17% high). The measured per-token spread has ~35% sampling
error from 8 observations, so 0.60 is near its ceiling. The first-order law
gives the scale of the random shift to ~20% and ranks tokens and items, with a
mild overshoot consistent with saturation of the largest shifts.

**Gate smoothing refuted; the mean shift is normalisation shrinkage.**
`probe_smoothed_gate.py`: replacing SiLU by its Gaussian smoothing (the exact
mean of the gate under the block noise) shifts margins by +0.002 to +0.005,
against measured -0.136 (norm 19) and -0.564 (38). The convexity of the gate
is *not* the mechanism. Per-token analysis of the measured noise shifts
(`probe_margins.pt`): the coefficient c = -E[dm]/m is 0.0122, 0.0508, 0.1155,
0.2006 at norms 19/38/57/76, i.e. c/n^2 = 3.4-3.6e-5 constant (exactly
quadratic), and identical for facts (0.0124, 0.0502, 0.1094, 0.1836) -- but per
token the shift correlates weakly with the margin (corr -0.03..-0.13; deciles
at norm 76: margin 3 -> shift -1.2, margin 17 -> -2.6, sub-proportional) and
better with the token's *removal* shift (corr 0.21 -> 0.56 with norm; coef
0.53 at 76). Joint regression at 76: shift = -0.089 m_t + 0.41 x removal_t.
Two-term reading, both normalisation effects of OLMo-2's architecture:
(i) the final RMSNorm sees extra variance from the noise and shrinks all
logits, hence margins, proportionally (coef k1 sigma^2 ~ 0.09 at norm 76);
(ii) each edited layer's post-feedforward RMSNorm sees extra variance in the
MLP output and shrinks that layer's genuine contribution -- whose effect on the
margin is the removal shift -- by a factor f ~ 0.41 at norm 76. Variance-budget
probe running to check both factors against the measured RMS ratios.

**Variance budget** (`probe_norm_shrink.py`, norm 76, memorized suffix tokens):
final residual RMS +0.09%, logit RMS unchanged -> the final norm is not the
mechanism (implied shrink 0.001 vs 0.20 needed). The edited layers' MLP
outputs gain 40/44/48% variance (RMS x1.185/1.20/1.22 at layers 23/24/25); the
post-feedforward RMSNorm therefore scales each layer's genuine contribution by
1/r = 0.84/0.83/0.82 (shrink 16-18% per layer). Three layers' MLP contributions
to a memorized token's margin shrunk by ~17% is the right size for the
measured -2.2 if those contributions total ~13 logits, and it explains the
regression on the removal shift (the block's contribution is a proxy for the
layer's). Mechanism: RMSNorm(x + n) = x/rms(x+n) + n/rms(x+n): the noise
enters as extra variance in the normaliser's denominator and shrinks the
signal it rides on. Predicts quadratic scaling (shrink ~ v/2s^2 ~ sigma^2),
identical coefficients for facts and memorized text relative to their layer
contributions, and saturation f = 1 - 1/sqrt(1 + v/s^2) (at norm 120,
v/s^2 ~ 1.1 -> f = 0.31 instead of the quadratic 0.44). Testing by running the
noise-free model with the three post-norm outputs scaled by 1/r_t (measured
r, then r predicted from sigma to first order).

**Post-norm shrinkage refuted as the mechanism** (`probe_postnorm_predict.py`):
running the noise-free model with each edited layer's normalised MLP output
scaled by the measured 1/r (r = 1.185/1.20/1.22 at norm 76, i.e. the 16-18%
shrink the noise imposes) moves the mean margin by -0.018 (norm 76) and +0.041
(38), against measured -2.185 and -0.564. Per token the correlation with the
measured shift is 0.19-0.36. So (a) the first-order variance prediction of r
from sigma is right (predicted-r and measured-r runs agree), but (b) the
three layers' MLP outputs have almost no *net* first-order effect on the margin
when scaled as a whole: their contributions to the target and to the best
competitor nearly cancel. The block removal shift (-2.9) is a different
quantity: removing a specific part of the output, not scaling all of it.

Where the systematic shift comes from, then: the zero-mean part n/rms(x+n)
enters the residual stream (tiny: the final residual RMS changes by 0.09%) and
the later layers (26-31: attention softmaxes, gates, norms) respond at second
order. E[dm_t] = 1/2 tr(H_t Sigma_t) with H_t the Hessian of the token's margin
w.r.t. the edited layers' outputs and Sigma_t the noise covariance there
(~ sigma^2 x the token's block input energy x the block's output structure).
Everything measured follows: quadratic in sigma; proportional to the token's
coupling (corr 0.56 with the removal shift); population coefficients equal
relative to margin at the population level; saturation once the perturbation
leaves the quadratic regime. What is not derived is the constant: the mean
downstream concavity of the margin, measured here as -E[dm]/Var[dm] ~ 0.22-0.32
per logit, a property of layers 26-31. Locating it (attention vs MLP) is an
open, cheap experiment (frozen attention patterns); it is not needed for the
memorization theory, where the shift's role is the noise/removal selectivity,
which the coupling law already fixes.

Three mechanisms tested for one number, two refuted by construction rather
than by fitting: this is the kind of test the theory should keep facing.

## 2026-09-05: targeted coherent removal (the theory's edit), with a held-out split

Guillaume: do the targeted edit, then the per-layer coupling map, mindful of
overfitting. Design (`make_targeted_edit.py`): split the 1054 memorized items
50/50 (seed 0). Direction = the margin gradient of the *train* half summed over
suffix tokens, projected onto the private block of gate/up in layers 23-25
(G_S G_S^T grad A_S A_S^T), sign chosen to reduce margins, per-matrix norm
matched to the deletion / noise runs (19, 38, 57, 76). Evaluate memorized
train and held-out halves separately, plus typical, Pile, arithmetic; then
recall (NQ/PopQA/Jeopardy) and HellaSwag/DROP through olmo-eval for the norm-38
and norm-76 models.

Predictions before running:
* P1 (coherence): forgetting of *train* items per unit norm far exceeds block
  deletion's (coherent AND aligned): strict recitation of the train half well
  below 0.18 at norm 76, and already low at norm 38 where deletion gives 0.73.
* P2 (orthogonal per-token code): transfer to *held-out* items is weak --
  their strict recitation at equal norm close to what isotropic noise of that
  norm gives (0.79 at 38, 0.28 at 76), not to what the train half shows. If
  transfer is strong instead, the memorized set shares directions the
  per-token picture missed, and that would be the more interesting outcome.
* P3 (shape invariance): in-distribution collateral equals that of any bulk
  edit of the same energy: typical +0.005 at 38, +0.02 at 76.
* P4 (coupling law): recall cost at equal norm below deletion's (facts couple
  weakly to the train items' directions), and at matched *train* forgetting far
  below it.

## 2026-09-05: per-layer coupling map (item 1 of the plan)

`probe_layer_coupling.py`: first-order sensitivity of a population's margins to
an isotropic perturbation of layer l's gate/up weights,
C_l = mean over targets of sum_s ||a_{l,s}||^2 sum_t ||dm_t/d(h,u)_{l,s}||^2
(all positions, random-sign trick for the sum over targets), for memorized
suffix tokens, fact answer tokens and ordinary windows (600 / 316 / 600).

| layer | mem/windows | mem/facts | facts/windows |
|---|---|---|---|
| 0-3 | 7-13 | 2.0-6.3 | 2-4 |
| 8-13 | 6.7-7.8 | 2.7-3.1 | 2.2-2.7 |
| 14-16 | 8.8-11.1 | 3.3-3.7 | 2.7-3.1 |
| 17 / 18 / 19 | 13.4 / 15.0 / 17.0 | 4.21 / 4.40 / 4.51 | 3.2 / 3.4 / 3.8 |
| 20 / 21 / 22 | 20.6 / 22.6 / 24.1 | 4.14 / 3.92 / 3.64 | 5.0 / 5.8 / 6.6 |
| 23 / 24 / 25 | 24.2 / 24.3 / 23.4 | 3.31 / 3.01 / 2.70 | 7.3 / 8.1 / 8.7 |
| 26 / 27 / 28 | 20.4 / 17.3 / 14.1 | 2.13 / 1.81 / 1.51 | 9.6 / 9.6 / 9.4 |
| 29 / 30 / 31 | 11.2 / 7.9 / 3.5 | 1.34 / 1.54 / 1.41 | 8.4 / 5.1 / 2.5 |

Share of total coupling in layers 23-25: memorized 0.19, facts 0.17, windows 0.10.

* The mem/facts ratio at 23-25 (3.0-3.3) reproduces the 3x output-side
  difference found on the block (probe_profiles_facts2), so the map is
  consistent with the block measurement.
* Memorized margins are most sensitive at layers 20-25; fact margins at
  24-28. The verbatim/facts separation therefore peaks *earlier* than the
  layers we have been editing: 4.2-4.5 at layers 17-20 against 3.0-3.3 at
  23-25 and < 2 at 27-31. Layers 26-31 are the worst place to edit for recall.
* Prediction (theory-driven placement): a private-block deletion at layers
  17-19 or 18-20 should cost ~1.4x less closed-book recall per unit of
  memorized-text forgetting than the 23-25 deletion, while forgetting less
  per unit norm (mem/windows 13-20 vs 24). Arithmetic is the risk: the layer
  sweep put arithmetic's flat-input dependence at layers 20-24, so 17-19
  should spare it and 18-20 should not entirely. Both being built and run
  through the recall suite now.

## 2026-09-05: targeted coherent removal -- results (P1 confirmed, P2 refuted, P3 violated)

`make_targeted_edit.py`: direction = bulk-projected margin gradient of the
train half (527 items). Only 6-19% of the margin gradient's energy lies in the
private block (most is in the head); alignment of the projected direction with
the block content itself is ~0.

| norm | train strict (from 0.996) | held-out strict (from 0.983) | typical | Pile | arith | reference at same norm (deletion / noise) |
|---|---|---|---|---|---|---|
| 19 | 0.019 | 0.156 | +0.0023 | +0.0001 | 0.892 | strict 0.917 / 0.91; typical +0.001 |
| 38 | 0.013 | 0.055 | +0.0137 | +0.022 | 0.887 | strict 0.731 / 0.789; typical +0.005 / +0.004 |
| 57 | 0.011 | 0.025 | +0.046 | +0.067 | 0.871 | 0.433 / 0.532; +0.013 / +0.010 |
| 76 | 0.008 | 0.013 | +0.106 | +0.143 | 0.838 | 0.180 / 0.282; +0.024 / +0.019 |

* **P1 confirmed, overwhelmingly.** Recitation of the train half falls from
  0.996 to 0.019 at norm 19, where deletion leaves 0.917. Coherent and aligned
  beats coherent-unaligned by an order of magnitude in norm.
* **P2 refuted: the transfer to held-out items is strong.** Items never used
  in the edit fall from 0.983 to 0.156 at norm 19 and 0.055 at 38. The
  memorized set shares a direction in the bulk of layers 23-25: the summed
  margin gradient of half the items has a large common component that removes
  the other half's recitation. The per-token orthogonality of noise couplings
  (within-item corr 0.01) coexists with a shared mean direction across items --
  a "recitation pathway" the set uses in common. This is the more interesting
  outcome flagged in the predictions, and it changes the theory: memorization
  is not only an item-specific code, it has a shared component, and the
  shared component is where the leverage is.
* **P3 violated: the targeted direction is not a generic bulk direction.** Its
  in-distribution collateral per unit norm is 2x (norm 19), 3x (38), 5x (76)
  that of energy-matched noise or deletion, and the Pile cost grows fast
  beyond norm 38. Shape invariance holds for *random* directions in the bulk
  and for removal of the content; this direction is special: the shared
  recitation component is also used by ordinary text (in-context copying?),
  so it sits on the higher-curvature end of the bulk for the population.
* **At matched held-out forgetting the edit is an order of magnitude better.**
  Held-out strict 0.156 at norm 19 vs deletion 0.180 at norm 76 and noise
  0.159 at norm 90: typical +0.0023 vs +0.024 / +0.026 (10x less), Pile
  +0.0001 vs +0.027 / +0.047 (~100x less), arithmetic unchanged vs -0.035 /
  -0.024. Recall (NQ/PopQA/Jeopardy) for the norm-38 and 76 models is running;
  norm 19 (and 10, 5) are being built and saved for the same test.
* Overfitting checks still to run: (a) a third memorized population not used
  anywhere -- the 129 ordinary dolmino windows the model recites; (b) the
  quotes benchmark (a different memorized distribution) via the pipeline;
  (c) the cosine between the two halves' gradient directions (how large the
  shared component is), full and bulk-projected.

**P4 confirmed: recall cost of the targeted edit** (olmo-eval, same subsets):

| model | held-out recitation | NQ | PopQA | Jeopardy | HellaSwag | DROP |
|---|---|---|---|---|---|---|
| targeted, norm 38 | 0.055 | -1.4 | -0.4 | -0.6 | -0.1 | +1.2 |
| targeted, norm 76 | 0.013 | -7.8 | -4.8 | -4.9 | -1.3 | -1.6 |
| deletion 23-25 (norm 76) | 0.180 | -7.9 | -8.0 | -2.7 | -1.3 | -0.1 |
| private noise 90 | 0.159 | -5.2 | -6.8 | -2.7 | -1.2 | +1.2 |

At norm 38 the targeted edit removes three times more held-out recitation than
deletion or noise and costs a fifth to a twentieth of their closed-book recall
(-1.4/-0.4 vs -7.9/-8.0 and -5.2/-6.8), with commonsense and DROP untouched.
At norm 76 its recall cost catches up with deletion's while forgetting is
essentially complete (0.013): the shared direction's collateral grows
super-linearly, as the typical/Pile costs showed, so the useful regime is the
small-norm one, where the theory's first-order account applies. Pending:
the third memorized population and the two-half cosine (running), the quotes
benchmark (queued), and the layer-placement deletions.

**How large is the shared component?** (`make_targeted_edit2.py`, first
output): cosine between the summed margin gradients of the two disjoint halves
of the memorized set, per matrix: full 0.74-0.92, bulk-projected 0.62-0.65,
complement (head) 0.76-0.94. Two sets of 527 unrelated documents produce
nearly the same weight-gradient direction. The shared recitation pathway is a
population-level object, stronger in the head than in the bulk; the edit uses
its bulk projection. Control needed: is it specific to memorized text or the
generic "raise any token's margin" direction? -- cosine with the summed margin
gradient of ordinary windows and of fact tokens (running next).

**Layer placement, population level** (`make_block_models.py --mode delete`):

| deletion | mem strict (from 0.99) | typical | Pile | arith |
|---|---|---|---|---|
| layers 17-19 | 0.542 | +0.016 | +0.007 | 0.872 |
| layers 18-20 | 0.460 | +0.017 | +0.013 | 0.787 |
| layers 23-25 | 0.204 | +0.021 | +0.027 | 0.827 |

As the coupling map predicted: less forgetting per unit norm at 17-20
(mem/windows 13-20 vs 24), Pile spared, and arithmetic hit at 18-20 (the
20-24 flat-input band) but not at 17-19. The recall test (the point of the
placement prediction: mem/facts 4.2-4.5 vs 3.0-3.3) is running on GPUs 4-5
after a vLLM start failure on GPUs shared with the DVC eval.

**Third-population control: the transfer is set-level.** (`make_targeted_edit2.py`)

| norm | train (from 0.996) | held-out Dolma (from 0.983) | recited dolmino windows (from 1.00) | clean | typical | Pile | arith |
|---|---|---|---|---|---|---|---|
| 5 | 0.049 | 0.480 | 0.984 | 0 | 0 | 0 | 0.892 |
| 10 | 0.027 | 0.309 | 0.969 | 0 | +0.0005 | 0 | 0.893 |
| 19 | 0.019 | 0.156 | 0.946 | 0 | +0.0023 | +0.0001 | 0.892 |

The edit removes 84% of the held-out *Dolma* items' recitation at norm 19 but
only 5% of the recitation of the 129 ordinary dolmino windows the model
recites -- a memorized population never used in the edit and drawn from a
different distribution. So the shared component of the memorized set's
margin gradient is shared by *that set* (source, format, style of the Dolma
duplicates), not by memorized text in general. Transfer across items within a
distribution is nearly complete; transfer across memorized distributions is
small. This is overfitting at the level of the distribution, as Guillaume
anticipated, and it bounds what a labelled edit buys: it removes what the
labelled examples share, which is much more than the examples themselves and
much less than "memorization". The quotes benchmark (pipeline run irate-prof)
will show the same thing on a third memorized distribution; the gradient
cosines against ordinary and fact tokens (running) will show how much of the
shared direction is a generic margin direction.

**Placement prediction confirmed, and exceeded** (recall suites for the
17-19 and 18-20 deletions):

| deletion | memorized forgotten (1 - strict) | NQ | PopQA | Jeopardy | HellaSwag | DROP | recall cost per unit forgotten (mean NQ/PopQA pts) |
|---|---|---|---|---|---|---|---|
| layers 17-19 | 0.45 | -1.8 | -2.2 | -0.3 | -2.0 | +1.2 | 4.4 |
| layers 18-20 | 0.53 | -3.7 | -3.0 | -1.1 | -2.3 | -0.3 | 6.3 |
| layers 23-25 | 0.79 | -7.9 | -8.0 | -2.7 | -1.3 | -0.1 | 10.1 |

The coupling map predicted ~1.4x less recall per unit forgetting at 17-19
(mem/facts 4.4 vs 3.1); measured 2.3x. Layer placement by the coupling ratio
works and is conservative. The price is HellaSwag (-2.0 vs -1.3) and, at
18-20, arithmetic (0.787). Per unit forgetting, the 17-19 deletion is the best
label-free edit we have; it forgets less in total (0.45 of items) because
memorized coupling is lower there, so reaching the 23-25 level of forgetting
would need more layers (e.g. 17-19 plus 23-25), which the map can now cost
before running.

**Specificity of the shared direction** (`probe_shared_direction.py`, cosines
of summed per-token margin gradients, full | bulk-projected): memorized train
vs held-out +0.854 | +0.634; memorized vs ordinary windows +0.16 | +0.004; vs
fact tokens +0.06 | +0.001; vs clean windows 0.00 | 0.00; windows vs facts
+0.05 | 0.00. Per-token margin-gradient norms: memorized 5.0, facts 5.7,
windows 0.49, clean 0.24. So the direction the two halves share is (i) large,
(ii) orthogonal in the bulk to what ordinary text and facts use -- which is
why its collateral and recall cost are tiny at small norm -- and (iii) shared
only within the Dolma set (third-population transfer 5%). Not a generic
margin direction, not general memorization: the set's common source and style,
written into the bulk of layers 23-25.

**Recall for the small-norm targeted models:**

| model | held-out Dolma recitation | recited-windows recitation | NQ | PopQA | Jeopardy | HellaSwag | DROP |
|---|---|---|---|---|---|---|---|
| targeted 10 | 0.309 | 0.969 | -0.2 | +0.0 | -0.6 | -0.1 | +0.3 |
| targeted 19 | 0.156 | 0.946 | -0.7 | +0.0 | -0.8 | -0.4 | +1.0 |
| targeted 38 | 0.055 | (n/a) | -1.4 | -0.4 | -0.6 | -0.1 | +1.2 |
| deletion 23-25 | 0.180 | -- | -7.9 | -8.0 | -2.7 | -1.3 | -0.1 |
| private noise 90 | 0.159 | -- | -5.2 | -6.8 | -2.7 | -1.2 | +1.2 |

At matched held-out forgetting (0.156 vs 0.159-0.180) the targeted edit costs
-0.7/0.0 recall points against -7.9/-8.0 (deletion) and -5.2/-6.8 (noise):
an order of magnitude, with commonsense and DROP untouched. Its reach is the
Dolma set (recited ordinary windows 0.946), so this is the trade-off *for a
memorized distribution one has examples of*, which is the realistic setting
for unlearning a corpus and not the setting for "remove memorization" in
general.
