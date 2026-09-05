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
| remove23_28 (delete, layers 23-28) | building | | | |

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
* Queue (batch 10, same pipeline as batch 9: rho=1, start_from_model):
  remove23_25, remove23_25_half, remove23_28, then remove19_28 as the
  negative control.
