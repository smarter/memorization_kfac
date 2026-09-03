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
pending) shows re-estimation adds ~1 sigma on GSM8K at best. Deprioritised.

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

---

## Backlog

Promising (benchmark-agnostic, next):
* **Breadth axis for the curvature spectrum.** Per direction: depth (mass) and
  breadth (participation entropy of per-sequence contributions). Conjecture:
  memorization = high depth from few contexts; rare skills = moderate depth,
  many contexts; general = both high. Test on held-out memorized vs clean
  Dolma text (the BSN sets) at matched depth. -> in progress
  (probe_populations).
* **Per-direction attribution of curvature to token populations** (memorized
  vs clean, rare vs frequent, numbers / entities / function words), with the
  coupled weights; characterise the directions plain K-FAC drops but the
  corrected curvature keeps. -> in progress.
* **Sides and locality:** A-side vs G-side specificity across all layers and
  attention.
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
