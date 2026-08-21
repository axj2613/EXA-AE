# Search-improvement plan & pre-registered gates

Derived from three evolutionary-NAS papers (Evolved Transformer / PDH; bilevel
multi-objective neuroevolution; ES-trains-Transformers robustness) mapped onto
this project's measured reality. **Pre-registered** so each iteration is judged
against fixed baselines and gates, not against the objective the search optimized.

## The honest ceiling (why we target what we target)

- **Reconstruction is near-linear-saturated.** A fair *cross-subject* ridge
  (fit on TRAIN subjects, evaluated on the 219 TEST subjects, mask 0.5) scores
  **R² = 0.315**; the evolved model scores **0.204**. Search on reconstruction MSE
  can at best close a 0.11 gap to a *linear* baseline it currently loses to.
- **Clinical prediction has headroom but the model loses to FC+ridge.** Classical
  functional-connectivity + ridge on the same TEST subjects: **sex AUC 0.907,
  age R² 0.119**. The model's best readout (reconstruction-error fingerprint):
  **sex AUC 0.754, age R² 0.10** (age: Pearson r 0.32, MAE 2.9 yr; report r/MAE,
  not R², for a 22–36 cohort). CLS embedding and attention-connectivity readouts
  were at/below chance.
- **Search dynamics are the concrete, fixable bottleneck** (depth-growth stall;
  noisy proxy, ρ≈0.3; pretrain→evolve brittleness). This is what the papers fix.

## Fixed baselines (reference lines on every chart)

| Metric | Baseline | Value |
|---|---|---|
| Reconstruction R² (TEST) | cross-subject ridge (ceiling) | **0.315** |
| Reconstruction R² (TEST) | current evolved model | 0.204 |
| Reconstruction R² (TEST) | BrainLM (their protocol, not comparable) | 0.28 |
| Sex AUC (TEST) | FC + ridge | **0.907** |
| Sex AUC (TEST) | model fingerprint readout | 0.754 |
| Age (TEST) | FC + ridge | R² 0.119 |
| Age (TEST) | model fingerprint readout | R² 0.10 / r 0.32 / MAE 2.9 yr |

## Test protocol (non-negotiable, given prior measurement errors)

- Subject-level splits, frozen (`subject_split.json`); **TEST never touched**
  during search, pretraining, or probe fitting.
- Reconstruction: masked-patch R² over the full TEST set (≈13M masked values).
- Clinical: k-fold CV over subjects, no leakage; **report Pearson r + MAE for age**
  (R² is calibration-penalized and misleads in a restricted-age cohort); never a
  post-hoc rescaled R² (uses test labels → invalid).
- Every change ships only if it beats the prior iteration by the pre-set margin
  **on held-out**, not on the fitness the search optimized.

## Ranked method catalog

| # | Method | Source | Targets | Status |
|---|---|---|---|---|
| 0 | Re-measure proxy fidelity + lock baselines | ours + ET | diagnostic | **run next** |
| 1 | **Progressive Dynamic Hurdles** | Evolved Transformer | search efficiency, depth-growth | **implemented** |
| 2 | Seed robustification + lower search LR/mutation | Lorenc (ES brittleness) | pretrain→evolve stall | planned |
| 3 | Multi-objective (MSE vs params): λ-parsimony → NSGA-II | bilevel + ET | efficiency, bloat | planned |
| 4 | Downstream-aware fitness (evolve on fingerprint clinical CV) | synthesis | clinical prediction | **implemented** |
| 5 | Richer block vocabulary (GLU/gating, activation/norm search) | ET | representation | exploratory |
| 6 | Novelty / quality-diversity | Lorenc (intro) | diversity, escape stalls | exploratory |

## Phased plan with gates

### Phase 0 — Instrument before optimizing
- **Run:** `proxy_fidelity_experiment.py` under the current single-patch config,
  from the pretrained seed.
- **Gate:** ρ ≥ ~0.7 → the cheap proxy is trustworthy; a bigger *fixed* budget may
  suffice and PDH is optional. ρ still low → empirically justifies PDH (the ET
  authors hit exactly this and concluded "no proxy worked → PDH").

### Phase 1 — Fix the search engine  *(PDH implemented; ready to A/B)*
- **1a. PDH** (`USE_PDH=True`): stage-0 cheap screen (`PDH_STEP_INCREMENTS[0]`),
  hurdle-clearers earn more steps; hurdles = population mean, created every
  `PDH_MODELS_PER_HURDLE` genomes. Escalation accumulates via Lamarckian inheritance.
- **1b. Robustification:** lower search LR when fine-tuning a pretrained genome;
  lower mutation rate; optionally weight-noise/flat-minima pretraining.
- **Hypothesis:** PDH lets deeper genomes prove themselves → the population grows
  depth and clears the shallow-seed ceiling.
- **Metric:** best held-out reconstruction R²; fitness-vs-generation curve;
  encoder/decoder block counts of the best genome over time.
- **Success gate:** best genome (i) grows depth beyond the seed **and** (ii) beats
  the current 0.204 on TEST, ideally approaching the 0.315 ridge line, at equal or
  less total compute than the fixed-budget run.
- **Kill gate:** if PDH still cannot beat the 0.315 ridge on reconstruction, that
  **confirms** linear saturation → stop optimizing reconstruction, commit to Phase 3.

### Phase 2 — Make efficiency an explicit objective
- **2a.** Cheap first: `fitness = MSE + λ·params`.
- **2b.** NSGA-II non-dominated sort on (MSE, active-params) → a Pareto front;
  structurally curbs the bloat that caused our OOMs.
- **Success gate:** a knee-point model matching current R² at materially fewer
  params, or higher R² at equal params; report hypervolume vs the single-objective run.

### Phase 3 — Strategic pivot: evolve for clinical signal  *(IMPLEMENTED; reconstruction confirmed at ceiling)*
- Fitness includes a **downstream-probe score**: per genome, compute the
  reconstruction-error fingerprint on a fixed probe set and run an internal CV
  ridge/logreg for age/sex; reward that. PDH ensures only hurdle-clearing genomes
  pay the probe cost.
- **Implementation** (`evolution/downstream_probe.py`, wired via `parallel_training`
  + `progressive_hurdles`):
  - `DownstreamProbe` caches a held-out probe split's windows once, then for any
    genome computes per-parcel fingerprints (vectorized), reduces with PCA-30, and
    runs KFold/StratifiedKFold CV → `combined = 2·(sexAUC−0.5)₊ + (ageR²)₊`.
  - Selection fitness = `recon_MSE − PROBE_FITNESS_WEIGHT · combined` (minimized);
    **PDH hurdles keep escalating on pure recon MSE**, preserved as
    `genome.recon_fitness`, so training-budget allocation stays reconstruction-driven
    and only genomes clearing `PROBE_MIN_STAGE` pay the probe cost.
  - **Probe split = VAL** (held out of reconstruction training → out-of-sample
    fingerprints, a more honest selection signal than in-sample TRAIN); TEST is never
    touched. Labels are never used in pretraining, so no label leakage.
  - Notebook toggles: `USE_DOWNSTREAM_PROBE`, `PROBE_SPLIT`, `PROBE_FITNESS_WEIGHT`,
    `PROBE_MIN_STAGE`, `PROBE_WINDOWS_PER_SUBJECT`, `PROBE_N_PCA`.
  - Validated: on the current best model the VAL probe reads sex AUC ≈ 0.80; in the
    integration test a genome with sex AUC 0.64 (chance-level MSE) is correctly
    promoted above lower-MSE, chance-clinical rivals.
- **Success gate:** narrow the gap to FC+ridge on TEST (e.g. sex AUC → ~0.80–0.85).
- **Risk control:** frozen probe set disjoint from TEST; nested CV; pre-registered margin.

### Phase 4 (exploratory)
Richer block vocabulary (after PDH affords the harder search); novelty/QD if the
population collapses to one lineage.

## Immediate next step
Phase 1 (PDH) is done: reconstruction reached **R² 0.275** on TEST — matching BrainLM
(0.28) at 3.0M params but still under the **0.315 linear ceiling**, which triggers the
Phase-1 **kill gate** (reconstruction is linearly saturated). So we commit to **Phase 3**,
now implemented. Run the notebook with `USE_DOWNSTREAM_PROBE=True` from the pretrained
seed; watch that the population's `probe sex AUC` climbs while `recon MSE` holds, then do
the honest **TEST** evaluation (`fingerprint_readout.py`) and check the success gate
(sex AUC 0.794 → ~0.80–0.85). Tune `PROBE_FITNESS_WEIGHT` if reconstruction degrades too
far in trade for clinical signal. Phase 2 (multi-objective params) remains available if a
smaller/cleaner front is wanted.
