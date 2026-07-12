# EXA-STAR: Evolutionary eXploration of Augmenting Spatio-Temporal ARchitectures

This repository contains two related neuro-evolution capabilities built on the EXAMM
(Evolutionary eXploration of Augmenting Memory Models) lineage:

1. **EXA-AE** — evolved LSTM recurrent autoencoders for multivariate time-series anomaly
   detection (the original work).
2. **Neuro-evolved ViT-MAE for fMRI** — evolved masked-autoencoder vision transformers that learn
   representations of brain activity from Human Connectome Project (HCP) fMRI, in the style of
   [BrainLM](https://github.com/vandijklab/BrainLM), but discovered by evolution rather than
   hand-designed. The evolutionary search produces models **hundreds to thousands of times smaller**
   than BrainLM's 111M/650M parameters.

Both share the same underlying machinery: a genome is a graph of nodes and edges, and EXAMM-style
mutation/crossover operators grow and recombine that graph, training each candidate briefly with
Lamarckian weight inheritance.

---

## Installation

```
pip install -r requirements.txt
```

The core training path needs `torch`, `numpy<2.0`, `loguru`, `pandas`, `matplotlib`, and
`graphviz`. Some scripts additionally use `scikit-learn`/`scipy` (evaluation), `boto3`/`tqdm` (S3
sync, Kaggle) — these are preinstalled on Kaggle.

---

## Part 1 — EXA-AE: evolved autoencoders for anomaly detection

EXA-AE co-evolves the asymmetric encoder and decoder of an autoencoder at the node/edge level,
producing compact LSTM recurrent autoencoders. Reconstruction residuals are combined with a
one-class SVM to define anomaly thresholds.

**Run the search:**
```
python exa_ae.py            # configured by exa_ae_config.ini
```
Config keys: `training_data`, `parameters`, `bidirectional_ae`, `num_generations`,
`num_iterations`, `learning_rate`. The best genome is pickled to `test_genomes/`.

**Evaluate a genome** (statistics, reconstructions, graphviz diagram):
```
python evaluation_scripts/evaluate_genome.py <genome_pkl> <testing_data> <output_filename>
```

**Anomaly detection** from reconstructions:
```
python evaluation_scripts/anomaly_detection_threshold.py <train_predictions> <test_predictions> <anomaly_labels>
```

---

## Part 2 — Neuro-evolved ViT-MAE foundation models for fMRI

A masked-autoencoder vision transformer is trained self-supervised on fMRI: each recording is a
`(parcels × time)` window, split into per-parcel temporal patches; a random subset of patches is
masked, an encoder processes the visible tokens, and a decoder reconstructs the masked ones. Each
patch token is the sum of a learned signal projection, a projection of the parcel's real 3D atlas
coordinate, and a sinusoidal temporal embedding (following BrainLM).

There are **two model families**:

### 2a. Fixed-topology ViT-MAE (hyperparameter search)

A standard ViT-MAE whose macro-hyperparameters (depth, `d_model`, heads, patch/mask sizes) are
evolved. A baseline for comparison.

- Model: `genomes/transformer_model/vision_transformer_mae.py`
- Genome: `genomes/vit_mae_genome.py`
- Run: `python exa_vit_mae.py` (config `exa_vit_mae_config.ini`)

### 2b. Evolvable block-graph ViT-MAE (the main capability)

Here the encoder **and** decoder internals are an **evolvable node/edge graph**. EXAMM
mutation/crossover compose and connect *block* nodes — mirroring how the original EXAMM paper mixes
LSTM/GRU/simple-neuron cells within one evolved RNN, but at token-block granularity. The fixed
scaffolding (patchify, positional embeddings, masking, decoder head) surrounds the evolvable graph.

Block node types (`genomes/nodes/`):

| Node type | What it does |
|---|---|
| `AttentionBlockNode` | full self-attention + FFN transformer block (mixes across all tokens) |
| `SimpleBlockNode` | residual per-token affine + tanh (no cross-token mixing) |
| `SequenceLSTMBlockNode` | LSTM over the token sequence (generic sequential mixing) |
| `TemporalLSTMBlockNode` | LSTM over the *true per-parcel temporal axis* (decoder region only) |

Newly mutated-in blocks use **near-identity warm-start** so insertion is non-destructive to the
parent's learned function. Crossover blends matched parent weights
(`weight_generators/lamarckian_block_weight_generator.py`).

- Genome: `genomes/vision_transformer_block_genome.py`
- Generators/selector: `evolution/vision_transformer_block_{node,edge}_generator.py`,
  `evolution/vision_transformer_block_reproduction_selector.py`
- Run locally: `python exa_vit_mae_evolved.py` (config `exa_vit_mae_evolved_config.ini`)
- Run on Kaggle GPUs: `exa_vit_mae_evolved_kaggle.ipynb`

**Choosing which block types evolve** — set `node_types` in the config to compare, e.g., an
attention-only model against a mixed-cell model (the EXAMM single-cell-type vs. all-types
experiment):
```ini
node_types = attention,simple,sequence_lstm,temporal_lstm   # mixed-cell search
node_types = attention                                       # attention-only search
```
(`temporal_lstm` is decoder-region only and cannot be the sole type.)

---

## Data pipeline (HCP)

The HCP corpus is organized one directory per subject, each holding that subject's ~18 `.npz`
recordings (4 resting-state + ~14 task) in the S3 layout `aal_424/<subject_id>/<recording>.npz`.

- **`scripts/sync_hcp_from_s3.py`** — one-time mirror of the S3 bucket to a local directory
  (resumable), which you then upload as a Kaggle Dataset.
- **`time_series/hcp_window_dataset.py` (`HCPWindowDataset`)** — the training dataset:
  - deterministic, persisted **subject-level 70/10/20 train/val/test split** (a subject's
    recordings never straddle the split — no identity leakage);
  - **train-split-frozen per-parcel z-score** normalization, reused for val/test;
  - lazy load from local disk with an LRU cache; rest/task tagging;
  - **length-filtered windowing** — recordings shorter than the window (truncated/corrupt runs) are
    excluded and logged; a window of `120` (÷ `time_patch_size` = 20 → 6 temporal patches) keeps
    ~99.7% of recordings including the short EMOTION task.
- **`time_series/fmri_patch_dataset.py` (`FMRIPatchDataset`)** — a simpler loader for a handful of
  local `.npz` files (small experiments / tests).

---

## Training on Kaggle (T4×2)

`exa_vit_mae_evolved_kaggle.ipynb` is the turnkey driver:

1. Upload the synced HCP corpus **once** as a private Kaggle Dataset; point `HCP_ROOT` at it.
2. The notebook's setup cell **auto-clones/pulls this repo** (`REPO_URL`/`REPO_BRANCH`) so the code
   always matches the notebook — no manual file migration. (For a private repo, put a token in
   `REPO_URL`.)
3. It builds the dataset, evolves genomes with GPU mixed precision, and **checkpoints to
   `/kaggle/working`** so it resumes across Kaggle's ~12h sessions.

Because Kaggle wipes `/kaggle/working` between sessions, enable **Persistence → Files only** in the
notebook settings (also required for resume), or **Save Version** / download the outputs.

**Key config** (`exa_vit_mae_evolved_config.ini` / notebook cell): `training_data`,
`atlas_coordinates`, `window_length`, `d_model`, `num_heads`, `d_ff`, `dropout`, `mask_ratio`,
`node_types`, `population_size`, `num_generations`, and the per-genome budget
(`num_iterations`, `batches_per_iteration`, `batch_size`, `learning_rate`). Note the per-genome
training cost is fixed by that budget and **independent of dataset size** — evolution trains each
genome briefly on a random slice of windows and evolves many; use `num_generations` to exploit more
data.

**Checkpoint/resume** (`evolution/checkpoint.py`) saves the population, the global innovation
counter, the completed generation, and RNG state, so a timed-out session resumes deterministically.

---

## Parameter-efficiency reporting

Neuro-evolution's payoff is finding *compact* architectures, so every genome is measured
(`VisionTransformerBlockGenome.parameter_report()`), logged during training, and stored on
`genome.complexity` (saved in the pickle). It reports the **active-graph** trainable parameter count
(fixed scaffolding + evolved blocks), plus active node/edge and per-block-type counts — e.g. a
strong 2–3 block genome is in the low hundreds of thousands of parameters, vs. BrainLM's
111M/650M. Selection itself stays pure validation-reconstruction MSE (EXAMM-faithful); the counts
support post-hoc accuracy-vs-size analysis.

---

## Evaluation (BrainLM-style)

The evolved model reuses the artifacts saved to `/kaggle/working`: `best_genome.pkl`,
`subject_split.json`, and `norm_stats.npz` (all three are needed so held-out test subjects and
normalization match training).

- **Reconstruction** — `evaluation_scripts/brain_lm/inference.py`: masked-patch reconstruction and
  MSE/R² on held-out windows.
- **Per-subject embeddings → clinical regression** —
  `evaluation_scripts/brain_lm/subject_embeddings.py` extracts one mean-CLS embedding per held-out
  **test** subject (the analog of the BrainLM `get_cls.py` flow) and saves an `.npz` in the format
  consumed by a downstream clinical/behavioral regression (`eval_cls.py`: age, gender, personality).
- **Attention introspection** — `evaluation_scripts/brain_lm/attention_analysis.py`: per-parcel CLS
  attention maps and latent extraction (`encode_for_analysis`) for functional-network analysis.

---

## Repository map

```
genomes/
  genome.py                          base graph genome
  recurrent_genome.py, autoencoder_genome.py, bAE_genome.py   EXA-AE (scalar LSTM) genomes
  vit_mae_genome.py                  fixed-topology ViT-MAE genome
  vision_transformer_block_genome.py evolvable block-graph ViT-MAE genome
  nodes/                             scalar nodes (LSTM, ...) + block nodes (attention/simple/lstm)
  edges/                             scalar recurrent edges + block edges
  transformer_model/                 attention/encoder building blocks + VisionTransformerMAE
evolution/                           EXAMM driver, node/edge generators, reproduction selectors,
                                     checkpoint.py
reproduction/                        mutation/crossover operators (shared)
weight_generators/                   Kaiming / Lamarckian (scalar + block) weight init
population/                          steady-state population strategy
time_series/                         TimeSeries, FMRIPatchDataset, HCPWindowDataset
scripts/sync_hcp_from_s3.py          S3 -> local corpus sync
evaluation_scripts/                  evaluate_genome, anomaly detection, brain_lm/ (fMRI eval)
exa_ae.py, exa_vit_mae.py, exa_vit_mae_evolved.py            driver scripts (+ .ini configs)
exa_vit_mae_evolved_kaggle.ipynb     Kaggle GPU training notebook
```
