# Neuro-Evolved ViT-MAE for Brain fMRI

Discover **compact masked-autoencoder vision transformers** for Human Connectome Project (HCP)
resting-state fMRI by **neuro-evolution** — in the spirit of
[BrainLM](https://github.com/vandijklab/BrainLM), but with the architecture *evolved* rather than
hand-designed. Built on the EXAMM / EXA-STAR lineage: a genome is a graph of *block* nodes and
edges that EXAMM-style mutation and crossover grow and recombine, training each candidate briefly
with Lamarckian weight inheritance.

**Headline result.** An evolved **3.0M-parameter** model matches BrainLM's reported masked-signal
reconstruction (**R² 0.275 vs. 0.28**) with **~37× fewer parameters**, and its per-parcel
reconstruction-error *fingerprint* predicts **general cognitive ability at r = 0.44** on held-out
subjects — signal that localizes to the frontoparietal-control and default-mode networks and
dissociates cleanly from age/sex (sensorimotor networks). See [Results](#results).

---

## How it works

A masked-autoencoder vision transformer is trained self-supervised on fMRI. Each recording becomes
a `(parcels × time)` window; tokens are per-parcel temporal patches. A random subset of tokens is
masked, an encoder processes the visible ones, and a decoder reconstructs the masked ones. Each
token embeds a learned signal projection **+** a projection of the parcel's real 3-D atlas
coordinate **+** a sinusoidal temporal code (following BrainLM).

What makes it *evolved*: the encoder and decoder **internals are an evolvable node/edge graph**.
EXAMM mutation/crossover compose and wire *block* nodes, the way the original EXAMM paper mixes
LSTM/GRU/simple cells inside one evolved RNN — but here at token-block granularity. The fixed
scaffolding (patchify, positional embeddings, masking, decoder head) surrounds the evolvable graph.

| Block node | Role |
|---|---|
| `AttentionBlockNode` | self-attention + FFN transformer block (mixes across all tokens) |
| `SimpleBlockNode` | residual per-token affine + tanh (no cross-token mixing) |
| `SequenceLSTMBlockNode` | LSTM over the token sequence |
| `TemporalLSTMBlockNode` | LSTM over the true per-parcel temporal axis (kept for multi-patch configs; excluded from the default single-patch pool) |

Newly mutated-in blocks use **near-identity warm-start** so insertion barely perturbs the parent's
learned function; crossover blends matched parent weights
([`weight_generators/lamarckian_block_weight_generator.py`](weight_generators/lamarckian_block_weight_generator.py)).

**Key mechanisms**

- **Single-temporal-patch config** — the learnable signal in HCP at this scale is *spatial*
  (functional connectivity), so `window_length == time_patch_size` gives one temporal patch (424
  spatial tokens), which trains cleanly toward the reconstruction ceiling.
- **Progressive Dynamic Hurdles** ([`evolution/progressive_hurdles.py`](evolution/progressive_hurdles.py))
  — adaptive per-genome compute (So et al., *The Evolved Transformer*): genomes that clear a
  population-mean fitness hurdle earn more training, concentrating budget on promising lineages and
  letting warm-started deeper genomes prove themselves.
- **Downstream-aware fitness** ([`evolution/downstream_probe.py`](evolution/downstream_probe.py))
  — optional: fold a held-out clinical-prediction score (from the reconstruction fingerprint) into
  the selection fitness, so evolution can optimize for clinical signal, not just reconstruction MSE.
- **Feed-forward-only block edges** — the reproduction operators are configured with
  `allow_recurrent=False` for this genome type, since BlockEdges have no recurrence.

---

## Results

All numbers are on the **held-out test split** (subject-level; test subjects never seen in
training, feature selection, or model selection).

**Reconstruction** (masked-patch, ≈8.7M values)

| Model | R² | Params |
|---|---|---|
| Predict-the-mean | 0.00 | — |
| **Evolved (this repo)** | **0.275** | **3.0M** |
| BrainLM (reported) | 0.28 | 111M |
| Linear-ridge ceiling | 0.315 | — |

**Clinical / behavioral readout** (per-parcel reconstruction-error fingerprint)

| Target | Score |
|---|---|
| Sex (M/F) | AUC 0.79 |
| Age | r 0.38 · R² 0.14 · MAE 2.9 yr |
| General cognitive factor (*g*) | r 0.44 · R² 0.18 |
| Big-Five personality | ≈ chance (near-zero) |

The cognition signal concentrates in the **frontoparietal-control and default-mode** networks; the
age/sex signal in **sensorimotor / attention** networks — a double dissociation that matches the
neuroscience and confirms the model reads distinct, anatomically-appropriate signatures rather than
one global axis. Full analysis lives in
[`evaluation_scripts/brain_lm/`](evaluation_scripts/brain_lm/); the research roadmap and
pre-registered evaluation gates are in [`docs/improvement_plan.md`](docs/improvement_plan.md).

---

## Installation

```bash
pip install -r requirements.txt
```

Core training needs `torch`, `numpy<2.0`, `pandas`, `matplotlib`, `loguru`, `graphviz`. Evaluation
and the behavioral analyses additionally use `scikit-learn`, `scipy`, `umap-learn`, and `nilearn`
(Yeo-network atlas). All are preinstalled or `pip`-installable on Kaggle.

---

## Data (HCP)

The corpus is one directory per subject, each holding that subject's ~18 parcellated `.npz`
recordings (4 resting-state + ~14 task), in the S3 layout `aal_424/<subject_id>/<recording>.npz`.

- [`scripts/sync_hcp_from_s3.py`](scripts/sync_hcp_from_s3.py) mirrors the S3 bucket to a local
  directory (resumable); you then upload that once as a private Kaggle Dataset.
- [`time_series/hcp_window_dataset.py`](time_series/hcp_window_dataset.py) (`HCPWindowDataset`) is
  the training dataset: a deterministic, persisted **subject-level 70/10/20 split** (no
  within-subject leakage), **train-frozen per-parcel z-scoring** reused for val/test, LRU-cached
  lazy loading, and length-filtered windowing.

> **Note.** HCP subject-level data are governed by the HCP Data Use Terms and are **not** included
> in this repo (they are git-ignored). Only the small `A424_Coordinates.dat` atlas file is tracked.
> Obtain HCP data yourself via ConnectomeDB / the S3 mirror.

---

## Workflow

The end-to-end pipeline runs as three Kaggle notebooks (turnkey, GPU, auto-resume) plus a local
`.py` entry point. Each notebook auto-clones/pulls this repo so the code always matches.

| Stage | Notebook / script | Output |
|---|---|---|
| 1. Pretrain a seed | `pretrain_seed_kaggle.ipynb` | `pretrained_seed.pkl` |
| 2. Evolve the search | `exa_vit_mae_evolved_kaggle.ipynb` · or `python exa_vit_mae_evolved.py` | `best_genome.pkl`, checkpoints |
| 3. Train the winner | `train_final_model_kaggle.ipynb` | `final_model.pkl` |
| 4. Evaluate | [`evaluation_scripts/brain_lm/`](evaluation_scripts/brain_lm/) | charts + metrics |

**Evolution config** (`exa_vit_mae_evolved_config.ini` or the notebook's config cell): windowing
(`window_length`, `time_patch_size`, `mask_ratio`), model width (`d_model`, `num_heads`, `d_ff`),
the evolvable `node_types`, search size (`population_size`, `num_generations`), the per-genome
budget, and the PDH / downstream-probe toggles. Per-genome training cost is fixed by the budget and
**independent of dataset size** — evolution trains each genome on a small random slice and evolves
many; raise `num_generations` to exploit more data.

**Checkpoint/resume** ([`evolution/checkpoint.py`](evolution/checkpoint.py)) persists the
population, global innovation counter, generation index, and RNG state, so a timed-out Kaggle
session resumes deterministically. Enable Kaggle **Persistence → Files only** for this to work.

Evaluation needs three artifacts kept together so held-out subjects and normalization match
training: `final_model.pkl`, `subject_split.json`, `norm_stats.npz`.

---

## Repository layout

```
genomes/
  genome.py                            base graph genome
  vision_transformer_block_genome.py   the evolvable block-graph ViT-MAE genome
  nodes/  block_{input,output}_node, attention/simple/sequence_lstm/temporal_lstm block nodes
  edges/  edge, block_edge
  transformer_model/  VisionTransformerMAE + attention/encoder building blocks
evolution/
  vision_transformer_block_{node,edge}_generator, ..._reproduction_selector
  parallel_training   multi-GPU genome training      progressive_hurdles   adaptive compute (PDH)
  downstream_probe    clinical-probe fitness (Phase 3) checkpoint / fitness_history
reproduction/         EXAMM mutation & crossover operators (add/split/merge/enable/disable, clone)
weight_generators/    Lamarckian block weight init (+ base interface)
population/            steady-state single-population strategy
innovation/           global innovation-number generator
time_series/          HCPWindowDataset (training) + FMRIPatchDataset (small local experiments)
evaluation_scripts/
  train_final_model.py           long-train the evolved winner
  proxy_fidelity_experiment.py   cheap-vs-full search-fidelity diagnostic
  brain_lm/                      BrainLM-parity evaluation + behavioral analysis (see its README)
scripts/sync_hcp_from_s3.py      HCP corpus S3 -> local sync
docs/improvement_plan.md         research roadmap + pre-registered evaluation gates
exa_vit_mae_evolved.py           local (non-Kaggle) evolution entry point
*_kaggle.ipynb                   pretrain / evolve / train-final notebooks
```

---

## Development

Continuous integration ([`.github/workflows/`](.github/workflows/)) runs `flake8` formatting checks,
`pytest`, and a Sphinx docs build. Style config is in [`.flake8`](.flake8).

This is a research codebase; the [`docs/improvement_plan.md`](docs/improvement_plan.md) roadmap
tracks the ongoing search-improvement phases (PDH → multi-objective → downstream-aware evolution)
against fixed, pre-registered baselines and gates.
