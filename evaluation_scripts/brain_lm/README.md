# BrainLM-parity evaluation & behavioral analysis

Evaluation and downstream-analysis pipeline for a trained `VisionTransformerBlockGenome`
(`final_model.pkl`). All scripts write charts/caches to `eval_out/` (git-ignored).

Three inputs recur and must come from the **same training run** so held-out subjects and
normalization match: the model pickle, `subject_split.json`, and `norm_stats.npz`. Analyses of
subject behavior additionally need the HCP metadata CSVs (`datasets/hcp/HCP_YA_subjects_*.csv`,
obtained separately under the HCP Data Use Terms).

## The two readouts

- **CLS embedding** — the encoder's summary token; cheap, transferable, but a weak clinical readout
  here (the token was trained only to aid reconstruction).
- **Reconstruction-error fingerprint** — each subject's per-parcel masked-reconstruction MSE
  (424-d). A normative "where does this brain reconstruct worse than expected" signature; the
  model's **strongest** clinical readout, and the basis of the behavioral analyses below.

## Pipeline

**1 · Core evaluation** (needs the model + HCP data)

```bash
# BrainLM-parity: reconstruction R²/MAE, per-parcel & anatomical maps, CLS→clinical, attention
python evaluate_final_model.py  <model.pkl> <hcp_root> <atlas_coords.dat> \
       subject_split.json norm_stats.npz HCP_YA_subjects_info.csv  [--device cpu]

# Reconstruction-error fingerprint → age/sex readout (the model's best clinical signal)
python fingerprint_readout.py   <model.pkl> <hcp_root> <atlas_coords.dat> \
       subject_split.json norm_stats.npz HCP_YA_subjects_info.csv
```

**2 · All-subject feature cache** (one pass; powers the higher-N behavioral analyses)

```bash
# CLS embedding + fingerprint for every subject (train+val+test) ->
#   eval_out/all_subject_embeddings.npz , eval_out/all_readout_features.npz
python compute_all_features.py  <model.pkl> <hcp_root> <atlas_coords.dat> \
       subject_split.json norm_stats.npz
```

**3 · Behavioral analyses** (fast; read the cached features from step 2)

```bash
python personality_readout.py        HCP_YA_subjects_personality.csv \
       --cls_cache eval_out/all_subject_embeddings.npz \
       --fingerprint_cache eval_out/all_readout_features.npz --tag _pooled
python personality_classification.py HCP_YA_subjects_personality.csv subject_split.json
python behavioral_plots.py           # screens 388 behaviors; FDR + demographic-confound control
python behavioral_presentation.py    # per-construct scatters, g-factor, positive-negative mode
python behavioral_maps_all.py        # reconstruction-corrected brain maps + Yeo-7 network charts
python yeo_networks.py               # (fetches the Yeo-7 atlas via nilearn on first run)
```

`corrected_maps.py` and `yeo_networks.py` are also imported as helpers by `behavioral_maps_all.py`;
run them standalone only to (re)generate the general-cognition map in isolation.

## Method notes

- **Held-out protocol** — readouts are fit on the train split and evaluated on the test split; the
  388-variable behavioral screen applies Benjamini–Hochberg FDR correction.
- **Confound controls** — every behavioral hit is checked against an age+sex baseline (to drop
  demographic proxies) and residualized on the overall reconstruction-quality map (to separate
  genuine spatial signal from a data-quality/motion artifact).
- **Representation-probe caveat** — the all-subject caches include train subjects the model saw
  (self-supervised) during pretraining; behavioral labels were never used in training, so there is
  no label leakage, but pooled-subject numbers are a representation probe, reported separately from
  the clean held-out-test numbers.
