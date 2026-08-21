from __future__ import annotations

import math

import numpy as np
import torch


class DownstreamProbe:
    """Phase 3 (see docs/improvement_plan.md): a CLINICAL-PREDICTION fitness component, so evolution
    can select architectures whose representations carry clinical signal -- not merely low
    reconstruction MSE (which is already near the data's linear ceiling, so it no longer discriminates
    architectures).

    MECHANISM. A fixed probe set of held-out subjects is chosen once and their windows cached. For a
    given trained genome we compute each probe subject's per-parcel reconstruction-error FINGERPRINT
    (the model's strongest clinical readout -- it beat the CLS embedding and the FC+ridge baseline on
    age), run an internal cross-validation to predict age (ridge R^2) and sex (logreg AUC), and turn
    that into a single `combined` score (higher = more clinical signal). evolution.parallel_training
    then sets the selection fitness to `recon_MSE - fitness_weight * combined`, while PDH keeps
    escalating on pure recon MSE (preserved as genome.recon_fitness) -- so reconstruction still gates
    training budget and only hurdle-clearing genomes pay the probe cost.

    PROBE SPLIT. Defaults to the VALIDATION subjects: they are held OUT of reconstruction training
    (train() validates on them), so their fingerprints are out-of-sample -- a more honest selection
    signal than in-sample TRAIN subjects -- while TEST stays completely untouched for the final
    honest evaluation. Labels are never used in pretraining, so there is no label leakage; this is a
    NAS selection signal, judged for real only on TEST.

    COST / THREADING. Windows are cached on CPU once at construction; score() moves them to the
    genome's device per call and never mutates shared state, so it is safe to call from
    parallel_training's per-genome worker threads. The internal sklearn CV runs on ~100 subjects x
    424 features and is cheap next to the forward passes.
    """

    def __init__(
        self,
        dataset,
        clinical_csv,
        window_length: int,
        probe_split: str = "val",
        max_subjects: int | None = None,
        windows_per_subject: int = 16,
        batch_size: int = 16,
        fitness_weight: float = 0.5,
        min_stage: int = 1,
        age_weight: float = 1.0,
        sex_weight: float = 1.0,
        n_pca: int = 30,
        seed: int = 0,
        verbose: bool = True,
    ):
        """
        Args:
            dataset: HCPWindowDataset (provides .splits and subject_windows()).
            clinical_csv: HCP_YA_subjects_info.csv (Subject, Gender, Age_in_Yrs).
            window_length: genome.window_length (windows must match the model).
            probe_split: which subject split to probe ("val" recommended; held out of training).
            max_subjects: cap the probe set (None = all labelled subjects in the split).
            windows_per_subject: even-stride windows per subject for the fingerprint.
            batch_size: forward-pass batch size during fingerprinting.
            fitness_weight: lambda in `fitness = recon_MSE - lambda * combined_score`.
            min_stage: only score genomes whose PDH stage >= this (so only hurdle-clearers pay the
                probe cost). Ignored when PDH is off (parallel_training scores all finite genomes).
            age_weight, sex_weight: weights on the two clinical terms inside `combined`.
            seed: RNG seed for the (deterministic) probe-subject selection.
        """
        import pandas as pd

        self.window_length = window_length
        self.batch_size = batch_size
        self.fitness_weight = fitness_weight
        self.min_stage = min_stage
        self.age_weight = age_weight
        self.sex_weight = sex_weight
        self.n_pca = n_pca

        clin = pd.read_csv(clinical_csv)
        clin["Subject"] = clin["Subject"].astype(str)
        age = dict(zip(clin["Subject"], clin["Age_in_Yrs"]))
        sex = dict(zip(clin["Subject"], clin["Gender"]))

        subjects = [str(s) for s in dataset.splits.get(probe_split, [])]
        np.random.default_rng(seed).shuffle(subjects)

        self.windows: list[torch.Tensor] = []
        ages: list[float] = []
        sexes: list[int] = []
        self.ids: list[str] = []
        for sid in subjects:
            if max_subjects and len(self.ids) >= max_subjects:
                break
            a, g = age.get(sid), sex.get(sid)
            if a is None or (isinstance(a, float) and math.isnan(a)) or g not in ("M", "F"):
                continue
            w = dataset.subject_windows(sid, window_length, types=("rest",))
            if w is None or len(w) == 0:
                continue
            if len(w) > windows_per_subject:
                idx = np.linspace(0, len(w) - 1, windows_per_subject).astype(int)
                w = w[idx]
            self.windows.append(w.cpu())
            ages.append(float(a))
            sexes.append(1 if g == "M" else 0)
            self.ids.append(sid)

        self.age = np.asarray(ages, dtype=float)
        self.sex = np.asarray(sexes, dtype=int)
        if len(self.ids) < 10 or self.sex.sum() == 0 or (1 - self.sex).sum() == 0:
            raise ValueError(
                f"probe set too small/degenerate ({len(self.ids)} subjects, "
                f"{int(self.sex.sum())} M) -- pick a larger probe_split or check labels"
            )
        if verbose:
            print(f"[probe] cached {len(self.ids)} {probe_split} subjects "
                  f"(age {self.age.min():.0f}-{self.age.max():.0f}, "
                  f"{int(self.sex.sum())} M / {int((1 - self.sex).sum())} F), "
                  f"fitness_weight={fitness_weight}, min_stage={min_stage}")

    @torch.no_grad()
    def fingerprints(self, genome, device) -> np.ndarray:
        """(n_subjects, num_parcels) per-parcel masked-reconstruction MSE for the probe subjects.

        The per-parcel reduction is fully vectorized ON DEVICE (only 424-vectors cross to CPU); the
        earlier per-parcel Python loop dominated runtime and made the per-genome probe cost far too
        high for use inside the evolution loop.
        """
        for module in genome._iter_modules():
            module.eval()
        n_parcels = genome.num_parcels
        X = np.full((len(self.ids), n_parcels), np.nan)
        for i, w in enumerate(self.windows):
            se = np.zeros(n_parcels)
            cnt = np.zeros(n_parcels)
            for s in range(0, len(w), self.batch_size):
                genome.reset()
                chunk = w[s:s + self.batch_size].to(device)
                loss, pred, mask, patches = genome.forward(chunk)
                if not torch.isfinite(loss):
                    continue
                # pred/patches: (B, parcels, window); mask: (B, parcels) with 1 = masked
                sq = ((pred.float() - patches.float()) ** 2).sum(dim=2)   # (B, parcels)
                msk = mask.float()
                se += (sq * msk).sum(dim=0).cpu().numpy()                 # (parcels,)
                cnt += (msk.sum(dim=0) * pred.shape[2]).cpu().numpy()     # timepoints per masked patch
            X[i] = np.where(cnt > 0, se / np.maximum(cnt, 1), np.nan)
        for module in genome._iter_modules():
            module.train()
        col_mean = np.nan_to_num(np.nanmean(X, axis=0))
        nan = np.where(np.isnan(X))
        X[nan] = np.take(col_mean, nan[1])
        return X

    def score(self, genome, device) -> dict:
        """Returns {'sex_auc', 'age_r2', 'combined'} for the genome on the probe set. `combined` is
        higher-is-better and >= 0 (chance-level readouts contribute 0)."""
        from sklearn.linear_model import RidgeCV, LogisticRegression
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.metrics import r2_score, roc_auc_score

        X = self.fingerprints(genome, device)
        # 424 per-parcel features >> ~100 probe subjects, so reduce dimensionality first: PCA both
        # stabilizes and (empirically) improves the small-n readout. Cap components below the fold's
        # sample count.
        n_pca = max(2, min(self.n_pca, X.shape[0] // 3, X.shape[1]))

        def prep(estimator):
            return make_pipeline(StandardScaler(), PCA(n_pca, random_state=0), estimator)

        try:
            prob = cross_val_predict(
                prep(LogisticRegression(max_iter=500, C=1.0)), X, self.sex,
                cv=StratifiedKFold(5, shuffle=True, random_state=0),
                method="predict_proba")[:, 1]
            sex_auc = float(roc_auc_score(self.sex, prob))
        except Exception:
            sex_auc = 0.5

        try:
            yhat = cross_val_predict(prep(RidgeCV(alphas=np.logspace(-1, 3, 9))), X, self.age,
                                     cv=KFold(5, shuffle=True, random_state=0))
            age_r2 = float(r2_score(self.age, yhat))
        except Exception:
            age_r2 = 0.0

        combined = (self.sex_weight * max(0.0, 2.0 * (sex_auc - 0.5))
                    + self.age_weight * max(0.0, age_r2))
        return {"sex_auc": sex_auc, "age_r2": age_r2, "combined": combined}
