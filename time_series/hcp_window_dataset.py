from __future__ import annotations

import json
import os
import random
import zipfile
from collections import OrderedDict

import numpy as np
import torch

from loguru import logger
from numpy.lib import format as npformat

from time_series.fmri_patch_dataset import load_atlas_coordinates


def recording_type(recording_name: str) -> str:
    """Classifies a recording as 'rest' or 'task' from its filename, matching the HCP naming
    convention used in get_cls.py (resting-state recordings start with 'rfMRI_REST', task
    recordings with 'tfMRI_')."""
    return "rest" if recording_name.startswith("rfMRI_REST") else "task"


def read_npz_data_length(path: str, key: str = "data") -> int:
    """Returns the number of timepoints (shape[0]) of a recording's 'data' array by reading ONLY
    the .npy header inside the .npz zip, without decompressing the array -- ~16x faster than
    np.load(...)[key].shape, so pre-scanning all ~19k recordings' lengths at startup is cheap.
    Falls back to a full load if the fast header read fails for any reason."""
    try:
        with zipfile.ZipFile(path) as zip_file:
            with zip_file.open(key + ".npy") as entry:
                version = npformat.read_magic(entry)
                if version == (1, 0):
                    shape, _fortran, _dtype = npformat.read_array_header_1_0(entry)
                elif version == (2, 0):
                    shape, _fortran, _dtype = npformat.read_array_header_2_0(entry)
                else:
                    raise ValueError(f"unsupported npy version {version}")
        return int(shape[0])
    except Exception:
        return int(np.load(path)[key].shape[0])


class HCPWindowDataset:
    """Subject-organized fMRI dataset for evolving/training the foundation model on the full HCP
    corpus (one subdirectory per subject, holding that subject's ~18 .npz recordings, the same
    layout as the S3 bucket in get_cls.py). Designed for the Kaggle workflow: the corpus is synced
    once to a local read-only mount (/kaggle/input/...), and recordings are lazily loaded from
    local SSD with a small LRU cache rather than held entirely in RAM.

    Key properties:
      - Subject-level train/validation/test split (default 70/10/20): ALL of a subject's
        recordings go entirely into one split, so a subject's data never straddles the split
        boundary (which would leak subject identity into held-out metrics). The split is
        deterministic from (sorted subject ids, seed, ratios) and can be persisted.
      - Per-parcel z-score normalization with statistics FROZEN from the training split (estimated
        from a sample of training recordings), reused for validation/test -- no leakage.
      - Random-recording-then-random-window sampling, so every recording gets equal expected
        representation regardless of length ("every recording has a say"), provided the window
        fits (choose window_length <= the shortest recording, e.g. <= EMOTION's ~176 timepoints).
      - Recording-type tags (rest/task) so a combined model can be trained on everything while
        downstream embeddings are extracted from a chosen recording type.

    Interface parity with FMRIPatchDataset (num_parcels, parcel_coordinates, sample_batch) lets it
    drop into the same genome train()/evaluation code.
    """

    def __init__(
        self,
        root_dir: str,
        atlas_coordinates_filename: str,
        window_length: int,
        split_ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
        split_seed: int = 42,
        cache_size: int = 256,
        normalization_sample: int = 400,
        stats_path: str | None = None,
        split_path: str | None = None,
        length_index_path: str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.root_dir = root_dir
        self.window_length = window_length
        self.cache_size = cache_size
        self.dtype = dtype
        self._length_index_path = length_index_path

        self.parcel_coordinates = load_atlas_coordinates(atlas_coordinates_filename)
        self.num_parcels = self.parcel_coordinates.shape[0]

        # {subject_id: {recording_name: absolute_path}}
        self.subject_recordings = self._discover(root_dir)
        if not self.subject_recordings:
            raise ValueError(f"no subject recordings found under '{root_dir}'")
        logger.info(
            f"discovered {len(self.subject_recordings)} subjects, "
            f"{sum(len(r) for r in self.subject_recordings.values())} recordings under '{root_dir}'"
        )

        # {f'{subject}/{recording}': num_timepoints}, cached to length_index_path if given
        self.lengths = self._build_length_index()

        self.splits = self._make_or_load_split(split_ratios, split_seed, split_path)
        # {split: [(subject_id, recording_name, path, type)]}, filtered to recordings >= window_length
        self.split_recordings = self._index_recordings()

        # LRU cache must exist before _load_or_compute_stats, which loads recordings via _load
        self._cache: OrderedDict[tuple[str, str], torch.Tensor] = OrderedDict()

        self.parcel_mean, self.parcel_std = self._load_or_compute_stats(stats_path, normalization_sample)

    # ------------------------------------------------------------------ discovery / split

    @staticmethod
    def _discover(root_dir: str) -> dict[str, dict[str, str]]:
        subject_recordings: dict[str, dict[str, str]] = {}
        for subject_id in sorted(os.listdir(root_dir)):
            subject_dir = os.path.join(root_dir, subject_id)
            if not os.path.isdir(subject_dir):
                continue
            for filename in sorted(os.listdir(subject_dir)):
                if not filename.endswith(".npz"):
                    continue
                recording_name = os.path.splitext(filename)[0]
                subject_recordings.setdefault(subject_id, {})[recording_name] = os.path.join(
                    subject_dir, filename
                )
        return subject_recordings

    def _make_or_load_split(
        self, split_ratios: tuple[float, float, float], split_seed: int, split_path: str | None
    ) -> dict[str, list[str]]:
        if split_path is not None and os.path.exists(split_path):
            with open(split_path) as split_file:
                splits = json.load(split_file)
            logger.info(f"loaded subject split from '{split_path}'")
            return splits

        assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios must sum to 1.0"
        subjects = sorted(self.subject_recordings.keys())
        random.Random(split_seed).shuffle(subjects)

        n = len(subjects)
        n_train = int(round(n * split_ratios[0]))
        n_val = int(round(n * split_ratios[1]))
        splits = {
            "train": subjects[:n_train],
            "val": subjects[n_train:n_train + n_val],
            "test": subjects[n_train + n_val:],
        }
        logger.info(
            f"subject split -- train: {len(splits['train'])}, val: {len(splits['val'])}, "
            f"test: {len(splits['test'])}"
        )

        if split_path is not None:
            with open(split_path, "w") as split_file:
                json.dump(splits, split_file, indent=2)
            logger.info(f"saved subject split to '{split_path}'")

        return splits

    def _build_length_index(self) -> dict[str, int]:
        """Maps '{subject}/{recording}' -> num_timepoints for every recording, reading only .npy
        headers (fast). Cached to length_index_path (keyed by relative subject/recording, so the
        cache survives a changed absolute mount path across Kaggle sessions)."""
        if self._length_index_path is not None and os.path.exists(self._length_index_path):
            with open(self._length_index_path) as index_file:
                lengths = json.load(index_file)
            logger.info(f"loaded recording length index from '{self._length_index_path}'")
            return lengths

        pairs = [
            (subject_id, recording_name, path)
            for subject_id, recordings in self.subject_recordings.items()
            for recording_name, path in recordings.items()
        ]
        logger.info(f"scanning lengths of {len(pairs)} recordings (header-only)...")
        lengths = {
            f"{subject_id}/{recording_name}": read_npz_data_length(path)
            for subject_id, recording_name, path in pairs
        }

        if self._length_index_path is not None:
            with open(self._length_index_path, "w") as index_file:
                json.dump(lengths, index_file)
            logger.info(f"saved recording length index to '{self._length_index_path}'")
        return lengths

    def _index_recordings(self) -> dict[str, list[tuple[str, str, str, str]]]:
        """Builds the per-split sampling pool, EXCLUDING recordings shorter than window_length
        (they cannot form even one window). The exclusion is logged so it is explicit rather than
        a silent sample-time skip -- a small fraction of HCP recordings are truncated/incomplete
        acquisitions (some tasks run as short as ~35 timepoints)."""
        split_recordings: dict[str, list[tuple[str, str, str, str]]] = {"train": [], "val": [], "test": []}
        dropped = {"train": 0, "val": 0, "test": 0}
        for split, subjects in self.splits.items():
            for subject_id in subjects:
                for recording_name, path in self.subject_recordings.get(subject_id, {}).items():
                    if self.lengths[f"{subject_id}/{recording_name}"] < self.window_length:
                        dropped[split] += 1
                        continue
                    split_recordings[split].append(
                        (subject_id, recording_name, path, recording_type(recording_name))
                    )

        kept = sum(len(v) for v in split_recordings.values())
        total_dropped = sum(dropped.values())
        if total_dropped:
            logger.warning(
                f"excluded {total_dropped} recordings shorter than window_length={self.window_length} "
                f"(per split: {dropped}); {kept} usable recordings remain"
            )
        for split in ("train", "val", "test"):
            if not split_recordings[split]:
                logger.warning(f"split '{split}' has NO usable recordings at window_length={self.window_length}")
        return split_recordings

    # ------------------------------------------------------------------ loading / caching

    def _load(self, subject_id: str, recording_name: str, path: str) -> torch.Tensor:
        """Lazily loads one recording as a (num_parcels, num_timepoints) float tensor, with LRU
        caching so hot recordings aren't re-read from disk each batch."""
        key = (subject_id, recording_name)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        npz = np.load(path)
        data = npz["data"]  # (num_timepoints, num_parcels)
        tensor = torch.tensor(data, dtype=self.dtype).transpose(0, 1).contiguous()  # (parcels, time)

        self._cache[key] = tensor
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return tensor

    # ------------------------------------------------------------------ normalization

    def _load_or_compute_stats(
        self, stats_path: str | None, normalization_sample: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stats_path is not None and os.path.exists(stats_path):
            stats = np.load(stats_path)
            logger.info(f"loaded normalization stats from '{stats_path}'")
            # force float32: the stats are computed/saved in float64, and a float64 mean/std would
            # upcast every normalized batch to float64, which then fails under autocast (float64 is
            # not downcast to the model's half/float32 weights).
            mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1)
            std = torch.tensor(stats["std"], dtype=torch.float32).view(-1, 1)
            return mean, std

        train_recordings = self.split_recordings["train"]
        sample = train_recordings
        if len(train_recordings) > normalization_sample:
            sample = random.Random(0).sample(train_recordings, normalization_sample)
        logger.info(f"computing per-parcel normalization stats from {len(sample)} training recordings...")

        # accumulate per-parcel sum and sum of squares over time across the sampled recordings
        total = torch.zeros(self.num_parcels, dtype=torch.float64)
        total_sq = torch.zeros(self.num_parcels, dtype=torch.float64)
        count = 0
        for subject_id, recording_name, path, _type in sample:
            data = self._load(subject_id, recording_name, path).to(torch.float64)  # (parcels, time)
            total += data.sum(dim=1)
            total_sq += (data ** 2).sum(dim=1)
            count += data.shape[1]

        mean = (total / count)
        var = (total_sq / count) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-12)).clamp(min=1e-6)

        mean = mean.view(-1, 1).to(torch.float32)
        std = std.view(-1, 1).to(torch.float32)

        if stats_path is not None:
            np.savez(stats_path, mean=mean.numpy(), std=std.numpy())
            logger.info(f"saved normalization stats to '{stats_path}'")

        return mean, std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Per-parcel z-score. Accepts (parcels, time) or (batch, parcels, time). Always returns
        float32 (defensively cast so a stray float64 stat can never upcast the batch and break
        mixed-precision training)."""
        return ((x.to(torch.float32) - self.parcel_mean) / self.parcel_std).to(torch.float32)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.parcel_std + self.parcel_mean

    # ------------------------------------------------------------------ sampling

    def sample_batch(self, batch_size: int, window_length: int, split: str = "train") -> torch.Tensor:
        """Samples a batch of random, normalized (num_parcels, window_length) windows from the
        given split -- a random recording then a random start offset per batch element. Returns a
        CPU tensor of shape (batch_size, num_parcels, window_length); the caller moves it to the
        training device."""
        recordings = self.split_recordings[split]
        if not recordings:
            raise ValueError(f"split '{split}' has no recordings")

        windows = []
        while len(windows) < batch_size:
            subject_id, recording_name, path, _type = random.choice(recordings)
            data = self._load(subject_id, recording_name, path)  # (parcels, time)
            series_length = data.shape[1]
            if series_length < window_length:
                # should not happen if window_length <= shortest recording; skip defensively
                continue
            start = random.randint(0, series_length - window_length)
            windows.append(data[:, start:start + window_length])

        batch = torch.stack(windows, dim=0)
        return self.normalize(batch)

    def subject_windows(
        self, subject_id: str, window_length: int, types: tuple[str, ...] = ("rest",), stride: int | None = None
    ) -> torch.Tensor:
        """Tiles a subject's recordings (of the given type(s)) into non-overlapping normalized
        windows for embedding extraction (get_cls.py's create_windows, but type-filtered and
        normalized with the frozen train stats). Returns (num_windows, num_parcels, window_length),
        or an empty tensor if the subject has no usable recordings of those types."""
        stride = stride or window_length
        windows = []
        for recording_name, path in self.subject_recordings.get(subject_id, {}).items():
            if recording_type(recording_name) not in types:
                continue
            data = self._load(subject_id, recording_name, path)  # (parcels, time)
            series_length = data.shape[1]
            for start in range(0, series_length - window_length + 1, stride):
                windows.append(data[:, start:start + window_length])

        if not windows:
            return torch.empty(0, self.num_parcels, window_length)
        return self.normalize(torch.stack(windows, dim=0))
