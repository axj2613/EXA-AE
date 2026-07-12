from __future__ import annotations

import numpy as np
import torch

from loguru import logger


def load_atlas_coordinates(filename: str) -> torch.Tensor:
    """Loads parcel 3D atlas coordinates from a whitespace-separated coordinates file (columns:
    1-based parcel index, x, y, z in mm -- the format used by A424_Coordinates.dat), and returns
    them sorted by parcel index as a (num_parcels, 3) float tensor.

    Note this assumes the file's 1-based parcel index i corresponds to roi_names[i - 1] / column
    i - 1 of the matching fMRI .npz 'data' array (both are the same "A424" atlas ordering);
    FMRIPatchDataset validates the row count matches num_parcels but cannot verify the ordering
    itself lines up beyond that.
    """
    indices = []
    coordinates = []
    with open(filename, "r") as coordinates_file:
        for line in coordinates_file:
            line = line.strip()
            if not line:
                continue
            parcel_index, x, y, z = line.split()
            indices.append(int(parcel_index))
            coordinates.append([float(x), float(y), float(z)])

    order = sorted(range(len(indices)), key=lambda i: indices[i])
    sorted_coordinates = [coordinates[i] for i in order]
    return torch.tensor(sorted_coordinates, dtype=torch.float32)


class FMRIPatchDataset:
    """Loads one or more fMRI .npz recordings (same 'data'/'roi_names' schema as
    TimeSeries.create_from_fmri_npz) and samples random fixed-length (parcel x time) windows
    for masked-autoencoder training, following the random-window sampling scheme used by BrainLM.

    Unlike TimeSeries (which stores each parcel as an independent named 1D series for the scalar,
    per-timestep EXAMM node graph), this class keeps each recording as a single (parcels x time)
    tensor so windows can be sliced and batched efficiently for the vision-transformer MAE.
    """

    def __init__(self, npz_filenames: list[str], atlas_coordinates_filename: str | None = None, normalize: bool = True):
        if len(npz_filenames) == 0:
            raise ValueError("FMRIPatchDataset requires at least one .npz filename")

        self.recordings: list[torch.Tensor] = []  # each: (num_parcels, series_length)
        self.roi_names: list[str] | None = None

        for filename in npz_filenames:
            npz = np.load(filename, allow_pickle=True)
            if "data" not in npz:
                raise ValueError(f"npz file '{filename}' does not contain a 'data' array")

            data = npz["data"]
            if data.ndim != 2:
                raise ValueError(
                    f"npz file '{filename}' 'data' array must be 2D (time x parcels), got shape {data.shape}"
                )

            roi_names = None
            if "roi_names" in npz:
                roi_names = [
                    name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name)
                    for name in npz["roi_names"]
                ]

            if self.roi_names is None:
                self.roi_names = roi_names
            elif roi_names is not None and roi_names != self.roi_names:
                raise ValueError(
                    f"npz file '{filename}' has roi_names that do not match the first loaded "
                    "recording; all recordings must share the same parcel ordering."
                )

            # (time, parcels) -> (parcels, time)
            recording = torch.tensor(data, dtype=torch.float32).transpose(0, 1).contiguous()
            self.recordings.append(recording)

            logger.info(
                f"loaded fMRI recording '{filename}': {recording.shape[0]} parcels x "
                f"{recording.shape[1]} timesteps"
            )

        self.num_parcels = self.recordings[0].shape[0]
        for recording in self.recordings:
            if recording.shape[0] != self.num_parcels:
                raise ValueError("all fMRI recordings must have the same number of parcels")

        self.min_series_length = min(recording.shape[1] for recording in self.recordings)

        self.parcel_coordinates: torch.Tensor | None = None
        if atlas_coordinates_filename is not None:
            self.parcel_coordinates = load_atlas_coordinates(atlas_coordinates_filename)
            if self.parcel_coordinates.shape[0] != self.num_parcels:
                raise ValueError(
                    f"atlas coordinates file '{atlas_coordinates_filename}' has "
                    f"{self.parcel_coordinates.shape[0]} parcels, but the loaded fMRI data has "
                    f"{self.num_parcels} parcels"
                )
            logger.info(f"loaded atlas coordinates from '{atlas_coordinates_filename}'")

        # per-parcel z-score normalization computed across all loaded recordings, so the model
        # is not dominated by parcels/subjects with naturally larger BOLD signal amplitude.
        self.normalize_enabled = normalize
        if normalize:
            all_values = torch.cat(self.recordings, dim=1)  # (num_parcels, total_timesteps)
            self.parcel_mean = all_values.mean(dim=1, keepdim=True)
            self.parcel_std = all_values.std(dim=1, keepdim=True).clamp(min=1e-6)
        else:
            self.parcel_mean = torch.zeros(self.num_parcels, 1)
            self.parcel_std = torch.ones(self.num_parcels, 1)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Applies per-parcel z-score normalization. Accepts (parcels, time) or (batch, parcels, time)."""
        return (x - self.parcel_mean) / self.parcel_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Inverts `normalize`, for reporting/plotting reconstructions in the original data scale."""
        return x * self.parcel_std + self.parcel_mean

    def sample_batch(self, batch_size: int, window_length: int, split: str = "train") -> torch.Tensor:
        """Samples a batch of random, independently-positioned, normalized windows across the
        loaded recordings (a random recording and a random start offset per batch element).

        Args:
            batch_size: number of windows to sample.
            window_length: number of contiguous time steps per window.
            split: accepted for interface parity with HCPWindowDataset (which does true
                subject-level splits) and ignored here -- this loader holds a single small pool of
                recordings with no split, so every call samples from the same pool.

        Returns:
            A normalized tensor of shape (batch_size, num_parcels, window_length).
        """
        if window_length > self.min_series_length:
            raise ValueError(
                f"window_length ({window_length}) is longer than the shortest loaded recording "
                f"({self.min_series_length} timesteps)"
            )

        windows = []
        for _ in range(batch_size):
            recording_idx = torch.randint(len(self.recordings), (1,)).item()
            recording = self.recordings[recording_idx]
            series_length = recording.shape[1]
            start = torch.randint(series_length - window_length + 1, (1,)).item()
            windows.append(recording[:, start:start + window_length])

        batch = torch.stack(windows, dim=0)
        return self.normalize(batch) if self.normalize_enabled else batch
