import numpy as np
import pandas as pd
import torch

from datetime import timedelta
from torch.utils.data import Dataset


class DayAheadDataset(Dataset):
    """
    Day-ahead dataset with:
      - Historical information strictly before D at 11:00 Paris time
      - Known future features for D+1
      - 24 French local clock-hour target slots
      - DST-aware validity masks

    DataFrame index must be hourly, continuous and UTC.
    """

    def __init__(
        self,
        data,
        start_date,
        end_date,
        history_cols,
        future_cols,
        input_len=336,
        target_col="load_mw",
        valid_target_index=None,
    ):
        self.data = data.sort_index()
        self.history_cols = list(history_cols)
        self.future_cols = list(future_cols)
        self.input_len = input_len
        self.target_col = target_col

        assert str(self.data.index.tz) == "UTC"
        assert self.data.index.is_unique
        assert self.data.index.is_monotonic_increasing

        # Optional original-observation validity mask
        self.valid_target_index = (
            None if valid_target_index is None
            else pd.DatetimeIndex(valid_target_index)
        )

        self.days = pd.date_range(
            start_date, end_date, freq="D"
        )

        self.samples = []

        for day in self.days:
            sample = self._build_sample(day)

            if sample is not None:
                self.samples.append(sample)

        print(
            f"DayAheadDataset: {len(self.samples)} days "
            f"({start_date} to {end_date})"
        )

    def _build_sample(self, day):

        target_date = day.date()

        # Origin: D at 11:00, Paris local time
        origin_date = target_date - timedelta(days=1)

        origin_fr = (
            pd.Timestamp(origin_date)
            .tz_localize("Europe/Paris")
            + pd.Timedelta(hours=11)
        )

        origin_utc = origin_fr.tz_convert("UTC")

        # Historical window ends at the last available hour
        history_index = pd.date_range(
            end=origin_utc - pd.Timedelta(hours=1),
            periods=self.input_len,
            freq="h",
            tz="UTC"
        )

        # Target day, selected in French local time
        local_index = self.data.index.tz_convert(
            "Europe/Paris"
        )

        target_index = self.data.index[
            local_index.date == target_date
        ]

        if len(target_index) not in (23, 24, 25):
            return None

        # Verify complete historical observations
        if not history_index.isin(self.data.index).all():
            return None

        history = self.data.loc[
            history_index, self.history_cols
        ]

        if not np.isfinite(history.to_numpy(dtype=float)).all():
            return None

        # Historical features must strictly precede the origin
        assert history_index.max() < origin_utc

        # Prepare fixed 24-slot output arrays
        n_future = len(self.future_cols)

        x_future = np.zeros((24, n_future), dtype=np.float32)
        y = np.zeros(24, dtype=np.float32)
        mask = np.zeros(24, dtype=np.float32)

        target_utc = [pd.NaT] * 24

        # Keep the second repeated 02:00 during autumn DST
        target_hours = target_index.tz_convert(
            "Europe/Paris"
        ).hour

        duplicates = pd.Index(target_hours).duplicated(
            keep="last"
        )

        for timestamp, hour, duplicate in zip(
            target_index, target_hours, duplicates
        ):

            if duplicate:
                continue

            target_utc[hour] = timestamp

            row = self.data.loc[timestamp]

            features = row[self.future_cols].to_numpy(
                dtype=np.float32
            )

            actual = row[self.target_col]

            # Missing future features invalidate the slot
            if not np.isfinite(features).all():
                continue

            x_future[hour] = features

            if not np.isfinite(actual):
                continue

            # Exclude originally invalid target observations
            if (
                self.valid_target_index is not None
                and timestamp not in self.valid_target_index
            ):
                continue

            y[hour] = float(actual)
            mask[hour] = 1.0

        if mask.sum() == 0:
            return None

        x_history = history.to_numpy(dtype=np.float32)

        return {
            "x_history": x_history,
            "x_future": x_future,
            "y": y,
            "mask": mask,
            "origin_utc": origin_utc,
            "target_utc": target_utc,
            "day": str(target_date)
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        return {
            "x_history": torch.tensor(sample["x_history"]),
            "x_future": torch.tensor(sample["x_future"]),
            "y": torch.tensor(sample["y"]),
            "mask": torch.tensor(sample["mask"])
        }