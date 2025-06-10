import __fix_relative_imports  # noqa: F401
import numpy as np
import pandas as pd
from typing import Generator
import torch
from mscEidalVesetrud.data_preprocessing.weather_preprocessing import (
    WindSpeedScaler,
)


class TimestampDataset(torch.utils.data.Dataset):
    def __init__(
        self, x: np.ndarray, y: np.ndarray, timestamps: pd.DatetimeIndex, device="cpu"
    ):
        assert len(y.shape) == len(timestamps.shape) == 1
        assert x.shape[0] == y.shape[0] == timestamps.shape[0]
        assert x.dtype == y.dtype == np.float32

        # Convert data to torch tensors
        self.x = torch.from_numpy(x).to(device=device)
        self.y = torch.from_numpy(y).to(device=device)
        self.timestamps = timestamps
        self.x_time = torch.from_numpy(
            TimestampDataset.calc_time_features(self.timestamps)
        ).to(device=device)
        # OBS: Since self.timestamps (and self.x_time) is missing some dates due to incomplete or invalid data
        # it is only suited to describe the time at the middle time of self.x
        # See self.get_timestamps_for_x_window()

    @staticmethod
    def calc_time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
        time_features = np.zeros((timestamps.shape[0], 5), dtype=np.float32)

        hour_of_day_float = timestamps.hour.values * (1 / 24)
        hour_of_day_float_scaled = 2 * np.pi * hour_of_day_float

        # Either 365 or 366 depending on weather it is a leap year
        year_scaling = (timestamps.is_leap_year).astype(np.float32) + 365

        # A float from 0 to 2pi indicating the day of year
        day_of_year_float = 2 * np.pi * timestamps.dayofyear / year_scaling

        time_features[:, 0] = hour_of_day_float
        time_features[:, 1] = np.cos(hour_of_day_float_scaled)
        time_features[:, 2] = np.sin(hour_of_day_float_scaled)
        time_features[:, 3] = np.cos(day_of_year_float)
        time_features[:, 4] = np.sin(day_of_year_float)

        return time_features

    @staticmethod
    def calc_time_feature(timestamp: str) -> np.ndarray:
        timestamp_idx = pd.date_range(timestamp, timestamp, freq="h")
        return TimestampDataset.calc_time_features(timestamp_idx)[0]

    def get_timestamps_for_x_window(self, index):
        """Return the timestamps within the 11 hour window of self.x[index]"""

        return pd.DatetimeIndex(
            pd.date_range(
                self.timestamps[index] - pd.Timedelta(hours=5),
                self.timestamps[index] + pd.Timedelta(hours=5),
                freq="h",
            )
        )

    def __getitem__(self, index):
        return self.x[index], self.x_time[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class DataSplitIndex:
    def __init__(self, train_start: int, train_val_split: int, val_end: int) -> None:
        self.train_start = train_start
        self.train_val_split = train_val_split
        self.val_end = val_end


class CrossValidationDatasets:
    """
    A class for generating cross-validation splits from time series data.

    This class takes weather, production, and timestamp data and creates multiple
    train/validation splits using a sliding window approach.  It accounts for
    NaN values in the data and ensures minimum sizes for the training and
    validation sets.

    Attributes:
        x (np.ndarray): The input weather data, a NumPy array of shape (n_samples, 13, 13, 3, 3).
        y (np.ndarray): The target production data, a NumPy array of shape (n_samples,).
        timestamps (pd.DatetimeIndex): The timestamps corresponding to the data.
        min_train_size (int): The minimum number of *non-NaN* samples in the training set.
        min_val_size (int): The minimum number of *non-NaN* samples in the validation set.

        splits (List[CrossValDataset]): A list of CrossValDataset objects, each representing a train/validation split.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        timestamps: pd.DatetimeIndex,
        train_size=18 * 30 * 24 - 24,
        val_size=30 * 24,
        verbose=True,
    ) -> None:
        assert len(y.shape) == len(timestamps.shape) == 1
        assert x.shape[0] == y.shape[0] == timestamps.shape[0]
        assert x.dtype == y.dtype == np.float32
        assert not np.isnan(x).any()
        assert not np.isnan(y).any()

        self.x = x
        self.y = y
        self.timestamps = timestamps
        self.train_size = train_size
        self.val_size = val_size
        del x, y, timestamps, train_size, val_size

        self.split_indices: list[DataSplitIndex] = []
        self.__split_dataset()
        self.wind_speed_scalers: list[WindSpeedScaler] = []
        self.n_splits = len(self.split_indices)

        for split in self.split_indices:
            self.wind_speed_scalers.append(
                WindSpeedScaler(self.x[split.train_start : split.train_val_split])
            )

        self.wind_speed_scaler_full = WindSpeedScaler(
            self.x[0 : self.split_indices[-1].train_val_split]
        )

        # Double check the wind speed scalers end up with approximatly the same scaling factors
        # avgs = np.concatenate(
        #     [
        #         self.wind_speed_scalers[i].mean[np.newaxis, :, :, :]
        #         for i in range(self.n_splits)
        #     ]
        # )
        # stds = np.concatenate(
        #     [
        #         self.wind_speed_scalers[i].std[np.newaxis, :, :, :]
        #         for i in range(self.n_splits)
        #     ]
        # )

        # print(avgs.shape)
        # print(avgs[0])
        # print(np.mean(avgs, axis=0))
        # print(np.std(avgs, axis=0))

        if verbose:
            self.print_split_info()

    def print_split_info(self):
        print(f"\nTotal size before split is: {self.x.shape[0]}")
        print(
            f"Using training size without NaN: {self.train_size} and validation size without NaN: {self.val_size}"
        )
        print(
            f"There was space for {self.n_splits} splits when accounting for NaN values"
        )
        print(
            f"There are {(self.x.shape[0] - self.unused_data_start_idx)} datapoints left unused at the end, about {(self.x.shape[0] - self.unused_data_start_idx) / 24 :.1f} days"
        )
        for split in self.split_indices:
            print(
                f"    train start: {self.timestamps[split.train_start]} {split.train_start}"
            )
            print(f"    train length: {split.train_val_split - split.train_start}")
            print(
                f"    train val split: {self.timestamps[split.train_val_split]} {split.train_val_split}"
            )
            print(f"    val length: {split.val_end - split.train_val_split}")
            print(f"    val end: {self.timestamps[split.val_end-1]} {split.val_end-1}")
            print()

    def __find_end_time_day_change(self, time_index: int) -> int | None:
        if time_index >= self.timestamps.shape[0]:
            return None

        while True:
            time_index += 1
            if time_index >= self.timestamps.shape[0]:
                # Relies on the split between the training and test set to be on a day change
                # So the sets do not overlap
                return time_index

            if (
                self.timestamps[time_index - 1].date()
                != self.timestamps[time_index].date()
            ):
                return time_index

    def __split_dataset(self):
        """
        Splits the dataset into multiple train/validation sets using a sliding window approach,
        ensuring the minimum sizes for both training and validation sets while accounting for NaN values.

        This method populates the `self.splits` list with `CrossValDataset` objects, each containing
        a training and a validation `Dataset`.  It uses the `_find_actual_size` helper function
        to determine the actual sizes of the training and validation sets, considering the presence
        of NaN values and the specified minimum sizes.

        The splitting process starts from the end of the data and works backwards.  The remaining
        data before the first training set is stored as unused data in `self.unused_data`.
        """

        # First training and validation set:
        train_start = 0
        train_val_split = self.__find_end_time_day_change(self.train_size)
        assert train_val_split is not None
        val_end = self.__find_end_time_day_change(train_val_split + self.val_size)

        # If None, train_size and val_size is larger than the available data
        assert val_end is not None

        while val_end is not None:

            self.split_indices.append(
                DataSplitIndex(train_start, train_val_split, val_end)
            )

            train_start = val_end - self.train_size
            train_val_split = val_end
            val_end = self.__find_end_time_day_change(train_val_split + self.val_size)

        # When the loop exits, it means val_end is None,
        # and the previous val_end is stored in train_val_split
        # containing the index where unused data begins
        self.unused_data_start_idx = train_val_split

    def get_train_val(self, index: int, scale=True, device="cpu"):
        split = self.split_indices[index]

        if scale:
            train_data = TimestampDataset(
                self.wind_speed_scalers[index].transform(
                    self.x[split.train_start : split.train_val_split]
                ),
                self.y[split.train_start : split.train_val_split],
                self.timestamps[split.train_start : split.train_val_split],
                device,
            )

            val_data = TimestampDataset(
                self.wind_speed_scalers[index].transform(
                    self.x[split.train_val_split : split.val_end]
                ),
                self.y[split.train_val_split : split.val_end],
                self.timestamps[split.train_val_split : split.val_end],
                device,
            )

            return train_data, val_data

        else:
            train_data = TimestampDataset(
                self.x[split.train_start : split.train_val_split],
                self.y[split.train_start : split.train_val_split],
                self.timestamps[split.train_start : split.train_val_split],
                device,
            )

            val_data = TimestampDataset(
                self.x[split.train_val_split : split.val_end],
                self.y[split.train_val_split : split.val_end],
                self.timestamps[split.train_val_split : split.val_end],
                device,
            )

            return train_data, val_data

    def split(
        self, scale=True, device="cpu"
    ) -> Generator[tuple[TimestampDataset, TimestampDataset], None, None]:

        for i in range(len(self.split_indices)):
            yield self.get_train_val(i, scale, device)

    def get_train_val_full(self, scale=True, device="cpu"):
        train_val_split = self.split_indices[-1].train_val_split
        val_end = self.split_indices[-1].val_end

        if scale:
            train_data = TimestampDataset(
                self.wind_speed_scaler_full.transform(self.x[0:train_val_split]),
                self.y[0:train_val_split],
                self.timestamps[0:train_val_split],
                device,
            )

            val_data = TimestampDataset(
                self.wind_speed_scaler_full.transform(self.x[train_val_split:val_end]),
                self.y[train_val_split:val_end],
                self.timestamps[train_val_split:val_end],
                device,
            )

            return train_data, val_data

        else:
            train_data = TimestampDataset(
                self.x[0:train_val_split],
                self.y[0:train_val_split],
                self.timestamps[0:train_val_split],
                device,
            )

            val_data = TimestampDataset(
                self.x[train_val_split:val_end],
                self.y[train_val_split:val_end],
                self.timestamps[train_val_split:val_end],
                device,
            )

            return train_data, val_data


def custom_shuffle_split(
    timestamps: pd.DatetimeIndex,
    n_splits: int,
):

    # Groups hourly timestamps by day, shuffles the days, and splits them into n groups of approximatly equal sizes
    days: list[pd.DatetimeIndex] = list(timestamps.groupby(timestamps.date).values())
    np.random.shuffle(days)

    timestamps_groups = [pd.DatetimeIndex([]) for _ in range(n_splits)]
    group_sizes = [0] * n_splits

    for day in days:
        # Find the group with the smallest current size
        min_group_index = group_sizes.index(min(group_sizes))

        # Add the day's timestamps to that group
        timestamps_groups[min_group_index] = timestamps_groups[min_group_index].union(
            day
        )
        group_sizes[min_group_index] += len(day)

    masks: list[np.ndarray] = []
    for timestamps_group in timestamps_groups:
        masks.append(np.isin(timestamps, timestamps_group))
    return masks
    # if wind_speed_scaler is None:
    #     for timestamps_group in timestamps_groups:
    #         mask = np.isin(timestamps, timestamps_group)

    #         yield (mask, TimestampDataset(
    #             x[mask],
    #             y[mask],
    #             timestamps[mask],
    #             device,
    #         ))
    # else:
    #     wind_speed_scaler_all_data = WindSpeedScaler(x)
    #     for timestamps_group in timestamps_groups:
    #         mask = np.isin(timestamps, timestamps_group)

    #         yield (mask, TimestampDataset(
    #             wind_speed_scaler_all_data.transform(x[mask]),
    #             y[mask],
    #             timestamps[mask],
    #             device,
    #         ))
