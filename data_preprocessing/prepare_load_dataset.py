import __fix_relative_imports  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mscEidalVesetrud.data_preprocessing.prod_preprocessing import (
    get_adjusted_prod_data,
)
from mscEidalVesetrud.data_preprocessing.weather_preprocessing import (
    get_weather_data,
)
from mscEidalVesetrud.data_preprocessing.cross_validation import (
    CrossValidationDatasets,
)
from mscEidalVesetrud.global_constants import (
    START_TIME,
    TEST_SPLIT_TIME,
    END_TIME,
    TRAIN_SIZE,
    VAL_SIZE,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
)
from mscEidalVesetrud.data_preprocessing.cross_validation import (
    TimestampDataset,
)
from mscEidalVesetrud.data_preprocessing.weather_preprocessing import (
    WindSpeedScaler,
)


def prepare_split_and_save_data(
    start_time: str, test_split_time: str, end_time: str, verbose_read=False
):

    # Assert we split at the end of a day, to stop any overlap between the test and training set
    assert test_split_time[-8:] == "00:00:00"

    weather, timestamps = get_weather_data(start_time, end_time, verbose_read)
    prod = get_adjusted_prod_data(start_time, end_time, verbose_read)

    assert weather.shape[0] == timestamps.shape[0] == prod.shape[0]
    assert weather.dtype == prod.dtype == np.float32

    nan_mask: np.ndarray = np.invert(
        np.isnan(weather).any(axis=tuple(range(1, weather.ndim))) | np.isnan(prod)
    )
    print("nan values total:", (~nan_mask).sum())
    print("total kept:", nan_mask.sum())
    weather = weather[nan_mask, :]
    prod = prod[nan_mask]
    timestamps = timestamps[nan_mask]

    # After the nan data points were removed above, it is not certain that the given
    # timestamp is present, so find the closest more new timestamp
    test_split_index = timestamps.searchsorted(test_split_time, side="left")
    assert test_split_index < timestamps.shape[0]

    print(f"Training size: {test_split_index}")
    print(f"Test size: {prod.shape[0] - test_split_index}")

    save_data(
        TRAIN_DATA_PATH,
        weather[:test_split_index],
        prod[:test_split_index],
        timestamps[:test_split_index],
    )
    save_data(
        TEST_DATA_PATH,
        weather[test_split_index:],
        prod[test_split_index:],
        timestamps[test_split_index:],
    )


def save_data(path: str, x: np.ndarray, y: np.ndarray, timestamps: pd.DatetimeIndex):
    np.save(path + "_x.npy", x)
    np.save(path + "_y.npy", y)
    np.save(path + "_t.npy", timestamps.values)


def load_cross_val(
    path: str, train_size: int, val_size: int, verbose=False
) -> CrossValidationDatasets:
    x: np.ndarray = np.load(path + "_x.npy")
    y: np.ndarray = np.load(path + "_y.npy")
    t: np.ndarray = np.load(path + "_t.npy")
    timestamps = pd.DatetimeIndex(t)

    # for i in range(t.shape[0]-1):
    #     time_diff = (timestamps[i+1] - timestamps[i]) // pd.Timedelta(hours=1)
    #     if time_diff != 1:
    #         print(i, timestamps[i], timestamps[i+1], time_diff)

    cv_datasets = CrossValidationDatasets(
        x, y, timestamps, train_size, val_size, verbose
    )

    return cv_datasets


def load_scale_test_dataset(
    wind_scaler_full: WindSpeedScaler | None,
    test_path: str,
    device: str,
) -> TimestampDataset:

    x_test: np.ndarray = np.load(test_path + "_x.npy")
    y_test: np.ndarray = np.load(test_path + "_y.npy")
    t_test: np.ndarray = np.load(test_path + "_t.npy")

    assert x_test.shape[0] == y_test.shape[0] == len(t_test)

    return TimestampDataset(
        wind_scaler_full.transform(x_test) if wind_scaler_full is not None else x_test,
        y_test,
        pd.DatetimeIndex(t_test),
        device=device,
    )


if __name__ == "__main__":

    prepare_split_and_save_data(
        start_time=START_TIME, test_split_time=TEST_SPLIT_TIME, end_time=END_TIME
    )

    datasets = load_cross_val(TRAIN_DATA_PATH, TRAIN_SIZE, VAL_SIZE, verbose=True)

    t_test: np.ndarray = np.load(TEST_DATA_PATH + "_t.npy")
    print(t_test[-1])

    weather: np.ndarray = datasets.x
    prod: np.ndarray = datasets.y
    timestamps = datasets.timestamps
    wind = weather[:, 5, 6, 6, 2, 0]

    for i in range(prod.shape[0]):
        if wind[i] > 15 and prod[i] < 0.29:
            print(i, timestamps[i], prod[i], wind[i])

    plt.scatter(wind, prod, s=1, alpha=0.4)
    plt.savefig("wind_power_curve.pdf", format="pdf")
    plt.show()
