import __fix_relative_imports  # noqa: F401
import numpy as np
import pandas as pd
from mscEidalVesetrud.global_constants import (
    START_TIME,
    END_TIME,
    WEATHER_DATA_PATH,
)


def __col_name_to_index(name: str):

    # Increasing Lonitude means more east
    # Increasing Latitude means more north
    lat_map = {
        "6400": 0,
        "6401": 0,
        "6403": 1,
        "6405": 2,
        "6407": 3,
        "6409": 4,
        "6410": 4,
        "6412": 5,
        "6414": 6,
        "6416": 7,
        "6418": 8,
        "6421": 9,
        "6423": 10,
        "6425": 11,
        "6427": 12,
    }
    long_map = {
        "1007": 0,
        "1012": 1,
        "1017": 2,
        "1023": 3,
        "1028": 4,
        "1033": 5,
        "1038": 6,
        "1043": 7,
        "1048": 8,
        "1053": 9,
        "1059": 10,
        "1064": 11,
        "1069": 12,
    }
    type_map = {
        "speed": 0,
        "direction": 1,
    }
    height_map = {"10m": 0, "80m": 1, "120m": 2}

    # Last to digits
    hour = int(name[-2:])
    # Hours from calculation time at 06:00
    # hour = 18 means 00:00
    # Want a inteval from 19:00 to 04:00 (inclusive) with 34 hours in between
    # This means an hour from 13 to 46 inclusive
    if hour < 13 or hour > 46:
        return None
    hour_idx = hour - 13

    # Example column name: 'wind_direction_10m_6400_1007_13'

    name_split = name.split("_")

    if name_split[-2] == "roan":
        return (hour_idx, 6, 6, height_map[name_split[2]], type_map[name_split[1]])

    return (
        hour_idx,
        lat_map[name_split[3]],
        long_map[name_split[4]],
        height_map[name_split[2]],
        type_map[name_split[1]],
    )


def rolling_window_axis_1(weather_data: np.ndarray, timestamps: pd.DatetimeIndex):
    """
    Creates a rolling window view of a NumPy array along the second axis (axis 1),
    makes a copy to avoid overlapping memory issues, and reshapes the array to merge
    the first two dimensions.

    Args:
        a: The input NumPy array.  Must have at least two dimensions.
        window_size: The size of the rolling window.

    Returns:
        A NumPy array representing the rolling window view. The returned array
        will have shape (a.shape[0]*(a.shape[1] - window_size + 1), window_size, *a.shape[2:]),

    Example:
        Calling rolling_window_axis_1 with window_size=3 on the array:
        [[71. 72. 73. 74. 75. 76.]
        [81. 82. 83. 84. 85. 86.]
        [91. 92. 93. 94. 95. 96.]]
        Results in strided_a becoming:
        [[[71. 72. 73.]
          [72. 73. 74.]
          [73. 74. 75.]
          [74. 75. 76.]]

          [[81. 82. 83.]
          [82. 83. 84.]
          [83. 84. 85.]
          [84. 85. 86.]]

          [[91. 92. 93.]
          [92. 93. 94.]
          [93. 94. 95.]
          [94. 95. 96.]]]
        And a final result of:
        [[71. 72. 73.]
        [72. 73. 74.]
        [73. 74. 75.]
        [74. 75. 76.]
        [81. 82. 83.]
        [82. 83. 84.]
        [83. 84. 85.]
        [84. 85. 86.]
        [91. 92. 93.]
        [92. 93. 94.]
        [93. 94. 95.]
        [94. 95. 96.]]
    """
    window_size = 11
    time_offset = 18
    assert weather_data.shape[1:] == (34, 13, 13, 3, 2)
    assert window_size < weather_data.shape[1]
    assert timestamps.shape[0] == weather_data.shape[0]

    # Keep the 0'th axis untouched, and split axis 1 into two new axes: number 1 and 2,
    # and increase the index of the remaining axes
    shape = (
        weather_data.shape[0],
        weather_data.shape[1] - window_size + 1,
        window_size,
    ) + weather_data.shape[2:]
    strides = (
        weather_data.strides[0],
        weather_data.strides[1],
        *weather_data.strides[1:],
    )
    rolled_weather_data = np.copy(
        np.lib.stride_tricks.as_strided(weather_data, shape=shape, strides=strides)
    )
    stacked_weather_data = np.reshape(
        rolled_weather_data, (-1, *rolled_weather_data.shape[2:])
    )

    assert stacked_weather_data.shape[1:] == (window_size, 13, 13, 3, 2)
    assert stacked_weather_data.shape[0] == 24 * weather_data.shape[0]

    time_start = timestamps[0] + pd.to_timedelta(time_offset, unit="h")
    time_end = timestamps[-1] + pd.to_timedelta(time_offset + 23, unit="h")
    new_timestamps = pd.DatetimeIndex(pd.date_range(time_start, time_end, freq="h"))

    def check_equal(
        weather_np_old: np.ndarray,
        timestamps_old: pd.DatetimeIndex,
        weather_np_new: np.ndarray,
        timestamps_new: pd.DatetimeIndex,
    ):

        assert weather_np_new.shape[0] == 24 * weather_np_old.shape[0]
        assert weather_np_new.shape[0] == timestamps_new.shape[0]
        assert weather_np_old.shape[0] == timestamps_old.shape[0]
        assert weather_np_new.shape[1:] == (11, 13, 13, 3, 2)
        assert weather_np_old.shape[1:] == (34, 13, 13, 3, 2)

        for index_new in range(weather_np_new.shape[0]):
            timestamp_new = timestamps_new[index_new]
            timestamp_old = (timestamp_new - pd.to_timedelta(1, unit="D")).replace(
                hour=6
            )
            index_old = timestamps_old.get_loc(timestamp_old)
            index_offset = (
                int((timestamp_new - timestamp_old).total_seconds() / 3600) - 18
            )

            assert type(index_old) is int
            assert index_old * 24 + index_offset == index_new
            assert np.array_equal(
                weather_np_new[index_new],
                weather_np_old[index_old, index_offset : index_offset + 11],
                True,
            )

    check_equal(weather_data, timestamps, stacked_weather_data, new_timestamps)

    return stacked_weather_data, new_timestamps


def transform_wind_direction(weather_np: np.ndarray):
    assert weather_np.shape[1:] == (11, 13, 13, 3, 2)
    assert weather_np.dtype == np.float32
    result = np.zeros((weather_np.shape[0], 11, 13, 13, 3, 3), dtype=np.float32)

    # Copy over speed data
    result[:, :, :, :, :, 0] = weather_np[:, :, :, :, :, 0]

    radians = np.deg2rad(weather_np[:, :, :, :, :, 1])
    # Calculate cos and sin of wind direction
    result[:, :, :, :, :, 1] = np.cos(radians)
    result[:, :, :, :, :, 2] = np.sin(radians)

    return result


def inverse_transform_wind_direction(weather_np: np.ndarray):
    assert weather_np.shape[1:] == (11, 13, 13, 3, 3)
    assert weather_np.dtype == np.float32
    result = np.zeros((weather_np.shape[0], 11, 13, 13, 3, 2), dtype=np.float32)

    # Copy over speed data
    result[:, :, :, :, :, 0] = weather_np[:, :, :, :, 0]

    # Calculate the angle
    result[:, :, :, :, :, 1] = np.rad2deg(
        np.arctan2(weather_np[:, :, :, :, :, 2], weather_np[:, :, :, :, :, 1])
    )

    return result


class WindSpeedScaler:
    # TODO: update this
    """
    Scales the wind speed data to a mean of 0, and a standard deviation of 0.667,
    mimicking the sine and cosine of the wind direction
    """

    def __init__(self, weather_train: np.ndarray) -> None:
        # TODO: Should we compute the same average for all wind speeds in the grid (like now), or one in total, or one per height?

        assert weather_train.shape[1:] == (11, 13, 13, 3, 3)

        # Because the weather data contains zero NaN values except for the timestamps that are
        # missing where every value is NaN, we can compute the average based on the non NaN values
        self.mean: np.ndarray = np.nanmean(weather_train[:, :, :, :, :, 0], axis=0)
        self.std: np.ndarray = np.nanstd(weather_train[:, :, :, :, :, 0], axis=0)

        # Articicially increase the measuerd standard deviation by 50%, to make the scaled wind speed typically
        # vary from -1 to 2, which is closer to the cosine and sine of the wind direction varying between -1 and 1
        # This results in the standard deviation of the scaled wind speed being 1/1.5 = 0.667, very close to the standard
        # deviation of the sine and cosine.
        self.std *= 1.5

    def transform(self, data: np.ndarray) -> np.ndarray:
        assert data.shape[1:] == (11, 13, 13, 3, 3)

        scaled_speed = (data[:, :, :, :, :, 0] - self.mean) / self.std
        return np.concat(
            [scaled_speed[..., np.newaxis], data[:, :, :, :, :, 1:]], axis=-1
        )

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        assert data.shape[1:] == (11, 13, 13, 3, 3)

        scaled_speed = data[:, :, :, :, :, 0] * self.std + self.mean
        return np.concat(
            [scaled_speed[..., np.newaxis], data[:, :, :, :, :, 1:]], axis=-1
        )


def get_weather_data(
    start_time, end_time, verbose=True
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Collects weather data indexed on calculation time from a csv file, and massages it into a
    numpy array indexed on prediction time. The semantics and shape of the resulting array is:
    (number_of_timestamps, hour_offset(11), latitude(13), longitude(13), height(3), speed_direction_cos_sin(3))
    """

    weather_df = pd.read_csv(WEATHER_DATA_PATH, index_col=0)
    weather_df.index = pd.DatetimeIndex(weather_df.index)

    # Dealing with this is no longer neccecary, as the relevant row in the csv file is removed
    # Enormous wind speeds (9.96921e+36) that is probably an artefact from Met
    # Setting erroneous data to NaN:
    # weather_df.replace({9.96921e+36: np.nan}, inplace=True)
    # They are present at wind_speed_10m at for all 13 by 13 grid values at:
    # 2021-06-16 02:00:00 and 2021-06-16 16:00:00

    # Fill in any missing date times with nan:
    start_calc_time = weather_df.index.min()
    end_calc_time = weather_df.index.max()
    weather_df = weather_df.reindex(
        pd.date_range(start=start_calc_time, end=end_calc_time, freq="D")
    )
    weather_df.index = pd.to_datetime(
        weather_df.index
    )  # Only for correct type analysis

    if verbose:
        df_nan = weather_df.isna().any(axis=1)
        df_nan_index = df_nan[df_nan].index
        date_dict = {}
        for date in df_nan_index:
            date_dict[str(date)[:10]] = round(
                float(weather_df.loc[date].isna().sum() / 1014), 1
            )
        print("Dates with nan:")
        print(date_dict)

    weather_np = np.zeros((weather_df.shape[0], 34, 13, 13, 3, 2), dtype=np.float32)

    for column in weather_df.columns:
        index = __col_name_to_index(column)
        if index is None:
            continue
        weather_np[:, *index] = weather_df[column].values

    weather_np, timestamps = rolling_window_axis_1(weather_np, weather_df.index)
    weather_np = transform_wind_direction(weather_np)

    try:
        start_idx = timestamps.get_loc(start_time)
        end_idx = timestamps.get_loc(end_time)
    except KeyError:
        raise KeyError(
            f"Given start and end times outside valid range from {timestamps.min()} to {timestamps.max()}"
        )

    return weather_np[start_idx : end_idx + 1], timestamps[start_idx : end_idx + 1]


if __name__ == "__main__":

    # old_weather_nan = {'2021-03-29': 24, '2021-06-01': 24, '2021-07-08': 24, '2021-08-19': 24, '2021-11-04': 24, '2021-11-12': 24,
    #                    '2022-01-17': 24, '2022-01-18': 24, '2022-03-08': 24, '2022-03-09': 24, '2022-03-11': 24, '2022-03-16': 24, '2022-03-17': 24, '2022-03-18': 24, '2022-07-31': 24, '2022-09-23': 24, '2022-12-26': 24, '2022-12-29': 24, '2023-05-23': 16}
    # new_weather_nan = {'2020-02-08': 29.0, '2020-02-09': 5.0, '2020-02-17': 11.7, '2020-02-26': 11.7, '2020-03-23': 23.3, '2020-07-21': 7.3, '2020-07-24': 11.7,
    #                    '2021-05-31': 11.7, '2021-06-15': 6.0, '2021-07-07': 23.3, '2021-07-25': 20.7, '2021-08-18': 11.7, '2021-11-03': 11.7,
    #                    '2022-01-16': 11.7, '2022-01-17': 11.7, '2022-03-07': 11.7, '2022-03-08': 11.7, '2022-03-10': 11.7, '2022-03-15': 11.7, '2022-03-16': 11.7, '2022-03-17': 11.7, '2022-07-17': 16.0, '2022-07-30': 11.7, '2022-09-22': 11.7, '2022-12-01': 4.7, '2022-12-25': 35.0, '2023-01-05': 7.3, '2023-05-22': 7.3, '2023-06-21': 16.7, '2023-10-01': 25.3, '2023-10-02': 33.3}
    # '2022-12-25 06:00:00' is missing entirely
    # new_new_nan     = {'2020-02-17': 34.0, '2020-02-26': 34.0, '2020-03-23': 34.0, '2020-07-21': 34.0, '2020-07-24': 34.0,
    #                    '2021-05-31': 34.0, '2021-06-15': 34.0, '2021-07-07': 34.0, '2021-07-25': 34.0, '2021-08-18': 34.0, '2021-11-03': 34.0,
    #                    '2022-01-16': 34.0, '2022-01-17': 34.0, '2022-03-07': 34.0, '2022-03-08': 34.0, '2022-03-10': 34.0, '2022-03-15': 34.0, '2022-03-16': 34.0, '2022-03-17': 34.0, '2022-07-17': 34.0, '2022-07-30': 34.0, '2022-09-22': 34.0, '2022-12-01': 34.0, '2022-12-25': 34.0, '2023-01-05': 34.0, '2023-05-22': 34.0, '2023-06-21': 34.0, '2023-12-30': 34.0, '2023-12-31': 34.0, '2024-02-07': 34.0, '2024-03-22': 34.0}

    weather_np_new, timestamps = get_weather_data(START_TIME, END_TIME, verbose=True)
