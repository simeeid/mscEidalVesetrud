import __fix_relative_imports  # noqa: F401
import pandas as pd
import numpy as np
from mscEidalVesetrud.data_preprocessing.data_reading import read_data
from mscEidalVesetrud.global_constants import (
    MIN_ALLOWABLE_AVAILABILITY,
    WIND_TURBINE_POWER_MW,
    MAX_POWER_MW,
    PROD_DATA_PATH,
    EARLY_WTGS_AVL_DATA_PATH,
)


def get_adjusted_prod_data(
    start_time, end_time, verbose, baseline: bool = False
) -> np.ndarray | pd.DataFrame:

    # def print_nan_percent(values: np.ndarray):
    #     print(f'{np.isnan(values).sum()/values.shape[0]:.2%}')

    prev_availability_df = read_data(
        EARLY_WTGS_AVL_DATA_PATH, start_time, end_time, index_col=4, verbose=verbose
    )
    df = read_data(PROD_DATA_PATH, start_time, end_time, verbose=verbose)
    df.drop("planned_avl", axis=1, inplace=True)

    # For the downreg case, we want to only keep the datepoint if it is NaN or 0.0
    df.fillna({"prod_downreg": 0.0}, inplace=True)

    # For the Availability case, our two sources of availability data do not overlap,
    # so adding them here is equivalent to outer merging
    # Replacing NaN with 0 in this case means we discard the data, as availability below MIN_ALLOWABLE_AVAILABILITY is discarded
    df["avl"] = df["measured_avl"].fillna(
        0.0
    ) + WIND_TURBINE_POWER_MW * prev_availability_df["value"].fillna(0.0)
    df["avl"] = df["avl"].where(df["avl"] >= MIN_ALLOWABLE_AVAILABILITY)

    df["scaled_prod"] = df["prod"] / df["avl"]
    df["scaled_prod"] = df["scaled_prod"].where(df["prod_downreg"] == 0.0)
    df["scaled_prod"] = df["scaled_prod"].where(df["scaled_prod"] <= 1.0)

    # Set negative prod values to zero
    # The most negative value is about 2.8 MW, and clipping them to zero does not
    # Change that much, except letting us use a final activation function like the sigmoid
    df["scaled_prod"] = np.maximum(0, df["scaled_prod"])

    # Before the date '2021-06-12 11:00:00' we do not have data on down regulation
    # Based on the roan_prod_with_downreg.csv file we manually exclude the following timestamps
    # (And based on a scatterplot of the production and wind speed to determine clear outliers)
    remove_list_early_downreg = pd.to_datetime(
        [
            *pd.date_range("2021-02-24 01:00:00", "2021-02-24 03:00:00", freq="h"),
            *pd.date_range("2021-02-27 21:00:00", "2021-02-28 01:00:00", freq="h"),
            "2021-02-28 13:00:00",
            "2021-02-28 14:00:00",
            "2021-03-01 02:00:00",
            "2021-03-01 03:00:00",
            *pd.date_range("2021-03-01 21:00:00", "2021-03-02 00:00:00", freq="h"),
            *pd.date_range("2021-03-26 22:00:00", "2021-03-27 01:00:00", freq="h"),
            *pd.date_range("2021-04-03 12:00:00", "2021-04-05 00:00:00", freq="h"),
            *pd.date_range("2021-05-10 23:00:00", "2021-05-11 03:00:00", freq="h"),
            "2023-04-13 17:00:00",  # Issue with downreg data
            # Measured_avl probably low even though it is reported as high:
            "2024-06-28 00:00:00",
            "2024-11-18 05:00:00",
            "2024-11-18 06:00:00",
            "2024-11-29 07:00:00",  # Issue with downreg data
        ]
    )
    df.loc[remove_list_early_downreg, "scaled_prod"] = np.nan

    if baseline:
        df.drop(["prod", "prod_downreg", "measured_avl", "avl"], axis=1, inplace=True)
        df.rename(columns={"scaled_prod": "prod"}, inplace=True)
        df.dropna(inplace=True)
        df["prod"] = df["prod"] * MAX_POWER_MW
        return df

    return np.float32(df["scaled_prod"].values)  # type: ignore[arg-type, return-value]


if __name__ == "__main__":
    start_time = "2021-02-08 23:00:00"
    end_time = "2024-12-30 23:00:00"
    prod = get_adjusted_prod_data(start_time, end_time, verbose=True)
