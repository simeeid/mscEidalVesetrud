import pandas as pd


def find_dates_with_nan(df: pd.DataFrame):
    df_nan = df.isna().any(axis=1)
    df_nan_index = df_nan[df_nan].index
    date_list = [str(dt)[:10] for dt in df_nan_index]
    date_list.sort()
    date_dict: dict[str, int] = {}
    for date in date_list:
        if date not in date_dict.keys():
            date_dict[date] = 1
        else:
            date_dict[date] += 1
    return date_dict


def read_data(
    path: str,
    start_time: str,
    end_time="2023-10-01 23:00:00",
    index_col=0,
    verbose=True,
) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=index_col)
    df.index = pd.to_datetime(df.index)

    # For us the park is always Roan
    if "park" in df.columns:
        df = df.drop("park", axis=1)

    if verbose:
        print("Opening:", path)
        print("Total number of rows before fill:", df.shape[0])
        print(
            "Number of rows containing NaN values before fill:",
            df.isna().any(axis=1).sum(),
        )
    # This discards data before start_time, and after end_time, and fills any missing timestamps with NaN values
    df = df.reindex(pd.date_range(start=start_time, end=end_time, freq="h"))

    if verbose:
        print("Total number of rows:", df.shape[0])
        print("Number of rows containing NaN values:", df.isna().any(axis=1).sum())
        print("Dates containing nan:")
        print(find_dates_with_nan(df))
        print()
    return df
