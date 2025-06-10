import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


class PaperFeatures:
    """
    This class is used to calculate the spatial and temporal features
    described in the paper "Improving Renewable Energy Forecasting
    with a Grid of Numerical Weather Predictions"
    by Jose R. Andrade and Ricardo J. Bessa
    """

    def __init__(
        self,
        power_plant: str,
        path_prefix: str = "",
        cos_sin_transform: bool = False,
        include_temporal_features: bool = False,
    ) -> None:
        """
        param power_plant: The name of the power plant, which corresponds to the
        alias of the center point of the grid.

        param bayes_opt_iterations: The number of iterations for the
        Bayesian Optimization.

        param bayes_opt_verbose: The verbosity level for the Bayesian Optimization.

        param path_prefix: The prefix to the path where the data is stored.
        This varies depending on the environment the code is run in.

        param cos_sin_transform: Whether to use the cosine and sine
        transformation for wind direction. Default is False.

        param include_temporal_features: Whether to indluce the calculated
        temporal features. This may lead to a subtle error because of the
        horizon of forecasts. Default is False.
        """

        self.power_plant: str = power_plant
        self.default_variables = [
            "wind_speed",
            "wind_direction",
            "u",
            "v",
        ]
        self.default_height_levels = ["10m", "80m", "120m"]
        self.default_calc_times = ["d1_06", "d2_06", "d2_12", "d2_18"]
        self.temporal_variance_window_sizes = [3, 7, 11]
        self.path_prefix = path_prefix
        self.cos_sin_transform = cos_sin_transform
        self.include_temporal_features = include_temporal_features

    def make_features(self, df: pd.DataFrame, num_lags_leads: int) -> pd.DataFrame:
        """
        This method calculates all the features from the given weather data.
        It calls the private methods to calculate spatial
        and temporal features, and combines them into a single DataFrame.
        :param df: The data frame containing the weather data
        :return: The data frame containing only the calculated features
        """
        df = self.make_wind_features(df)
        print("Calculating spatial features.")
        df_spatial = self.make_spatial_features(df)
        if self.include_temporal_features:
            print("Calculating temporal features.")
            df_temporal = self.make_temporal_features(df, num_lags_leads)

            # check of "prod" is in the columns
            if "prod" in df.columns:
                df = pd.concat(
                    [df_spatial, df_temporal, df[["prod"]]], axis=1, join="inner"
                )
            else:
                df = pd.concat([df_spatial, df_temporal], axis=1, join="inner")
        else:
            # check of "prod" is in the columns
            if "prod" in df.columns:
                df = pd.concat([df_spatial, df[["prod"]]], axis=1, join="inner")
            else:
                df = df_spatial

        df = df.dropna(subset=df.columns[~df.columns.str.contains("d2")])

        return df

    def make_wind_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method.
        This method calculates the Azimuthal and Meridional wind components
        from the wind speed and direction (u and v components).
        :param df: The data frame containing the wind speed and direction
        :return: The data frame with the added columns for
        the Azimuthal and Meridional wind components
        """
        cols = df.columns

        # Get all unique "rest of the name" parts
        names = {
            col.split("_", 2)[-1]
            for col in cols
            if col.startswith("wind_speed_") or col.startswith("wind_direction_")
        }

        # Create an empty dictionary for the new columns
        new_cols = {}
        for name in names:
            speed_col = f"wind_speed_{name}"
            dir_col = f"wind_direction_{name}"

            # Convert wind direction from degrees to radians and store in a new column
            rad_dir_col = df[dir_col].apply(np.deg2rad)
            new_cols[f"u_{name}"] = -df[speed_col] * np.sin(rad_dir_col)
            new_cols[f"v_{name}"] = -df[speed_col] * np.cos(rad_dir_col)

        # Convert the dictionary to a DataFrame
        df_new = pd.DataFrame(new_cols)
        df = pd.concat([df, df_new], axis=1)

        return df

    def make_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method uses the helper methods for standard deviation, smoothing and PCA
        to calculate the spatial features and combine these into one data frame.
        :param df: The data frame containing the weather data
        :return: The data frame only containing the spatial features
        """
        df = df.copy()

        df = self.spatial_standard_deviation(df)
        df = self.spatial_smoothing(df)
        PCA_df = self.principal_component_analysis(df)

        filtered_columns = [
            col
            for col in df.columns
            if "spatial_smoothing" in col or "spatial_std" in col
        ]

        df = df[filtered_columns]
        df = pd.concat([df, PCA_df], axis=1)

        return df

    def spatial_standard_deviation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the spatial standard deviation for the weather data.
        :param df: The data frame containing the weather data
        :return: The data frame with added columns for the spatial standard deviation
        """

        # Iterate over each feature type, height, and time
        for variable in self.default_variables:
            for height in self.default_height_levels:
                # Create a list to store the column names that match
                # the current feature, height, and time
                matching_columns = [
                    col
                    for col in df.columns
                    if all(x in col for x in [variable, height, "d1_06"])
                ]

                # Calculate the spatial standard deviation of the matching columns
                if self.cos_sin_transform and variable == "wind_direction":
                    df[f"{variable}_sin_{height}_d1_06_spatial_std"] = df[
                        matching_columns
                    ].apply(
                        lambda row: np.sqrt(
                            (
                                (
                                    np.sin(np.deg2rad(row))
                                    - np.sin(np.deg2rad(row).mean())
                                )
                                ** 2
                            ).sum()
                            / (np.sin(np.deg2rad(row)).count() - 1)
                        ),
                        axis=1,
                    )

                    df[f"{variable}_cos_{height}_d1_06_spatial_std"] = df[
                        matching_columns
                    ].apply(
                        lambda row: np.sqrt(
                            (
                                (
                                    np.cos(np.deg2rad(row))
                                    - np.cos(np.deg2rad(row).mean())
                                )
                                ** 2
                            ).sum()
                            / (np.cos(np.deg2rad(row)).count() - 1)
                        ),
                        axis=1,
                    )
                else:
                    df[f"{variable}_{height}_d1_06_spatial_std"] = df[
                        matching_columns
                    ].apply(
                        lambda row: np.sqrt(
                            ((row - row.mean()) ** 2).sum() / (row.count() - 1)
                        ),
                        axis=1,
                    )

        return df

    def spatial_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the spatial smoothing for the weather data.
        NOTE: This feature is not calculated for wind direction.
        The authors of the paper do the same.
        :param df: The data frame containing the weather data
        :return: The data frame with added columns for the spatial smoothing
        """

        variables = [var for var in self.default_variables if var != "wind_direction"]

        # Iterate over each feature type, height, and time
        for variable in variables:
            for height in self.default_height_levels:
                # Create a list to store the column names that match
                # the current feature, height, and time
                matching_columns = [
                    col
                    for col in df.columns
                    if all(x in col for x in [variable, height, "d1_06"])
                    and "spatial_std" not in col
                ]

                # Calculate the spatial standard deviation of the matching columns
                if matching_columns:  # Only calculate if there are matching columns
                    df[f"{variable}_{height}_d1_06_spatial_smoothing"] = df[
                        matching_columns
                    ].apply(
                        lambda row: row.mean(),
                        axis=1,
                    )

        return df

    def principal_component_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform principal component analysis on the weather data.
        The PCA is performed on the wind direction, wind speed, u (only 120m),
        and v (only 120m) components
        For wind direction, the sine and cosine transformed components are used,
        because of the circular nature of wind direction.
        :param df: The data frame containing the weather data
        :return: The data frame containing the principal components up to 95% variance
        """

        # List to store the resulting principal components
        pca_results_list = []

        # Dictionary to store the count of principal components for each group
        pca_count_dict = {}

        for variable in self.default_variables:
            for height in self.default_height_levels:
                # u and v only for height 120m
                if (variable == "u" or variable == "v") and height != "120m":
                    continue

                # Filter columns containing the current type and height
                relevant_columns = [
                    col for col in df.columns if f"{variable}_{height}_d1" in col
                ]

                # Exclude columns containing "spatial_smoothing" or "spatial_std"
                relevant_columns = [
                    col
                    for col in relevant_columns
                    if ("spatial_smoothing" not in col) and ("spatial_std" not in col)
                ]

                if not relevant_columns:
                    print(f"No columns found for {variable} at {height}")
                    continue
                relevant_columns_df = df[relevant_columns]

                # Handle circular nature of wind direction
                if variable == "wind_direction":
                    wind_direction_rad = np.deg2rad(relevant_columns_df.values)

                    sin_components = np.sin(wind_direction_rad)
                    cos_components = np.cos(wind_direction_rad)

                    # Combine sine and cosine components into a single DataFrame
                    components = np.hstack((sin_components, cos_components))
                    relevant_columns_df = pd.DataFrame(components)

                pca = PCA(n_components=0.95)  # Variance threshold of 95%
                pca_result = pca.fit_transform(relevant_columns_df)

                # Store the resulting principal components in the list
                for i in range(pca_result.shape[1]):
                    pc_name = f"PC_{variable}_{height}_{i + 1}"
                    pca_results_list.append((pc_name, pca_result[:, i]))

                # Store the count of principal components for this group
                pca_count_dict[f"{variable}_{height}"] = pca_result.shape[1]

        # Create a new dataframe from the list of principal components
        pca_df = pd.DataFrame(dict(pca_results_list))
        pca_df.index = df.index

        return pca_df

    def principal_component_analysis_train_and_test(
        self,
        train_index: list,
        test_index: list,
        pca_data: pd.DataFrame,
        pca_split: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform principal component analysis on the weather data.
        The PCA is performed on the wind direction, wind speed, u (only 120m),
        and v (only 120m) components.
        For wind direction, the sine and cosine transformed components are used,
        because of the circular nature of wind direction.
        :param train_df: The training data frame containing the weather data
        :param test_df: The test data frame containing the weather data
        :return: A tuple containing the transformed training and test data frames
        """

        if pca_split:
            train_df = pca_data.loc[train_index]
            test_df = pca_data.loc[test_index]

            # List to store the resulting principal components for train and test
            train_pca_results_list = []
            test_pca_results_list = []
        else:
            # both_index = train_index + test_index # TypeError: cannot add
            # DatetimeArray and DatetimeArray
            both_index = train_index.union(test_index)
            both_df = pca_data.loc[both_index]
            both_pca_results_list = []

        # Dictionary to store the count of principal components for each group
        pca_count_dict = {}

        for variable in self.default_variables:
            for height in self.default_height_levels:
                # u and v only for height 120m
                if (variable == "u" or variable == "v") and height != "120m":
                    continue

                if pca_split:
                    # Filter columns containing the current type and height
                    relevant_columns = [
                        col for col in train_df.columns if f"{variable}_{height}" in col
                    ]

                    # Exclude columns containing "spatial_smoothing" or "spatial_std"
                    relevant_columns = [
                        col
                        for col in relevant_columns
                        if ("spatial_smoothing" not in col)
                        and ("spatial_std" not in col)
                    ]

                    if not relevant_columns:
                        print(f"No columns found for {variable} at {height}")
                        continue

                    train_relevant_columns_df = train_df[relevant_columns]
                    test_relevant_columns_df = test_df[relevant_columns]

                    # Handle circular nature of wind direction
                    if variable == "wind_direction":
                        wind_direction_rad_train = np.deg2rad(
                            train_relevant_columns_df.values
                        )
                        wind_direction_rad_test = np.deg2rad(
                            test_relevant_columns_df.values
                        )

                        sin_components_train = np.sin(wind_direction_rad_train)
                        cos_components_train = np.cos(wind_direction_rad_train)
                        sin_components_test = np.sin(wind_direction_rad_test)
                        cos_components_test = np.cos(wind_direction_rad_test)

                        # Combine sine and cosine components into a single DataFrame
                        train_components = np.hstack(
                            (sin_components_train, cos_components_train)
                        )
                        test_components = np.hstack(
                            (sin_components_test, cos_components_test)
                        )
                        train_relevant_columns_df = pd.DataFrame(train_components)
                        test_relevant_columns_df = pd.DataFrame(test_components)

                    pca = PCA(n_components=0.95)  # Variance threshold of 95%
                    train_pca_result = pca.fit_transform(train_relevant_columns_df)
                    test_pca_result = pca.transform(test_relevant_columns_df)

                    # Store the resulting principal components in the list
                    for i in range(train_pca_result.shape[1]):
                        pc_name = f"PC_{variable}_{height}_{i+1}"
                        train_pca_results_list.append((pc_name, train_pca_result[:, i]))
                        test_pca_results_list.append((pc_name, test_pca_result[:, i]))

                    # Store the count of principal components for this group
                    pca_count_dict[f"{variable}_{height}"] = train_pca_result.shape[1]
                else:
                    # Filter columns containing the current type and height
                    relevant_columns = [
                        col for col in both_df.columns if f"{variable}_{height}" in col
                    ]

                    # Exclude columns containing "spatial_smoothing" or "spatial_std"
                    relevant_columns = [
                        col
                        for col in relevant_columns
                        if ("spatial_smoothing" not in col)
                        and ("spatial_std" not in col)
                    ]

                    if not relevant_columns:
                        print(f"No columns found for {variable} at {height}")
                        continue

                    both_relevant_columns_df = both_df[relevant_columns]

                    # Handle circular nature of wind direction
                    if variable == "wind_direction":
                        wind_direction_rad_both = np.deg2rad(
                            both_relevant_columns_df.values
                        )

                        sin_components_both = np.sin(wind_direction_rad_both)
                        cos_components_both = np.cos(wind_direction_rad_both)

                        # Combine sine and cosine components into a single DataFrame
                        both_components = np.hstack(
                            (sin_components_both, cos_components_both)
                        )
                        both_relevant_columns_df = pd.DataFrame(both_components)

                    pca = PCA(n_components=0.95)  # Variance threshold of 95%
                    both_pca_result = pca.fit_transform(both_relevant_columns_df)

                    # Store the resulting principal components in the list
                    for i in range(both_pca_result.shape[1]):
                        pc_name = f"PC_{variable}_{height}_{i+1}"
                        both_pca_results_list.append((pc_name, both_pca_result[:, i]))

                    # Store the count of principal components for this group
                    pca_count_dict[f"{variable}_{height}"] = both_pca_result.shape[1]

        # Create new dataframes from the list of principal components
        if pca_split:
            train_pca_df = pd.DataFrame(dict(train_pca_results_list))
            test_pca_df = pd.DataFrame(dict(test_pca_results_list))
            train_pca_df.index = train_df.index
            test_pca_df.index = test_df.index
        else:
            both_pca_df = pd.DataFrame(dict(both_pca_results_list))
            both_pca_df.index = both_df.index

            train_pca_df = both_pca_df.loc[train_index]
            test_pca_df = both_pca_df.loc[test_index]

        return train_pca_df, test_pca_df

    def make_temporal_features(
        self, df: pd.DataFrame, num_lags_leads: int
    ) -> pd.DataFrame:
        """
        This method calculates the temporal features for each variable and height level.
        These features are the temporal variance and the lags and leads
        used in the paper. Temporal features are calculated only for the center point
        of the grid.

        param df: The data frame containing the variables for each height level
        param num_lags_leads: The number of lags and leads to calculate in the
        lags and leads method

        return: The data frame containing the temporal features for each variable
        and height level
        """
        df = df.copy()
        tv_df = self.temporal_variance(df)
        ll_df = self.lags_and_leads(df, num_lags_leads)
        # NOTE: Drop rows where values for d2_18 calc time are missing.
        # These rows are assumed to be invalid.
        df = df.dropna(subset=df.columns[df.columns.str.contains("d2_18")])

        # Concatenate the data frames with temporal features
        # and remove the original columns
        df = pd.concat([df, tv_df, ll_df], axis=1, join="inner")
        df = df[[*tv_df.columns, *ll_df.columns]]

        df = df.dropna()

        return df
