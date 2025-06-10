import __fix_relative_imports  # noqa: F401

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Generator

from mscEidalVesetrud.baseline.lightgbm_model import LightGBMModel
from mscEidalVesetrud.baseline.data_prep.paper_features import PaperFeatures
from mscEidalVesetrud.global_constants import TRAIN_SIZE, VAL_SIZE


class CrossValidation:
    def __init__(
        self,
        print_feature_importance: bool = True,
    ) -> None:
        self.print_feature_importance = print_feature_importance

    def __find_end_time(
        self, time_index: int, datetime_list: list[pd.DatetimeIndex]
    ) -> int | None:
        if time_index >= len(datetime_list) - 1:
            return None
        while True:
            if time_index + 1 == len(datetime_list):
                return time_index
            elif datetime_list[time_index].day != datetime_list[time_index + 1].day:
                return time_index + 1
            else:
                time_index += 1

    def __split(
        self, data: pd.DataFrame, train_size: int, val_size: int, buffer_size: int
    ) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        This function splits the data into training and testing sets
        based on the train_size, val_size, and step_size.
        It yields the training and testing sets as a tuple.
        :param data: pd.DataFrame - the data to split
        :return: Generator - a generator that yields the training and testing sets
        """
        train_start = 0
        train_end = self.__find_end_time(train_size, data.index)
        val_start = train_end + buffer_size
        val_end = self.__find_end_time(val_start + val_size, data.index)
        while True:
            if not val_end or val_end > len(data):
                break

            print(
                f"train size: {train_end - train_start}, val size: {val_end - val_start}"
            )
            print(
                f"train end: {data.iloc[train_end].name}, val end: {data.iloc[val_end].name}"
            )

            yield data[train_start:train_end], data[val_start:val_end]

            train_end = self.__find_end_time(train_end + val_size, data.index)
            train_start = train_end - train_size
            val_start = train_end + buffer_size
            val_end = self.__find_end_time(val_start + val_size, data.index)

    def sliding_window_cross_validation(
        self,
        data: pd.DataFrame,
        target: str = "prod",
        pca_data: pd.DataFrame = None,
        pca_split: bool = False,
        train_size: int = TRAIN_SIZE,
        val_size: int = VAL_SIZE,
        buffer_size: int = 0,
        remove_negative: bool = False,
    ) -> tuple[list[tuple[float, float]], pd.DataFrame]:
        """
        This function performs sliding window cross-validation
        on the given data. It returns the results of the cross-validation
        in a tuple containing MAE and MSE.
        When including df_all_calc_times, the feature Weighted Mean Past will
        automatically be calculated and included in the model.
        :param power_plant: str - the name of the power plant
        :param data: pd.DataFrame - the data to perform cross-validation on
        :param df_all_calc_times: pd.DataFrame - optional data to make
        feature Weighted Mean Past
        :param use_logit: bool - whether to use logit transform
        :param alt_test_data: pd.DataFrame - alternative test data, must include
        the same columns as the original test data and an availability column
        containing the availability in range [0, 1]
        :return: tuple - a tuple containing the results of the cross-validation
        """
        model = LightGBMModel()
        results = []
        all_predictions = pd.DataFrame()

        if pca_data is not None:
            pf = PaperFeatures(power_plant="roan")
            flag = True

        for train, test in self.__split(data, train_size, val_size, buffer_size):
            if pca_data is not None:
                if flag:
                    print("performing local PCA")
                    flag = False
                pca_train, pca_test = pf.principal_component_analysis_train_and_test(
                    train.index, test.index, pca_data, pca_split=pca_split
                )
                train = pd.concat([train, pca_train], axis=1, join="inner")
                test = pd.concat([test, pca_test], axis=1, join="inner")

            train_score, test_score, predictions = model.test_model(
                train.copy(),
                test.copy(),
                target=target,
                remove_negative=remove_negative,
            )
            results.append((train_score, test_score))
            all_predictions = pd.concat([all_predictions, predictions])

        if self.print_feature_importance:
            model.print_feature_importances()

        return results, all_predictions

    def print_results(
        self,
        results: tuple[list[tuple[float, float]], list[tuple[float, float]]],
        fancy_format: bool = False,
    ) -> None:
        """
        Prints the result in the form of a bar chart.
        :param results: tuple - a tuple containing the results of the cross-validation
        """
        if fancy_format:
            try:
                # Set the font globally
                plt.rcParams["text.usetex"] = True
                plt.rcParams["font.family"] = "Computer Modern Roman"
                # Use the below instead if you do not have LaTeX installed
                # plt.rcParams["font.family"] = "cmr10"
                # plt.rcParams["axes.formatter.use_mathtext"] = True

                plt.rcParams["font.size"] = 10
                plt.rcParams["legend.fontsize"] = "medium"
                plt.rcParams["axes.labelsize"] = "medium"
                plt.rcParams["axes.titlesize"] = "large"
                plt.rcParams["xtick.labelsize"] = "medium"
                plt.rcParams["ytick.labelsize"] = "medium"
            except Exception as e:
                print(f"Error setting font globally: {e}. Running with standard font.")

        fair_data, overfit_data = results

        fair_data_mae = [mae for mae, _ in fair_data]
        mean_fair_data_mae = np.mean(fair_data_mae)
        n_folds = len(fair_data_mae)

        # Create an array for the x-axis positions with 1-indexing
        x = np.arange(1, n_folds + 1)

        # Create the plot with the specified size and layout
        _, ax = plt.subplots(constrained_layout=True, figsize=(5.5, 4))

        # Plot the bars for individual folds
        ax.bar(x, fair_data_mae, width=0.4, label="MAE", color="blue", zorder=1)

        if overfit_data is not None:
            overfit_data_mae = [mae for mae, _ in overfit_data]
            mean_overfit_data_mae = np.mean(overfit_data_mae)

            ax.bar(
                x,
                overfit_data_mae,
                width=0.4,
                label="Overfit MAE",
                color="orange",
                zorder=2,
            )

            # Add a horizontal line for the mean overfit MAE with increased visibility
            if mean_fair_data_mae == mean_overfit_data_mae:
                x_offset = 1.1  # Horizontal offset
                ax.axhline(
                    y=mean_overfit_data_mae,
                    xmin=x[0] - x_offset,
                    xmax=x[-1] + x_offset,
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    label="Mean Overfit MAE",
                    zorder=4,
                )
            else:
                ax.axhline(
                    y=mean_overfit_data_mae,
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    label="Mean Overfit MAE",
                    zorder=4,
                )

        # Add a horizontal line for the mean fair MAE with increased visibility
        ax.axhline(
            y=mean_fair_data_mae,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Mean MAE",
            zorder=3,
        )

        # Set x-axis ticks to be integers
        ax.set_xticks(x)

        ax.set_xlabel("Fold Index")
        ax.set_ylabel("MAE")
        ax.set_title("MAE Comparison Across Folds")
        ax.legend()
        plt.show()
