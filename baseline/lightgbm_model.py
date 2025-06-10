import pandas as pd
import lightgbm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import shap


class LightGBMModel:
    """
    A simple class to train and evaluate a LightGBM model
    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = params
        self.model = None

    def test_model(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target: str = "prod",
        remove_negative: bool = False,
    ) -> tuple[float, float, pd.DataFrame | None]:
        """
        Train and evaluate a LightGBM model.
        :param train: pd.DataFrame - training data
        :param test: pd.DataFrame - testing data
        """
        x_train = train.drop(columns=[target])
        y_train = train[target]
        x_test = test.drop(columns=[target])
        y_test = test[[target]].copy()

        # Train model
        if self.params:
            self.model = lightgbm.LGBMRegressor(**self.params)
        else:
            self.model = lightgbm.LGBMRegressor()
        self.model.fit(x_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(x_test)

        y_pred = pd.DataFrame(y_pred, columns=[target])

        y_pred.index = x_test.index

        if remove_negative:
            y_pred[y_pred < 0] = 0
            y_test[y_test < 0] = 0

        mae: float = mean_absolute_error(y_test, y_pred)
        mse: float = mean_squared_error(y_test, y_pred)

        return mae, mse, y_pred

    def print_feature_importances(self) -> None:
        """
        Print the feature importances of the model
        """
        feature_importances = pd.Series(
            self.model.feature_importances_, index=self.model.feature_name_
        )
        sorted_importances = feature_importances.nlargest(10).sort_values()
        sorted_importances.plot(kind="barh")
        plt.show()

    def explain_single_prediction(
        self, x: pd.DataFrame, index: int | str, top_n: int = 5
    ) -> None:
        """
        Explain the feature contributions for a single prediction using SHAP.
        :param x: pd.DataFrame - input data (with features, excluding target)
        :param index: int | str - index of the specific example to explain
        :param top_n: int - number of top features to display
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Ensure x does not contain the target column
        single_example = x.loc[[index]]  # Select the specific example

        # Initialize SHAP TreeExplainer for LightGBM
        explainer = shap.TreeExplainer(self.model)

        # Compute SHAP values for the single example
        shap_values = explainer.shap_values(single_example)

        # Get feature names
        feature_names = single_example.columns

        # Create a DataFrame of SHAP values for the single example
        shap_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "SHAP_Value": shap_values[0],  # SHAP values for the single example
                "Feature_Value": single_example.iloc[0].values,  # Actual feature values
            }
        )

        # Sort by absolute SHAP value to get the most influential features
        shap_df["Abs_SHAP_Value"] = shap_df["SHAP_Value"].abs()
        top_features = shap_df.sort_values(by="Abs_SHAP_Value", ascending=False).head(
            top_n
        )

        # Print the top features
        print(f"\nTop {top_n} features influencing the prediction for index {index}:")
        print(top_features[["Feature", "SHAP_Value", "Feature_Value"]])

        # Plot SHAP values for the top features
        plt.barh(top_features["Feature"], top_features["SHAP_Value"])
        plt.xlabel("SHAP Value (Impact on Prediction)")
        plt.title(f"Top {top_n} Feature Contributions for Prediction at Index {index}")
        plt.show()

        # Optional: SHAP force plot for visualization
        shap.initjs()  # Initialize JavaScript for visualization (if in a Jupyter notebook)
        shap.force_plot(
            explainer.expected_value, shap_values[0], single_example, matplotlib=True
        )
        plt.show()
