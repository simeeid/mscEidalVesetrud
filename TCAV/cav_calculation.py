import __fix_relative_imports  # noqa: F401

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge  # type: ignore[import-untyped]
from mscEidalVesetrud.TCAV.sign_test import (
    one_sample_sign_test_target_greater_than_sample_median,
)
from mscEidalVesetrud.deep_learning.neural_nets import IndexableModule

from mscEidalVesetrud.TCAV.test_model import SimpleNN
from captum.attr import LayerGradientXActivation  # type: ignore[import-untyped]
from scipy.stats import ttest_ind, ttest_1samp, wilcoxon
from scipy.stats._result_classes import TtestResult


class CAVData:
    def __init__(
        self,
        coefficients: np.ndarray,
        bias: float,
        coefficient_of_determination: float,
        tcav_sign_score: float | None,
        tcav_magnitude_score: float | None,
        sensitivities: np.ndarray | None,
    ) -> None:
        self.coefficients = coefficients
        self.__coef_length = np.linalg.norm(coefficients)
        self.bias = bias
        self.coefficient_of_determination = coefficient_of_determination
        self.tcav_sign_score = tcav_sign_score
        self.tcav_magnitude_score = tcav_magnitude_score
        self.sensitivities = sensitivities

    def __str__(self) -> str:
        with np.printoptions(precision=5, suppress=True):
            return (
                f"Coefficients:\n"
                f"{self.coefficients}\n"
                f"Bias:                               {str(np.array([self.bias]))[1:-1]}\n"
                f"Coefficient of Determination (R^2): {str(np.array([self.coefficient_of_determination]))[1:-1]}\n"
                f"TCAV Sign Score:                    {str(np.array([self.tcav_sign_score]))[1:-1]}\n"
                f"TCAV Magnitude Score:               {str(np.array([self.tcav_magnitude_score]))[1:-1]}\n"
            )

    def normalize(self):
        if np.linalg.norm(self.coefficients) == self.__coef_length:
            self.coefficients /= self.__coef_length

    def unnormalize(self):
        if np.linalg.norm(self.coefficients) == 1.0:
            self.coefficients *= self.__coef_length


class CAVStatsData:
    def __init__(
        self,
        sign_score_t_test_result: TtestResult | None,
        sign_score_t_test_result_1samp: TtestResult,
        tcav_sign_score_avg: float,
        tcav_sign_score_std: float,
        tcav_magnitude_score_avg: float,
        tcav_magnitude_score_std: float,
        coefficients_avg: np.ndarray,
        coefficients_std: np.ndarray,
        bias_avg: float,
        bias_std: float,
        coefficient_of_determination_avg: float,
        coefficient_of_determination_std: float,
    ) -> None:
        self.sign_score_t_test_result = sign_score_t_test_result
        self.sign_score_t_test_result_1samp = sign_score_t_test_result_1samp
        self.tcav_sign_score_avg = tcav_sign_score_avg
        self.tcav_sign_score_std = tcav_sign_score_std
        self.tcav_magnitude_score_avg = tcav_magnitude_score_avg
        self.tcav_magnitude_score_std = tcav_magnitude_score_std
        self.coefficients_avg = coefficients_avg
        self.coefficients_std = coefficients_std
        self.bias_avg = bias_avg
        self.bias_std = bias_std
        self.coefficient_of_determination_avg = coefficient_of_determination_avg
        self.coefficient_of_determination_std = coefficient_of_determination_std

    def concept_findable_in_network(self, treshold: float = 0.2) -> bool:
        return (
            self.coefficient_of_determination_avg
            - self.coefficient_of_determination_std
            > treshold
        )

    # def one_cav_for_concept(
    #     self, findable_treshold: float = 0.2, sgd_to_normal_std_ratio: float = 10
    # ):
    #     if not self.concept_findable_in_network(findable_treshold):
    #         return False

    #     return np.linalg.norm(
    #         self.sgd.coefficients_std
    #     ) < sgd_to_normal_std_ratio * np.linalg.norm(self.normal.coefficients_std)

    # def concept_used_by_network(
    #     self,
    #     findable_treshold: float = 0.2,
    #     sgd_to_normal_std_ratio: float = 10,
    #     coefficients_std_norm_treshold: float = 0.2,
    # ) -> bool:
    #     if not self.one_cav_for_concept(findable_treshold):
    #         return False

    @staticmethod
    def __display_float(x: float) -> str:
        return str(np.array([x]))[1:-1]

    @staticmethod
    def __display_t_test_pvalue(x: TtestResult | None) -> str:
        if x is None:
            return "None"
        return str(x.pvalue)

    def __str__(self) -> str:
        with np.printoptions(precision=5, suppress=True):
            result = (
                f"Coefficient of Determination Avg: {self.__display_float(self.coefficient_of_determination_avg)}"
                f"     Std: {self.__display_float(self.coefficient_of_determination_std)}\n"
                f"       Sign Score T-Test p-value: {self.__display_t_test_pvalue(self.sign_score_t_test_result)}\n"
                f"Sign Score T-Test p-value 1 samp: {self.__display_t_test_pvalue(self.sign_score_t_test_result_1samp)}\n"
                f"             TCAV Sign Score Avg: {self.__display_float(self.tcav_sign_score_avg)}"
                f"     Std: {self.__display_float(self.tcav_sign_score_std)}\n"
                f"        TCAV Magnitude Score Avg: {self.__display_float(self.tcav_magnitude_score_avg)}"
                f"     Std: {self.__display_float(self.tcav_magnitude_score_std)}\n"
            )
            if self.coefficients_avg.shape[0] <= 8:
                result += (
                    f"                Coefficients Avg: {self.coefficients_avg}\n"
                    f"                Coefficients Std: {self.coefficients_std}\n"
                    f"                        Bias Avg: {self.__display_float(self.bias_avg)} "
                    f"Std: {self.__display_float(self.bias_std)}"
                )
            else:
                result += (
                    f"Coefficients Avg:\n{self.coefficients_avg}\n"
                    f"Coefficients Std:\n{self.coefficients_std}\n"
                    f"Bias Avg: {self.__display_float(self.bias_avg)} "
                    f"Std: {self.__display_float(self.bias_std)}"
                )
            return result


class CAVStatsDataR2:
    def __init__(
        self,
        r2_greater_than_random_pvalue: float | None,
        r2_greater_than_random_test_statistic: float | None,
        r2_greater_than_random_degrees_of_freedom: float | None,
        coefficient_of_determination_avg: float,
        coefficient_of_determination_std: float,
        coefficients_avg: np.ndarray,
        coefficients_std: np.ndarray,
        bias_avg: float,
        bias_std: float,
    ) -> None:
        self.r2_greater_than_random_pvalue = r2_greater_than_random_pvalue
        self.r2_greater_than_random_test_statistic = (
            r2_greater_than_random_test_statistic
        )

        self.r2_greater_than_random_degrees_of_freedom = (
            r2_greater_than_random_degrees_of_freedom
        )
        self.coefficient_of_determination_avg = coefficient_of_determination_avg
        self.coefficient_of_determination_std = coefficient_of_determination_std
        self.coefficients_avg = coefficients_avg
        self.coefficients_std = coefficients_std
        self.bias_avg = bias_avg
        self.bias_std = bias_std

    @staticmethod
    def __display_float(x: float) -> str:
        return str(np.array([x]))[1:-1]

    def __str__(self) -> str:
        with np.printoptions(precision=5, suppress=True):
            result = (
                f"           Coefficient of Determination Avg: {self.__display_float(self.coefficient_of_determination_avg)}"
                f"     Std: {self.__display_float(self.coefficient_of_determination_std)}\n"
                f"Coefficient of Determination T-Test p-value: {self.r2_greater_than_random_pvalue}\n"
            )
            if self.coefficients_avg.shape[0] <= 8:
                result += (
                    f"                           Coefficients Avg: {self.coefficients_avg}\n"
                    f"                           Coefficients Std: {self.coefficients_std}\n"
                    f"                                   Bias Avg: {self.__display_float(self.bias_avg)} "
                    f"Std: {self.__display_float(self.bias_std)}"
                )
            else:
                result += (
                    f"Coefficients Avg:\n{self.coefficients_avg}\n"
                    f"Coefficients Std:\n{self.coefficients_std}\n"
                    f"Bias Avg: {self.__display_float(self.bias_avg)} "
                    f"Std: {self.__display_float(self.bias_std)}"
                )
            return result


class CAVSignTestR2Data:
    def __init__(
        self,
        r2_greater_than_random_pvalue: float,
        num_random_r2_greater: int,
        num_random_r2_less: int,
        coefficient_of_determination: float,
        coefficients: np.ndarray,
        bias: float,
    ) -> None:
        self.r2_greater_than_random_pvalue = r2_greater_than_random_pvalue
        self.num_random_r2_greater = num_random_r2_greater

        self.num_random_r2_less = num_random_r2_less
        self.coefficient_of_determination = coefficient_of_determination
        self.coefficients = coefficients
        self.bias = bias

    @staticmethod
    def __display_float(x: float) -> str:
        return str(np.array([x]))[1:-1]

    def __str__(self) -> str:
        with np.printoptions(precision=5, suppress=True):
            result = (
                f"R²: {self.__display_float(self.coefficient_of_determination)}"
                f"  P-val: {self.__display_float(self.r2_greater_than_random_pvalue)}"
                f" [{self.__display_float(self.num_random_r2_less)}"
                f", {self.__display_float(self.num_random_r2_greater)}]\n"
            )

            result += (
                f"Coefficients:\n{self.coefficients}\n"
                f"Bias: {self.__display_float(self.bias)} "
            )
            return result


class GradientTracker:
    """
    Extracts gradients for specified layers of a neural network.

    Registers gradient computation to capture layer gradients with respect to the output.
    This class returns the gradients of the specified layers when called.

    Parameters
    ----------
    model : IndexableModule
        The neural network model.
    layer_names : list[str]
        List of layer names to track gradients for.

    Returns
    ------
    dict[str, np.ndarray]
        A dictionary mapping layer names to their gradient arrays.

    Notes
    -----
    - Assumes each layer name is unique within the model.
    - Uses `LayerGradientXActivation` for gradient computation.

    Examples
    --------
    >>> model = IndexableModule()
    >>> layer_names = ['layer1', 'layer2']
    >>> tracker = GradientTracker(model, layer_names)
    >>> input_data = torch.randn(1, 10)
    >>> gradients = tracker(input_data)
    >>> print(gradients.keys())
    dict_keys(['layer1', 'layer2'])
    """

    def __init__(self, model: IndexableModule, layer_names: list[str]) -> None:
        self.model = model
        self.layer_names = layer_names

        # Set the model to evaluation mode to make sure batch norm and dropout layers work appropriatly
        # Parameters are not updated, so they do not need to be frozen
        self.model.eval()

        # Initialize the gradient calculator
        self.__layer_attribution_method = LayerGradientXActivation(
            model, [model[name] for name in layer_names], multiply_by_inputs=False
        )

    def __call__(
        self,
        input_data: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> dict[str, np.ndarray]:

        assert not self.model.training
        self.__activations: dict[str, np.ndarray] = {}

        # Pass the input through the model once (And saving the activations via hooks)
        # Then calculate the gradients of the given layers
        gradient_list: list[torch.Tensor] = self.__layer_attribution_method.attribute(
            input_data, attribute_to_layer_input=False
        )
        # Convert the gradient output to a dictionary on the same format as self.__activations
        gradients: dict[str, np.ndarray] = {}
        for i, gradient in enumerate(gradient_list):
            gradients[self.layer_names[i]] = gradient.detach().cpu().numpy()

        return gradients


class ActivationTracker:
    """
    Extracts activations for specified layers of a neural network.

    Registers forward hooks to capture layer activations.
    This class should be used as a context manager.

    Parameters
    ----------
    model : IndexableModule
        The neural network model.
    layer_names : list[str]
        List of layer names to track activations for.

    Returns
    ------
    dict[str, np.ndarray]
        A dictionary mapping layer names to their activation arrays.

    Notes
    -----
    - Assumes each layer name is unique within the model.
    - Hooks are automatically removed when exiting the context.

    Examples
    --------
    >>> model = IndexableModule()
    >>> layer_names = ['layer1', 'layer2']
    >>> with ActivationTracker(model, layer_names) as tracker:
    ...     input_data = torch.randn(1, 10)
    ...     activations = tracker(input_data)
    ...     print(activations.keys())
    dict_keys(['layer1', 'layer2'])
    """

    def __init__(
        self,
        model: IndexableModule,
        layer_names: list[str],
        batch_size: int = torch.inf,
    ) -> None:

        self.model = model
        self.layer_names = layer_names
        self.__activations: dict[str, np.ndarray] = {}
        self.__handles: dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.batch_size = batch_size

        # Set the model to evaluation mode to make sure batch norm and dropout layers work appropriatly
        # Parameters are not updated, so they do not need to be frozen
        self.model.eval()

        # Register hooks to capture the output of the specified layers
        # The outputs of the layers are stored in self.__activations
        # Assumes each layer is only used once in the model,
        # so does not work for the 2D conv layer in our LSTM model
        for layer_name in self.layer_names:

            # We take the layer_name as input to force python to save the current value of layer_name
            # as a default parameter value, rather than just referencing the layer_name variable directly
            # which will end up being the last value in layer_names by the end of the loop
            def hook(
                module: torch.nn.Module,
                input: tuple[torch.Tensor],
                output: torch.Tensor,
                layer_name=layer_name,
            ):
                self.__activations[layer_name] = output.detach().cpu().numpy()

            self.__handles[layer_name] = model[layer_name].register_forward_hook(hook)

    def __enter__(self) -> "ActivationTracker":
        return self

    def __call__(
        self,
        input_data: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> dict[str, np.ndarray]:

        assert not self.model.training
        self.__activations: dict[str, np.ndarray] = {}

        total_activations: dict[str, list[np.ndarray]] = {
            layer_name: [] for layer_name in self.layer_names
        }

        def update_total_activations():
            for name in self.layer_names:
                total_activations[name].append(np.copy(self.__activations[name]))

        # Pass the input through and discard result
        # Also split up the input into batches to avoid running out of memory
        if isinstance(input_data, torch.Tensor):
            n_full_batches = int(input_data.shape[0] // self.batch_size)

            for i in range(n_full_batches):
                self.model(input_data[i * self.batch_size : (i + 1) * self.batch_size])
                update_total_activations()
            self.model(input_data[n_full_batches * self.batch_size :])
            update_total_activations()

        else:
            n_full_batches = int(input_data[0].shape[0] // self.batch_size)
            num_input_tensors = len(input_data)

            for i in range(n_full_batches):
                reduced_input = []
                for j in range(num_input_tensors):
                    reduced_input.append(
                        input_data[j][i * self.batch_size : (i + 1) * self.batch_size]
                    )
                self.model(*reduced_input)
                update_total_activations()

            reduced_input = []
            for j in range(num_input_tensors):
                reduced_input.append(input_data[j][n_full_batches * self.batch_size :])
            self.model(*reduced_input)
            update_total_activations()

        for layer_name, activations in total_activations.items():
            total_activations[layer_name] = np.concatenate(activations, axis=0)

        return total_activations

    def finalize(self):
        # Remove the hooks to prevent side effects
        for handle in self.__handles.values():
            handle.remove()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()


def __train_test_split(
    x: np.ndarray, y: np.ndarray, test_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits arrays or matrices into random train and test subsets. Assumes

    Parameters
    ----------
    x : numpy.ndarray
        Input data, where rows are samples and columns are features.
    y : numpy.ndarray
        Target data, typically a 1D array corresponding to the labels for x.
    test_size : float, optional
        The proportion of the dataset to include in the test split. Must be in range [0-1]

    Returns
    -------
    x_train : numpy.ndarray
        Training data features.
    y_train : numpy.ndarray
        Training data labels.
    x_test : numpy.ndarray
        Testing data features.
    y_test : numpy.ndarray
        Testing data labels.
    """
    length = x.shape[0]
    p = np.random.permutation(length)
    x, y = x[p], y[p]
    split_index = int(length * test_size)
    x_test, y_test = x[:split_index], y[:split_index]
    x_train, y_train = x[split_index:], y[split_index:]
    return x_train, y_train, x_test, y_test


def compute_cav(
    activations_train: np.ndarray,
    targets_train: np.ndarray,
    activations_test: np.ndarray,
    targets_test: np.ndarray,
    l2_regularization: float,
    use_sgd: bool,
) -> tuple[np.ndarray, float, float, np.ndarray, float, float]:
    """
    Trains a linear regressor on the provided data and returns the coefficients.
    Returns a CAV. Here, a is number of 'images', b is the number of nodes in the NN layer.

    Parameters
    ----------
    activations : numpy.ndarray
        The input data for the regressor, in the form of activations of a layer in a neural network
    targets : numpy.ndarray
        The target data for the regressor, in the form of concept 'labels' (float)

    Returns
    -------
    tuple[np.ndarray, float, float]
        The coefficients, intercept and coeficient of determination of the trained linear regressor
    """

    # NOTE: the CAV is the vector pointing from random values to the concept values. Therefore, only the
    # direction of the vector is important, not the bias of the model. Nevertheless, the bias is included in the
    # model, so that the direction of the vector can be calculated correctly. The bias is not used in the final CAV.

    if use_sgd and l2_regularization == 0.0:
        model = SGDRegressor(
            penalty=None,
            alpha=0.0,
            learning_rate="adaptive",
            max_iter=1000,
            tol=0.001,
            eta0=0.001,
            early_stopping=True,
        )
    elif use_sgd and l2_regularization > 0.0:
        model = SGDRegressor(
            penalty="l2",
            alpha=l2_regularization,
            learning_rate="adaptive",
            max_iter=1000,
            tol=0.001,
            eta0=0.001,
            early_stopping=True,
        )
    elif (not use_sgd) and l2_regularization == 0.0:
        # Switching to LinearRegression on zero for numerical stability reasons
        model = LinearRegression()
    elif (not use_sgd) and l2_regularization > 0.0:
        model = Ridge(alpha=l2_regularization)
    else:
        raise ValueError("l2_regularization is negative")
    model.fit(activations_train, targets_train)

    # NOTE: The condition number and effective rank of the activations matrix does not help us uncover when there are multiple solutions
    # However rank of the matrix in the network does say something about it

    # Singular Value Analysis
    # _, s, _ = np.linalg.svd(activations)
    # print(s.shape)
    # ratio = np.max(s) / np.min(s)
    # print(f"Singular value ratio: {ratio}")
    # print(s)
    # # Effective Rank:
    # print(np.sum(s > np.max(s) * 1e-10))

    # residuals = targets - model.predict(activations)
    # print(f"Residuals: {np.linalg.norm(residuals)}")
    # print(f"Score: {model.score(activations, targets)}")

    model_score = model.score(activations_test, targets_test)

    # with np.printoptions(precision=5, suppress=True):
    #     print(
    #         f"{model_sgd.n_iter_}, {difference:.5f}, {model_score - model_sgd_score:.5f}, \n{model_sgd.coef_}, {model.coef_}"
    #     )

    # TCAV does not use the bias term, only the direction of the vector is important

    return (model.coef_, model.intercept_, model_score)


def compute_tcav_score_sign(
    sensitivity: np.ndarray,
) -> float:
    """
    Computes the TCAV score based on the sign of the sensitivities.

    Parameters
    ----------
    sensitivity : numpy.ndarray
        The sensitivity of the concept with respect to the layer.

    Returns
    -------
    float
        The TCAV score.
    """
    return float(
        (np.sum(sensitivity > 0) + 0.5 * np.sum(sensitivity == 0))
        / sensitivity.shape[0]
    )


def compute_tcav_score_magnitude(
    sensitivity: np.ndarray,
) -> float:
    """
    Computes the TCAV score based on the magnitude of the sensitivities.

    Parameters
    ----------
    sensitivity : numpy.ndarray
        The sensitivity of the concept with respect to the layer.

    Returns
    -------
    float
        The TCAV score.
    """
    return np.mean(sensitivity)


def __check_input(input_data: torch.Tensor | tuple[torch.Tensor, ...]) -> int:
    if isinstance(input_data, tuple):
        data_length = input_data[0].shape[0]
        for input_tensor in input_data:
            assert (
                input_tensor.shape[0] == data_length
            ), "Input data must have the same number of rows"
    elif isinstance(input_data, torch.Tensor):
        data_length = input_data.shape[0]
    else:
        raise ValueError("input_data has wrong type")

    return data_length


def calculate_cavs(
    activations_tracker: ActivationTracker,
    layer_names: list[str],
    input_data_cav_train: torch.Tensor | tuple[torch.Tensor, ...],
    concepts_target_data_train: dict[str, np.ndarray],
    input_data_cav_test: torch.Tensor | tuple[torch.Tensor, ...],
    concepts_target_data_test: dict[str, np.ndarray],
    l2_regularization: float,
    use_sgd: bool,
    normalize_cav: bool,
    verbose: bool = False,
) -> dict[str, dict[str, CAVData]]:
    """Calculates Concept Activation Vectors (CAVs) for specified layers.

    This function computes CAVs by training linear least squares regressors to predict
    the given concepts using the activations from the given layers.

    Parameters
    ----------
    activations_tracker : ActivationTracker
        An ActivationTracker instance to extract activations.
    layer_names : list[str]
        A list of layer names for which to compute CAVs.
    input_data_cav_training : torch.Tensor | tuple[torch.Tensor, ...]
        The model inputs for training the CAVs,
        either a single tensor or a tuple of tensors depending on what the model expects.
    concepts_target_data : dict[str, np.ndarray]
        A dictionary mapping concept names to target data arrays. Each array must have the
        same number of rows as the input data.

    Returns
    -------
    dict[str, dict[str, CAVData]]
        A dictionary mapping concept names to dictionaries mapping layer names to CAV Data,
        containing the coefficients and bias of the trained linear models.

    Raises
    ------
    AssertionError
        If the input data tensors or target data arrays do not have the same number of rows.
    """

    data_length = __check_input(input_data_cav_train)
    for target_tensor in concepts_target_data_train.values():
        assert (
            target_tensor.shape[0] == data_length
        ), "Target data must have same number of rows as input data"

    # Use the given input data and ensure
    if verbose:
        print("Calculating activations")
    activations_train = activations_tracker(input_data_cav_train)
    activations_test = activations_tracker(input_data_cav_test)
    result: dict[str, dict[str, CAVData]] = {}

    for i, (
        (concept_name_train, concept_target_data_train),
        (concept_name_test, concept_target_data_test),
    ) in enumerate(
        zip(concepts_target_data_train.items(), concepts_target_data_test.items())
    ):
        assert concept_name_train == concept_name_test

        if verbose:
            print(
                f"[{i}/{len(concepts_target_data_train.keys())}] {concept_name_train}"
            )

        result[concept_name_train] = {}

        for layer_name in layer_names:

            # Train the linear regressor on these activations
            (coefficients, bias, coefficient_of_determination) = compute_cav(
                activations_train[layer_name],
                concept_target_data_train,
                activations_test[layer_name],
                concept_target_data_test,
                l2_regularization,
                use_sgd,
            )

            result[concept_name_train][layer_name] = CAVData(
                coefficients,
                bias,
                coefficient_of_determination,
                None,
                None,
                None,
            )
            if normalize_cav:
                result[concept_name_train][layer_name].normalize()

    return result


def calculate_sensitivity(
    gradient_tracker: GradientTracker,
    cavs: dict[str, dict[str, CAVData]],
    input_data_sensitivy_testing: torch.Tensor | tuple[torch.Tensor, ...],
) -> None:
    """Calculates the sensitivity of the model's output to Concept Activation Vectors (CAVs).

    This function computes the sensitivity by calculating the dot product of gradients and CAVs,
    and extracting the sign and magnitude of the results.

    Parameters
    ----------
    gradient_tracker : GradientTracker
        A GradientTracker instance to extract gradients.
    cavs : dict[str, dict[str, CAVData]]
        A dictionary mapping concept names to dictionaries mapping layer names to CAV Data,
        containing the CAVs for each layer and concept.
    input_data_sensitivy_testing : torch.Tensor | tuple[torch.Tensor, ...]
        The model inputs for which to calculate the sensitivity,
        either a single tensor or a tuple of tensors depending on what the model expects.

    Returns
    -------
    None
        This function modifies the `cavs` dictionary in-place, adding sensitivity and TCAV scores.

    Raises
    ------
    AssertionError
        If the input data is not a torch.Tensor or a tuple of torch.Tensors.
    """

    __check_input(input_data_sensitivy_testing)

    concept_names = list(cavs.keys())
    layer_names = list(cavs[concept_names[0]].keys())

    # Use the given input data and ensure
    gradients = gradient_tracker(input_data_sensitivy_testing)

    # Updating the given cavs dict:
    for concept_name in concept_names:
        for layer_name in layer_names:
            coefficients = cavs[concept_name][layer_name].coefficients
            # Compute how sensitive the network output is to nudges in the directino of the CAV
            # i.e. in the direction of the coefficient vector, i.e. the directional derivative
            sensitivities = np.dot(gradients[layer_name], coefficients)

            cavs[concept_name][layer_name].tcav_sign_score = compute_tcav_score_sign(
                sensitivities
            )
            cavs[concept_name][layer_name].tcav_magnitude_score = (
                compute_tcav_score_magnitude(sensitivities)
            )
            cavs[concept_name][layer_name].sensitivities = sensitivities


def calculate_sensitivity_specific_cav(
    model: IndexableModule,
    input_data_sensitivy_testing: torch.Tensor | tuple[torch.Tensor, ...],
    layer_name: str,
    cav_coefficients: np.ndarray,
    cav_bias: float = None,
    cav_coefficient_of_determination: float = None,
) -> CAVData:
    gradient_tracker = GradientTracker(model, [layer_name])

    cav_data = CAVData(
        cav_coefficients,
        cav_bias,
        cav_coefficient_of_determination,
        None,
        None,
        None,
    )
    result = {"concept": {layer_name: cav_data}}
    calculate_sensitivity(
        gradient_tracker,
        result,
        input_data_sensitivy_testing,
    )
    return result["concept"][layer_name]


def bonferroni_correction(p_value: float, n_tests: int):
    # After this correction, rejecting null-hypothesises above a treshold 0.05 gives a
    # 0.05 chance of at least 1 null-hypothesis being falsly labeled as true
    return min(1.0, float(p_value) * n_tests)


def statistical_significance_t_test_cavs(
    model: IndexableModule,
    layer_names: list[str],
    input_data_cav_training: list[torch.Tensor | tuple[torch.Tensor, ...]],
    input_data_sensitivy_testing: torch.Tensor | tuple[torch.Tensor, ...],
    concepts_target_datas: dict[str, list[np.ndarray]],
    random_key: str,
    test_size: float,
    l2_regularization: float,
    use_sgd: bool,
    normalize_cav: bool,
) -> tuple[
    dict[str, dict[str, CAVStatsData]], dict[str, dict[str, list[CAVData]]], int
]:

    # Problems with TCAV from google:
    # 1. No bonferroni correction, just using results from scipy.stats.ttest_ind
    # 2. Assuming independence between random concepts and concepts under investigation (Maybe correct)
    # 3. Assuming tcav scores are normally distributed
    # 4. Assuming equal variance, but the random concepts may have a higher variance in TCAV scores

    # TCAV Original implementation:
    # Create one group of images for each concept, like striped, these are the positive class.
    # Create x groups of random images, each group acts as a different negative class for training of CAVs.
    # Create x different concept CAVs using the same positive class of the concept and a different negative random class.
    # Create one group of random images that is the counterpart to the different concepts.
    # Create another x different random CAVs using the random counterpart as the positive class and a different negative random class.
    # Use statistical significance testing to check that the mean of the concept CAVs is different from the mean of the random CAVs.
    # NOTE: They treat the negative CAV as special, it should be meaningfully the opposite, and have (1-TCAV_sign_score) of the normal CAV

    num_experiments = len(input_data_cav_training)

    assert num_experiments > 0, "At least 1 experiment"
    for concept_datas in concepts_target_datas.values():
        assert len(concept_datas) == num_experiments

    # The shapes of the ndarrays in each concept target and input data is checked in the calculate_cav_and_sensitivity() call

    # Our implementation (TODO):
    # Instead of using x groups of random images, we are given x groups of input_data, concepts_target_data pairs
    # x is num_experiments
    # Create x different CAVs for the different inputs (for each of the layers and concepts given)
    # Also create x different random CAVs using the random_counterpart (for each layer)
    # random_counterpart takes the same form as a concept in concepts_target_datas
    # Use statistical significance testing on to test whether the mean TCAV_sign_score of the
    # randcom_counterpart (should be 0.5) is different from each of the tested concepts
    # For each of the concepts, check how similar the CAVs are and print the mean and std for each coeficient
    # return a dict of concepts, with each concept having its mean TCAV_sign_score, the p_value assigned to it, and
    # a representative CAV, perhaps this representative can be computed by merging the input data and targets?
    concept_names = [*concepts_target_datas.keys()]
    assert random_key in concept_names

    gradient_tracker = GradientTracker(model, layer_names)
    with ActivationTracker(model, layer_names) as activation_tracker:

        # Train CAVs for all the concepts and all the input data (Including random concept):
        results: dict[str, dict[str, list[CAVData]]] = {}
        for concept_name in concept_names:
            results[concept_name] = {}
            for layer_name in layer_names:
                results[concept_name][layer_name] = []
        for i in range(num_experiments):
            concepts_target_data: dict[str, np.ndarray] = {}
            for concept_name, concept_datas in concepts_target_datas.items():
                concepts_target_data[concept_name] = concept_datas[i]

            # Calculate the cavs and the sensitivites for those cavs
            cavs_one_experiment = calculate_cavs(
                activation_tracker,
                layer_names,
                input_data_cav_training[i],
                concepts_target_data,
                test_size=test_size,
                l2_regularization=l2_regularization,
                use_sgd=use_sgd,
                normalize_cav=normalize_cav,
            )
            calculate_sensitivity(
                gradient_tracker, cavs_one_experiment, input_data_sensitivy_testing
            )

            for concept_name in concept_names:
                for layer_name in layer_names:
                    results[concept_name][layer_name].append(
                        cavs_one_experiment[concept_name][layer_name]
                    )

    stats_result: dict[str, dict[str, CAVStatsData]] = {}
    for concept_name in concept_names:
        stats_result[concept_name] = {}

    for layer_name in layer_names:
        sign_scores_random = [
            cav_data.tcav_sign_score for cav_data in results[random_key][layer_name]
        ]
        for concept_name in concept_names:

            concept_data = results[concept_name][layer_name]

            # Extract arrays of related data across samples:
            sign_scores_concept = np.array(
                [cav_data.tcav_sign_score for cav_data in concept_data]
            )
            magnitude_scores_concept = np.array(
                [cav_data.tcav_magnitude_score for cav_data in concept_data]
            )
            coefficient_of_determinations_concept = np.array(
                [cav_data.coefficient_of_determination for cav_data in concept_data]
            )
            coefficients_concept = np.array(
                [cav_data.coefficients for cav_data in concept_data]
            )
            biases_concept = np.array([cav_data.bias for cav_data in concept_data])

            # Assumtions:
            # 1. Assuming independence between random concepts and concepts under investigation
            #    * The random concepts are trained on random targets,
            #      and the correlation between the random and real concept is low
            # 2. Assuming tcav sign scores are normally distributed
            #    * Helge: Usually 30 samples is enough to ensure average of a distribution is normal if you
            #      do not have outliers, which we don't as all sign scores are [0,1]

            if concept_name == random_key:
                # Testing the exact same sequence against itself does not make sense
                tcav_sign_is_different_t_test_result = None
            else:
                tcav_sign_is_different_t_test_result = ttest_ind(
                    sign_scores_random,
                    sign_scores_concept,
                    equal_var=False,
                    alternative="two-sided",
                )

            tcav_sign_1samp_t_test_result = ttest_1samp(
                sign_scores_concept,
                0.5,
                alternative="two-sided",
            )

            stats_result[concept_name][layer_name] = CAVStatsData(
                tcav_sign_is_different_t_test_result,
                tcav_sign_1samp_t_test_result,
                np.mean(sign_scores_concept),
                np.std(sign_scores_concept),
                np.mean(magnitude_scores_concept),
                np.std(magnitude_scores_concept),
                np.mean(coefficients_concept, axis=0),
                np.std(coefficients_concept, axis=0),
                np.mean(biases_concept),
                np.std(biases_concept),
                np.mean(coefficient_of_determinations_concept),
                np.std(coefficient_of_determinations_concept),
            )

    # n-tests is the number of layers times the number of non-random concepts
    n_tests = len(layer_names) * (len(concept_names) - 1)
    return stats_result, results, n_tests


def t_test_cavs(
    model: IndexableModule,
    layer_names: list[str],
    input_data_train: list[torch.Tensor | tuple[torch.Tensor, ...]],
    input_data_test: torch.Tensor | tuple[torch.Tensor, ...],
    concepts_target_data_train: dict[str, list[np.ndarray]],
    concepts_target_data_test: dict[str, np.ndarray],
    random_key: str,
    batch_size: int,
    l2_regularization: float,
    use_sgd: bool,
    normalize_cav: bool,
    verbose: int,
) -> tuple[
    dict[str, dict[str, CAVStatsDataR2]], dict[str, dict[str, list[CAVData]]], int
]:

    num_experiments = len(input_data_train)

    assert num_experiments > 0, "At least 1 experiment"
    for concept_datas in concepts_target_data_train.values():
        assert len(concept_datas) == num_experiments

    concept_names = [*concepts_target_data_train.keys()]
    assert random_key in concept_names
    for i in range(len(concept_names)):
        assert list(concepts_target_data_test.keys())[i] == concept_names[i]

    # The shapes of the ndarrays in each concept target and input data is checked in the calculate_cav_and_sensitivity() call

    with ActivationTracker(
        model, layer_names, batch_size=batch_size
    ) as activation_tracker:

        # Train CAVs for all the concepts and all the input data (Including random concept):
        results: dict[str, dict[str, list[CAVData]]] = {}
        for concept_name in concept_names:
            results[concept_name] = {}
            for layer_name in layer_names:
                results[concept_name][layer_name] = []
        for i in range(num_experiments):
            concepts_target_data_train_one_experiment: dict[str, np.ndarray] = {}
            for concept_name in concept_names:
                concepts_target_data_train_one_experiment[concept_name] = (
                    concepts_target_data_train[concept_name][i]
                )

            if verbose > 0:
                print(f"[{'{:2d}'.format(i)}/{num_experiments}]")

            # Calculate the cavs and the sensitivites for those cavs
            cavs_one_experiment = calculate_cavs(
                activation_tracker,
                layer_names,
                input_data_train[i],
                concepts_target_data_train_one_experiment,
                input_data_test,
                concepts_target_data_test,
                l2_regularization=l2_regularization,
                use_sgd=use_sgd,
                normalize_cav=normalize_cav,
                verbose=verbose > 1,
            )

            for concept_name in concept_names:
                for layer_name in layer_names:
                    results[concept_name][layer_name].append(
                        cavs_one_experiment[concept_name][layer_name]
                    )

    stats_result: dict[str, dict[str, CAVStatsDataR2]] = {}
    for concept_name in concept_names:
        stats_result[concept_name] = {}

    for layer_name in layer_names:
        coefficient_of_determinations_random = [
            cav_data.coefficient_of_determination
            for cav_data in results[random_key][layer_name]
        ]
        for concept_name in concept_names:

            concept_data = results[concept_name][layer_name]

            # Extract arrays of related data across samples:
            coefficient_of_determinations_concept = np.array(
                [cav_data.coefficient_of_determination for cav_data in concept_data]
            )
            coefficients_concept = np.array(
                [cav_data.coefficients for cav_data in concept_data]
            )
            biases_concept = np.array([cav_data.bias for cav_data in concept_data])

            # Assumtions:
            # 1. Assuming independence between random concepts and concepts under investigation
            #    * The random concepts are trained on random targets,
            #      TODO: Check correlation
            # 2. Assuming ttest statistic (avg of R^2 scores) are normally distributed
            #    * Helge: Usually 30 samples is enough to ensure average of a distribution is normal if you
            #      do not have outliers, which we don't as all sign scores are [0,1]

            if concept_name == random_key:
                # Testing the exact same sequence against itself does not make sense
                r2_greater_than_random_t_test_result = None
            else:
                r2_greater_than_random_t_test_result = ttest_ind(
                    coefficient_of_determinations_random,
                    coefficient_of_determinations_concept,
                    equal_var=True,
                    alternative="less",
                )

            if r2_greater_than_random_t_test_result is None:
                stats_result[concept_name][layer_name] = CAVStatsDataR2(
                    None,
                    None,
                    None,
                    np.mean(coefficient_of_determinations_concept),
                    np.std(coefficient_of_determinations_concept),
                    np.mean(coefficients_concept, axis=0),
                    np.std(coefficients_concept, axis=0),
                    np.mean(biases_concept),
                    np.std(biases_concept),
                )
            else:
                stats_result[concept_name][layer_name] = CAVStatsDataR2(
                    r2_greater_than_random_t_test_result.pvalue,
                    r2_greater_than_random_t_test_result.statistic,
                    r2_greater_than_random_t_test_result.df,
                    np.mean(coefficient_of_determinations_concept),
                    np.std(coefficient_of_determinations_concept),
                    np.mean(coefficients_concept, axis=0),
                    np.std(coefficients_concept, axis=0),
                    np.mean(biases_concept),
                    np.std(biases_concept),
                )

    # n-tests is the number of layers times the number of non-random concepts
    n_tests = len(layer_names) * (len(concept_names) - 1)
    return stats_result, results, n_tests


def sign_test_cavs(
    model: IndexableModule,
    layer_names: list[str],
    input_data_train: torch.Tensor | tuple[torch.Tensor, ...],
    input_data_test: torch.Tensor | tuple[torch.Tensor, ...],
    concepts_target_data_train: dict[str, np.ndarray],
    concepts_target_data_test: dict[str, np.ndarray],
    random_keys: list[str],
    batch_size: int,
    l2_regularization: float,
    use_sgd: bool,
    normalize_cav: bool,
    verbose: bool,
) -> tuple[
    dict[str, dict[str, CAVSignTestR2Data]],
    dict[str, dict[str, list[CAVData]]],
    int,
    dict[str, np.ndarray],
]:

    all_keys = [*concepts_target_data_train.keys()]
    concept_keys = [key for key in all_keys if key not in random_keys]
    assert len(concept_keys) > 0, "At least 1 non-random concept"

    # Checking concepts are in the same order:
    for i in range(len(all_keys)):
        assert list(concepts_target_data_test.keys())[i] == all_keys[i]
    # Checking random_keys are subset of all_keys
    for random_key in random_keys:
        assert random_key in all_keys

    with ActivationTracker(
        model, layer_names, batch_size=batch_size
    ) as activation_tracker:

        # Calculate the cavs and the R^2 score for those cavs
        results = calculate_cavs(
            activation_tracker,
            layer_names,
            input_data_train,
            concepts_target_data_train,
            input_data_test,
            concepts_target_data_test,
            l2_regularization=l2_regularization,
            use_sgd=use_sgd,
            normalize_cav=normalize_cav,
            verbose=verbose,
        )

    stats_result: dict[str, dict[str, CAVSignTestR2Data]] = {
        key: {} for key in concept_keys
    }

    random_r2_scores: dict[str, np.ndarray] = {
        name: np.zeros((len(random_keys)), dtype=np.float32) for name in layer_names
    }
    for layer_name in layer_names:
        for i, random_key in enumerate(random_keys):
            random_r2_scores[layer_name][i] = results[random_key][
                layer_name
            ].coefficient_of_determination

        # random_r2_scores[layer_name].sort(stable=False)

    for concept_key in concept_keys:
        for layer_name in layer_names:

            concept_data = results[concept_key][layer_name]

            # Calculating how many
            # insertion_index = np.searchsorted(
            #     random_r2_scores[layer_name],
            #     concept_data.coefficient_of_determination,
            #     side="left",
            # )

            num_less, num_greater, total, sign_test_pvalue = (
                one_sample_sign_test_target_greater_than_sample_median(
                    random_r2_scores[layer_name],
                    concept_data.coefficient_of_determination,
                )
            )

            # Not doing the wilcoxon test, it assumes a hypothetical distribution around concept_data.coefficient_of_determination that is symetric
            # wilcoxon_result = wilcoxon(
            #     random_r2_scores[layer_name],
            #     concept_data.coefficient_of_determination,
            #     alternative="less",
            # )

            stats_result[concept_key][layer_name] = CAVSignTestR2Data(
                sign_test_pvalue,
                num_greater,
                num_less,
                concept_data.coefficient_of_determination,
                concept_data.coefficients,
                concept_data.bias,
            )

    return stats_result, results, len(concept_keys) * len(layer_names), random_r2_scores


if __name__ == "__main__":

    # np.random.seed(42)
    # torch.manual_seed(42)

    # Example usage:
    # Assuming `simple_nn` is your pre-trained PyTorch model and `layer_name` is the name of the layer
    # input_data: list of input data for the DL network
    # target_data: list of target data for the regressor (CAV stuff)
    simple_nn = SimpleNN()

    # summary(simple_nn, [(10,), (2,)])

    layer_names = ["fc11", "fc12"]

    # random input data (uses both weather and time data)
    input_data = np.random.rand(100, 10) * 4 - 2
    input_time_data = np.random.rand(100, 2) * 4 - 2

    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    input_tensor_time = torch.tensor(input_time_data, dtype=torch.float32)

    grad_tracker = GradientTracker(simple_nn, layer_names)
    with ActivationTracker(simple_nn, layer_names) as act_tracker:

        cavs = calculate_cavs(
            act_tracker,
            layer_names,
            (input_tensor, input_tensor_time),
            {"random": np.random.rand(100)},
        )
        calculate_sensitivity(grad_tracker, cavs, (input_tensor, input_tensor_time))

    print("hi")

    print(cavs["random"]["fc11"])
