import __fix_relative_imports  # noqa: F401

import numpy as np
import torch
import torch.nn as nn
import copy

import torch.optim as optim
from collections import OrderedDict
import random
import shap

from mscEidalVesetrud.deep_learning.neural_nets import (
    IndexableModule,
)
from mscEidalVesetrud.data_preprocessing.prepare_load_dataset import (
    load_cross_val,
)
from mscEidalVesetrud.global_constants import (
    TRAIN_SIZE,
    VAL_SIZE,
    SEEDS,
    TRAIN_DATA_PATH,
    CONTAINING_FOLDER,
)


def set_seed(seed: int):
    """
    Sets the random seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # remove to avoid global RNG setting
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def regularization_r1(concept_layer_weights, layer_l_output, lambda_1, K):
    """
    Compute the R1 regularization term to encourage coherence of concepts.

    Args:
        concept_layer_weights (torch.Tensor): Shape (m, d), where m is number of concepts,
                                             d is activation dimension.
        layer_l_output (torch.Tensor): Shape (batch_size, d), activations from the layer
                                      before the concept layer.
        lambda_1 (float): Scaling factor for R1 term.
        K (int): Number of top nearest neighbors per concept.

    Returns:
        r1 (torch.Tensor): Scalar, the R1 regularization term scaled by lambda_1.
    """
    batch_size = layer_l_output.shape[0]
    # Adjust K dynamically to be at most the batch size
    K_adjusted = min(K, batch_size)

    # Normalize concept vectors and activations to unit norm
    concept_norm = torch.nn.functional.normalize(
        concept_layer_weights, p=2, dim=1
    )  # Shape: (m, d)
    activation_norm = torch.nn.functional.normalize(
        layer_l_output, p=2, dim=1
    )  # Shape: (batch_size, d)

    # Compute dot products between concepts and activations
    dot_products = torch.matmul(
        activation_norm, concept_norm.t()
    )  # Shape: (batch_size, m)

    # For each concept, select top-K_adjusted nearest neighbors
    top_k_values, _ = torch.topk(
        dot_products, k=K_adjusted, dim=0, largest=True, sorted=False
    )  # Shape: (K_adjusted, m)

    # Sum dot products for top-K_adjusted neighbors across all concepts
    r1_sum = top_k_values.sum()  # Sum over K_adjusted and m

    # Normalize by number of concepts (m) and K_adjusted
    m = concept_layer_weights.shape[0]
    r1 = r1_sum / (m * K_adjusted)  # Use K_adjusted for normalization

    return lambda_1 * r1


def regularization_r2(concept_layer_weights, lambda_2):
    """
    Compute the R2 regularization term to encourage distinctiveness of concepts.

    Args:
        concept_layer_weights (torch.Tensor): Shape (m, d), where m is number of concepts,
                                             d is activation dimension.
        lambda_2 (float): Scaling factor for R2 term.

    Returns:
        r2 (torch.Tensor): Scalar, the R2 regularization term scaled by lambda_2.
    """
    # Normalize concept vectors to unit norm
    concept_norm = torch.nn.functional.normalize(
        concept_layer_weights, p=2, dim=1
    )  # Shape: (m, d)

    # Compute dot products between all pairs of concept vectors
    dot_products = torch.matmul(concept_norm, concept_norm.t())  # Shape: (m, m)

    # Exclude diagonal (self-similarity) by setting it to 0
    mask = torch.ones_like(dot_products) - torch.eye(
        dot_products.shape[0], device=dot_products.device
    )
    off_diagonal = dot_products * mask

    # Sum all off-diagonal elements
    r2_sum = off_diagonal.sum()

    # Normalize by m * (m-1)
    m = concept_layer_weights.shape[0]
    r2 = r2_sum / (m * (m - 1))

    return -lambda_2 * r2  # Negative to minimize similarity


activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()

    return hook


def get_concept_activation(model, data, time_data, concept_idx, pre_caf: bool = True):
    model.eval()
    try:
        with torch.no_grad():
            _ = model(data, time_data)  # Forward pass to populate activations
            if pre_caf:
                concept_activations = activations["concept_layer"][:, concept_idx]
            else:
                concept_activations = activations["CAF"][:, concept_idx]
            # Ensure output is a NumPy array
            return concept_activations.cpu().numpy()
    except Exception as e:
        print(f"Error in get_concept_activation: {e}")
        raise


class CAF(nn.Module):
    """
    Concept Activation Function
    """

    def __init__(self, beta_threshold):
        super(CAF, self).__init__()
        self.beta_threshold = beta_threshold

    def forward(self, x):
        # similarity = torch.where(x >= self.beta_threshold, x, torch.zeros_like(x))
        # similarity = similarity / torch.norm(similarity, p=2, dim=1, keepdim=True)
        similarity = torch.where(x >= self.beta_threshold, x, torch.zeros_like(x))
        norm = torch.norm(similarity, p=2, dim=1, keepdim=True)

        # Create a mask for non-zero values in norm
        non_zero_mask = norm != 0

        # Perform division only where norm is non-zero
        similarity = torch.where(non_zero_mask, similarity / norm, similarity)

        return similarity


def replace_layer_with_multiple(model, layer_name, new_layers):
    """
    Replace a layer in the model's dense section with multiple new layers.
    Args:
        model: The ConvNet3D model.
        layer_name (str): The name of the layer to replace (must be in the dense section).
        new_layers (OrderedDict): The new layers to insert.
    """
    # Convert the model's dense layers to a list of tuples
    dense_layers_list = list(model.dense.named_children())

    # Find the index of the layer to be replaced
    layer_index = next(
        i for i, (name, _) in enumerate(dense_layers_list) if name == layer_name
    )

    # Insert new layers after the specified layer (not replacing it, just adding)
    layer_index += 1  # Insert after the named layer
    for i, (new_layer_name, new_layer) in enumerate(new_layers.items()):
        dense_layers_list.insert(layer_index + i, (new_layer_name, new_layer))

    # Update the model's dense sequential module
    model.dense = nn.Sequential(OrderedDict(dense_layers_list))

    # Update all_layers to reflect the changes
    model.all_layers = OrderedDict(
        list(model.conv.named_children()) + list(model.dense.named_children())
    )


def add_concept_layer(
    model: IndexableModule,
    layer_name: tuple[str] | tuple[str, str],
    concept_layer_size: int,
    function_g_size: list[int],
    activation_function: nn.Module = nn.SiLU,
    final_activation_function: nn.Module = nn.SiLU,
    batchnorm: bool = True,
    beta_threshold: float = 0.0,
    p_dropout: float = 0.0,
    seed: int = 42,
    freeze_old_parameters: bool = True,
) -> tuple[IndexableModule, nn.Module]:
    """
    Creates and adds a concept layer in the given model, directly after the layer with the given name.
    Function g reconstructs the output of the concept layer, and is placed directly after the concept layer.
    NOTE: The layer_name should contain one or two items:
    one: the name of the layer to set the input size of the concept layer, and
    the place where the concept layer should be placed.
    two: (1) the name of the layer to set the input size of the concept layer,
    and (2) the name of the layer where the concept layer should be placed.
    Example: ["linear_layer_1", "activation_layer_1"], here, "linear_layer_1" defines
    the input size of the concept layer, but the concept layer is placed after the
    "activation_layer_1" layer, and thus takes the output of the activation layer as input.
    NOTE: For expected behavior, the final activation function should be the same as the one used in the model.
    Args:
        model (IndexableModule): The model containing the layer to be used.
        layer_name (tuple[str] | tuple[str, str]): The first item is the name of the layer to set the input size of the concept layer, the second item is the input to the concept layer.
        concept_layer_size (int): The size of the (new) concept layer.
        function_g_size (list[int]): A list specifying the sizes of the layers in the function g network.
        activation_function (nn.Module, optional): The activation function to be used. Defaults to nn.SiLU.
        final_activation_function (nn.Module, optional): The activation function to be used in the final layer. Defaults to nn.SiLU.
        batchnorm (bool, optional): Whether to use batch normalization. Defaults to True.
        beta_threshold (float, optional): The beta threshold for the CAF layer. Defaults to 0.1.
        p_dropout (float, optional): The dropout probability. Defaults to 0.0.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
    Returns:
        IndexableModule: The model with the added concept layer and reconstructive function g.
        nn.Module: The concept layer.
    Notes:
        - The function adds a specific hook to the specified layer in the model to capture its activations, to be used for training.
        - The function adds a new set of layers after the specified layer in the model.
        - The function freezes all old parameters in the model to prevent them from being updated during training of the new parameters.
        - The function returns the modified model and the concept layer.
    """
    set_seed(seed)
    model = copy.deepcopy(model)

    # Determine input size and insertion point
    if len(layer_name) == 1:
        size_layer_name = insertion_layer_name = layer_name[0]
    else:
        size_layer_name, insertion_layer_name = layer_name

    # Find the layer to determine input size
    layer = model[size_layer_name]  # Uses __getitem__
    layer.register_forward_hook(get_activation("layer_l"))

    # Infer dense_layer_size from the first dense layer as a fallback
    dense_layer_size = next(
        layer.out_features
        for name, layer in model.dense.named_children()
        if name.startswith("dense_layer") and isinstance(layer, nn.Linear)
    )

    # Determine input size: use out_features if Linear, else fallback to dense_layer_size
    input_size = (
        layer.out_features if isinstance(layer, nn.Linear) else dense_layer_size
    )

    # Define the concept layer and CAF
    concept_layer = nn.Linear(input_size, concept_layer_size)
    concept_activation = CAF(beta_threshold)  # Assuming CAF is defined elsewhere
    concept_layer.register_forward_hook(
        get_activation("concept_layer")
    )  # Hook for pre-CAF output
    concept_activation.register_forward_hook(
        get_activation("CAF")
    )  # Hook for CAF output

    # Build the new layers
    ordered_layers: OrderedDict[str, nn.Module] = OrderedDict()
    ordered_layers["concept_layer"] = concept_layer
    ordered_layers["CAF"] = concept_activation

    activation_function_name = (
        str(activation_function).split(".")[-1].split("'")[0].lower()
    )
    final_activation_function_name = (
        str(final_activation_function).split(".")[-1].split("'")[0].lower()
    )

    # Create the function g network
    current_size = concept_layer_size
    for i, size in enumerate(function_g_size):
        ordered_layers[f"cl_layer_{i}"] = nn.Linear(current_size, size)
        if batchnorm:
            ordered_layers[f"cl_batchnorm_{i}"] = nn.BatchNorm1d(size)
        ordered_layers[f"cl_{activation_function_name}_{i}"] = activation_function()
        if p_dropout > 0.0:
            ordered_layers[f"cl_dropout_{i}"] = nn.Dropout(p=p_dropout)
        current_size = size

    ordered_layers["cl_final"] = nn.Linear(current_size, dense_layer_size)
    if batchnorm:
        ordered_layers["cl_batchnorm_final"] = nn.BatchNorm1d(dense_layer_size)
    ordered_layers[f"cl_{final_activation_function_name}_final"] = (
        final_activation_function()
    )
    if p_dropout > 0.0:
        ordered_layers["cl_dropout_final"] = nn.Dropout(p=p_dropout)

    if freeze_old_parameters:
        # Freeze all parameters in the model
        for param in model.parameters():
            param.requires_grad = False

    # Add the new layers after the specified insertion point in the dense section
    replace_layer_with_multiple(model, insertion_layer_name, ordered_layers)

    return model, concept_layer


def train_concept_layer(
    model: IndexableModule,
    train_data: torch.Tensor,
    time_data: torch.Tensor,
    train_labels: torch.Tensor,
    concept_layer: nn.Module,
    lambda_1: float = 0.05,
    lambda_2: float = 0.05,
    k_inputs: int = 5,
    num_epochs: int = 1,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    seed: int = 42,
):
    """
    Trains a concept layer in a neural network model with regularization.
    This function trains a neural network model by optimizing the weights of a
    concept layer using mean squared error (MSE) loss and two additional
    regularization terms. The regularization terms are designed to enforce
    sparsity and other constraints on the concept layer weights.
    Args:
        model (IndexableModule): The neural network model containing the concept layer.
        train_data (torch.Tensor): The input training data.
        time_data (torch.Tensor): Additional time-related input data.
        train_labels (torch.Tensor): The ground truth labels for the training data.
        concept_layer (nn.Module): The concept layer whose weights are being optimized.
        lambda_1 (float, optional): Regularization coefficient for the first regularization term. Defaults to 0.05.
        lambda_2 (float, optional): Regularization coefficient for the second regularization term. Defaults to 0.05.
        k_inputs (int, optional): Number of inputs to consider for the first regularization term. Defaults to 5.
        num_epochs (int, optional): Number of training epochs. Defaults to 1000.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    Returns:
        Tuple[IndexableModule, float]: The trained model and the final MSE loss value.
    Notes:
        - The function assumes that the model has a specific hook named "layer_l"
          whose activations are used in the regularization terms.
        - The regularization functions `regularization_r1` and `regularization_r2`
          must be defined elsewhere in the codebase.
        - The optimizer used is Adam with a lr=learning_rate and weight_decay=weight_decay.
    """
    set_seed(seed)
    train_labels = train_labels.float().view(-1, 1)  # Reshape to [batch_size, 1]

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Initial forward pass to populate activations
    model(train_data, time_data)
    if "layer_l" not in activations:
        raise KeyError(
            "Forward hook for 'layer_l' did not capture activations. Check hook registration in add_concept_layer."
        )
    layer_l_output = activations["layer_l"]

    print(
        f"Initial loss: {criterion(model(train_data, time_data), train_labels).item()}"
    )

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(train_data, time_data)
        concept_layer_weights = concept_layer.weight
        mse_loss = criterion(outputs, train_labels)
        reg_1_loss = regularization_r1(
            concept_layer_weights, layer_l_output, lambda_1, k_inputs
        )
        reg_2_loss = regularization_r2(concept_layer_weights, lambda_2)
        loss = mse_loss + reg_1_loss + reg_2_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {mse_loss.item():.6f}, Total Loss: {loss.item():.4f}"
            )

    print("Final loss:", mse_loss.item())
    return model, criterion(model(train_data, time_data), train_labels).item()


def train_one_epoch_cacbe(
    train_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    concept_layer: nn.Module,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    lambda_1: float,
    lambda_2: float,
    k_inputs: int,
):
    mae_metric = nn.L1Loss()

    n_datapoints_train = 0
    running_train_loss = 0
    running_train_mae = 0
    running_total_loss = 0

    model.train()
    for x, x_time, y in train_dataloader:
        batch_size = x.shape[0]
        y = y.unsqueeze(-1)

        optimizer.zero_grad()

        pred = model(x, x_time)

        mse_loss = loss_fn(pred, y)

        concept_layer_weights = concept_layer.weight
        if "layer_l" not in activations:
            raise KeyError("Layer_l activations not captured. Check hook registration.")
        layer_l_output = activations["layer_l"]

        reg_1_loss = regularization_r1(
            concept_layer_weights, layer_l_output, lambda_1, k_inputs
        )
        reg_2_loss = regularization_r2(concept_layer_weights, lambda_2)

        total_loss = mse_loss + reg_1_loss + reg_2_loss

        total_loss.backward()
        optimizer.step()

        running_train_loss += mse_loss.item() * batch_size
        running_total_loss += total_loss.item() * batch_size
        running_train_mae += mae_metric(pred, y).item() * batch_size
        n_datapoints_train += batch_size

    train_loss = running_train_loss / n_datapoints_train
    train_total_loss = running_total_loss / n_datapoints_train
    train_mae = running_train_mae / n_datapoints_train

    return train_loss, train_mae, train_total_loss


def check_layer_weights(
    old_model: IndexableModule, new_model: IndexableModule, verbose=False
):
    for name in old_model.layers:
        if verbose:
            print(name, end="")
        try:
            old_model[name].weight
            if verbose:
                print()
        except AttributeError:
            if verbose:
                print(" - No weight attribute")
            continue

        # assert that all of new_model[name].weight equals all of old_model[name].weight
        assert torch.allclose(
            new_model[name].weight, old_model[name].weight, rtol=1e-05, atol=1e-08
        )
        # assert that all of new_model[name].bias equals all of old_model[name].bias
        assert torch.allclose(
            new_model[name].bias, old_model[name].bias, rtol=1e-05, atol=1e-08
        )


def train_nn(
    model, train_data, time_data, train_labels, epochs=10000, learning_rate=0.001
):
    """
    Trains the given model on the given data and labels.
    Args:
        model (SimpleNN): The model to be trained.
        train_data (torch.Tensor): The training data.
        time_data (torch.Tensor): The time data associated with the model (also train).
        train_labels (torch.Tensor): The labels for the training data.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
    Returns:
        SimpleNN: The trained model.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    print("Initial loss:", criterion(model(train_data, time_data), train_labels).item())

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(train_data, time_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

    print("Final loss:", loss.item())

    return model, criterion(model(train_data, time_data), train_labels).item()


def analyze_concept_importance_with_shap(
    model: IndexableModule,
    # concept_layer: nn.Module,
    train_data: torch.Tensor,
    time_data: torch.Tensor,
    background_samples: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, shap.KernelExplainer]:
    """
    Analyzes the importance of each concept in the concept layer using SHAP values.

    Args:
        model (IndexableModule): The trained model with the concept layer.
        concept_layer (nn.Module): The concept layer whose contributions are analyzed.
        train_data (torch.Tensor): The input training data.
        time_data (torch.Tensor): The time-related input data.
        background_samples (int, optional): Number of background samples for SHAP. Defaults to 100.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, shap.KernelExplainer]: SHAP values, concept activations, and the SHAP explainer.
    """
    set_seed(seed)
    model.eval()

    # Step 1: Extract concept activations (post-CAF) for the entire dataset
    with torch.no_grad():
        _ = model(train_data, time_data)  # Forward pass to populate activations
        concept_activations = activations.get("CAF", None)
        if concept_activations is None:
            raise ValueError(
                "CAF activations not found. Ensure hook is registered correctly."
            )

    # Convert to numpy for SHAP
    concept_activations_np = concept_activations.detach().cpu().numpy()
    # predictions_np = model(train_data, time_data).detach().cpu().numpy()

    # Step 2: Define a prediction function that takes concept activations as input
    def model_predict(concept_inputs):
        """
        Predicts the output given concept activations, applying only the layers after CAF (function g).
        """
        concept_tensor = torch.tensor(concept_inputs, dtype=torch.float32).to(
            train_data.device
        )
        with torch.no_grad():
            current_output = concept_tensor
            start_processing = False
            for name, layer in model.dense.named_children():  # Use model.dense instead
                if name == "CAF":
                    start_processing = True
                    continue
                if start_processing:
                    current_output = layer(current_output)
            return current_output.detach().cpu().numpy()

    # Step 3: Select a background dataset (subset of concept activations)
    background_data = shap.sample(
        concept_activations_np, background_samples, random_state=seed
    )

    # Step 4: Compute SHAP values
    explainer = shap.KernelExplainer(model_predict, background_data)
    shap_values = explainer.shap_values(concept_activations_np, nsamples="auto")

    # Step 5: Return SHAP values and concept activations
    print("SHAP values computed successfully.")
    return shap_values, concept_activations_np, explainer


def find_top_bottom_activations(
    model: IndexableModule,
    data: torch.Tensor,
    time_data: torch.Tensor,
    n: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    model.eval()

    # Step 1: Extract concept activations (pre-CAF and post-CAF)
    with torch.no_grad():
        _ = model(data, time_data)  # Forward pass to populate activations
        concept_activations = activations.get("CAF", None)  # Post-CAF
        pre_caf_activations = activations.get("concept_layer", None)  # Pre-CAF
        if concept_activations is None or pre_caf_activations is None:
            raise ValueError(
                "CAF or concept_layer activations not found. Ensure hooks are registered correctly."
            )

    # Step 2: Convert activations to numpy
    post_caf_np = concept_activations.cpu().numpy()  # Shape: (n_samples, n_concepts)
    pre_caf_np = pre_caf_activations.cpu().numpy()  # Shape: (n_samples, n_concepts)

    # Step 3: Find top and bottom n indices for each concept
    _, n_concepts = post_caf_np.shape
    top_n_indices = np.zeros((n_concepts, n), dtype=int)
    bottom_n_indices = np.zeros((n_concepts, n), dtype=int)

    for concept_idx in range(n_concepts):
        # Top n from post-CAF activations
        post_caf_values = post_caf_np[:, concept_idx]
        top_n_indices[concept_idx] = np.argpartition(-post_caf_values, n - 1)[:n]
        top_n_indices[concept_idx] = top_n_indices[concept_idx][
            np.argsort(-post_caf_values[top_n_indices[concept_idx]])
        ]

        # Bottom n from pre-CAF activations
        pre_caf_values = pre_caf_np[:, concept_idx]
        bottom_n_indices[concept_idx] = np.argpartition(pre_caf_values, n - 1)[:n]
        bottom_n_indices[concept_idx] = bottom_n_indices[concept_idx][
            np.argsort(pre_caf_values[bottom_n_indices[concept_idx]])
        ]

    return top_n_indices, bottom_n_indices
