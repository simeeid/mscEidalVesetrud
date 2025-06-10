import __fix_relative_imports  # noqa: F401
import torch
import torch.nn as nn
import wandb
import numpy as np
from mscEidalVesetrud.data_preprocessing.prepare_load_dataset import (
    load_cross_val,
)
from mscEidalVesetrud.deep_learning.neural_nets import (
    IndexableModule,
    DenseNN,
    ConvNet3D,
    LSTMConv2D,
)
from mscEidalVesetrud.global_constants import (
    TRAIN_SIZE,
    VAL_SIZE,
    TRAIN_DATA_PATH,
    SEEDS,
    MODEL_FOLDER,
)
from mscEidalVesetrud.complete_concepts.complete_concepts import (
    add_concept_layer,
    train_one_epoch_cacbe,
)

# Requires logging in with the command "wandb login"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


def train_one_epoch(
    train_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn,
    optimizer: torch.optim.Optimizer,
):

    mae_metric = nn.L1Loss()

    # Train on one epoch
    n_datapoints_train = 0
    running_train_loss = 0
    running_train_mae = 0
    model.train()
    for x, x_time, y in train_dataloader:
        # Adjusting for batch size because the last batch will be of a different size
        batch_size = x.shape[0]

        # Assumes y is a 1d array, and makes it a 2d array that matches the output of model(x)
        y = y.unsqueeze(-1)

        optimizer.zero_grad()

        pred = model(x, x_time)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * batch_size
        running_train_mae += mae_metric(pred, y).item() * batch_size
        n_datapoints_train += batch_size

    train_loss = running_train_loss / n_datapoints_train
    train_mae = running_train_mae / n_datapoints_train

    return train_loss, train_mae


def eval_one_epoch(
    val_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn,
    return_pred: bool = False,
):
    mae_metric = nn.L1Loss()

    # Test on validation dataset:
    model.eval()
    n_datapoints_val = 0
    running_val_loss = 0
    running_val_mae = 0

    all_preds = []
    for x, x_time, y in val_dataloader:
        # Adjusting for batch size because the last batch will be of a different size
        batch_size = x.shape[0]
        y = y.unsqueeze(-1)

        pred = model(x, x_time)

        running_val_loss += loss_fn(pred, y).item() * batch_size
        running_val_mae += mae_metric(pred, y).item() * batch_size
        n_datapoints_val += batch_size

        all_preds.append(pred.detach().cpu().numpy())

    val_loss = running_val_loss / n_datapoints_val
    val_mae = running_val_mae / n_datapoints_val

    if return_pred:
        return val_loss, val_mae, all_preds

    return val_loss, val_mae


def save_checkpoint(
    save_path: str,
    model: IndexableModule,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    fold: int,
    loss: float,
    cl_config: dict | None = None,
):
    save_dict = {
        "model_type": model.class_name,
        "model_config": model.configuration,
        "model_state_dict": model.state_dict(),
        "optimizer_class_name": str(optimizer.__class__).split(".")[-1].split("'")[0],
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_class_name": (
            str(scheduler.__class__).split(".")[-1].split("'")[0]
            if scheduler is not None
            else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "epoch": epoch,
        "fold": fold,
        "loss": loss,
    }
    if cl_config:
        save_dict["concept_layer_name"] = cl_config["concept_layer_name"]
        save_dict["concept_layer_size"] = cl_config["concept_layer_size"]
        save_dict["function_g_sizes"] = cl_config["function_g_sizes"]
        save_dict["batch_norm"] = cl_config["batch_norm"]
        save_dict["beta_threshold"] = cl_config["beta_threshold"]
        save_dict["cacbe_dropout"] = cl_config["cacbe_dropout"]
        save_dict["seed"] = cl_config["seed"]
        save_dict["freeze_old_parameters"] = cl_config["freeze_old_parameters"]

    torch.save(save_dict, save_path)


def load_checkpoint(
    save_path: str,
    device: torch.device,
    force_final_activation: None | str = None,
    cl_config: dict | None = None,
) -> tuple[
    IndexableModule,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler | None,
    int,
    int,
    float,
]:
    save_dict = torch.load(save_path, map_location=device)

    model_config = save_dict["model_config"]
    if force_final_activation is not None:
        model_config["final_activation"] = force_final_activation

    match save_dict["model_type"]:
        case "dense":
            model = DenseNN(**model_config).to(device)
        case "cnn":
            model = ConvNet3D(**model_config).to(device)
        case "lstm":
            model = LSTMConv2D(**model_config).to(device)
        case _:
            raise AssertionError()

    if cl_config:
        # Add CACBE layers
        model, _ = add_concept_layer(
            model=model,
            layer_name=save_dict["concept_layer_name"],
            concept_layer_size=save_dict["concept_layer_size"],
            function_g_size=save_dict["function_g_sizes"],
            activation_function=nn.SiLU,
            final_activation_function=nn.SiLU,
            batchnorm=save_dict["batch_norm"],
            beta_threshold=save_dict["beta_threshold"],
            p_dropout=save_dict["cacbe_dropout"],
            seed=save_dict["seed"],
            freeze_old_parameters=save_dict["freeze_old_parameters"],
        )

    # Load the model state dict
    model.load_state_dict(save_dict["model_state_dict"])

    if not cl_config:
        # Use the saved class name (as a string) to look up the class with the same name in torch.optim
        # Works for AdamW and SGD for example
        optimizer_class = getattr(torch.optim, save_dict["optimizer_class_name"])
        optimizer: torch.optim.Optimizer = optimizer_class(model.parameters())
    else:
        # Create optimizer with only trainable parameters (requires_grad=True)
        optimizer_class = getattr(torch.optim, save_dict["optimizer_class_name"])
        optimizer: torch.optim.Optimizer = optimizer_class(
            filter(
                lambda p: p.requires_grad, model.parameters()
            )  # Only trainable params
        )
    # The loading of the state dict sets the correct hyperparameters
    optimizer.load_state_dict(save_dict["optimizer_state_dict"])

    # Load scheduler
    if save_dict["scheduler_class_name"] is not None:
        scheduler_class = getattr(
            torch.optim.lr_scheduler, save_dict["scheduler_class_name"]
        )
        scheduler: torch.optim.lr_scheduler.LRScheduler = scheduler_class(optimizer)
        scheduler.load_state_dict(save_dict["scheduler_state_dict"])
    else:
        scheduler = None

    # model.eval()

    return (
        model,
        optimizer,
        scheduler,
        save_dict["epoch"],
        save_dict["fold"],
        save_dict["loss"],
    )


def main(config: dict):

    np.random.seed(SEEDS[0])
    torch.manual_seed(SEEDS[0])

    run = wandb.init()

    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    n_epochs = config["n_epochs"]
    weight_decay = config["weight_decay"]

    datasets = load_cross_val(TRAIN_DATA_PATH, TRAIN_SIZE, VAL_SIZE)

    train_loss_history = np.zeros((n_epochs, datasets.n_splits))
    train_mae_history = np.zeros((n_epochs, datasets.n_splits))
    val_loss_history = np.zeros((n_epochs, datasets.n_splits))
    val_mae_history = np.zeros((n_epochs, datasets.n_splits))

    for fold, (train_data, val_data) in enumerate(
        datasets.split(scale=True, device=device)
    ):

        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_data, batch_size=batch_size, shuffle=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset=val_data, batch_size=batch_size, shuffle=True
        )

        # model = DenseNN(
        #     config["layer_size"],
        #     config["n_layers"],
        #     config["p_dropout"],
        #     final_activation="hardsigmoid",
        # ).to(device)
        model = ConvNet3D(
            config["conv_channels"],
            config["dense_layer_size"],
            config["n_dense_layers"],
            config["p_dropout"],
            config["architecture"],
            final_activation="hardsigmoid",
        ).to(device)
        # model = LSTMConv2D(
        #     config["conv_channels"],
        #     config["lstm_size"],
        #     config["n_lstm_layers"],
        #     config["dense_layer_size"],
        #     config["n_dense_layers"],
        #     config["p_dropout"],
        #     final_activation="hardsigmoid",
        # ).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.25, patience=10, threshold=0.0001
        )

        # Saving best mae
        best_val_mae = np.inf
        save_model_name = f"best-model-run-{run.name}-fold-{fold}"
        save_path = f"{MODEL_FOLDER}/{save_model_name}.pth"
        for epoch in range(n_epochs):
            print(f"Fold: {fold} Epoch: {epoch}")

            train_loss, train_mae = train_one_epoch(
                train_dataloader, model, loss_fn, optimizer
            )
            val_loss, val_mae = eval_one_epoch(val_dataloader, model, loss_fn)

            scheduler.step(val_mae)

            print(f"val_mae: {val_mae*255.6} val_loss: {val_loss}")
            print(scheduler.get_last_lr())

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                save_checkpoint(
                    save_path, model, optimizer, scheduler, epoch, fold, best_val_mae
                )

            train_loss_history[epoch, fold] = train_loss
            train_mae_history[epoch, fold] = train_mae
            val_loss_history[epoch, fold] = val_loss
            val_mae_history[epoch, fold] = val_mae

        artifact = wandb.Artifact(save_model_name, type="model")
        artifact.add_file(save_path)
        run.log_artifact(artifact)

    for epoch in range(n_epochs):
        for fold in range(datasets.n_splits):
            wandb.log(
                {
                    "fold": fold,
                    "epoch": epoch,
                    f"fold_{fold}_train_loss": train_loss_history[epoch, fold],
                    f"fold_{fold}_train_mae": train_mae_history[epoch, fold],
                    f"fold_{fold}_val_loss": val_loss_history[epoch, fold],
                    f"fold_{fold}_val_mae": val_mae_history[epoch, fold],
                }
            )
        wandb.log(
            {
                "epoch": epoch,
                "avg_train_loss": np.average(train_loss_history[epoch, :]),
                "avg_train_mae": np.average(train_mae_history[epoch, :]),
                "avg_val_loss": np.average(val_loss_history[epoch, :]),
                "avg_val_mae": np.average(val_mae_history[epoch, :]),
            }
        )

    wandb.finish()


# config = {
#     "batch_size": 32,
#     "n_epochs": 16,
#     "learning_rate": 0.0005,
#     "momentum": 0.9,
#     'weight_decay': 0.01,

#     'layer_size': 16,
#     'n_layers': 2,
#     'p_dropout': 0,
# }

config_light_rain = {
    "batch_size": 1024,
    "n_epochs": 100,
    "learning_rate": 0.006,
    "weight_decay": 0.01385,
    "conv_channels": 70,
    "dense_layer_size": 116,
    "n_dense_layers": 5,
    "p_dropout": 0.3,
    "architecture": "4_avgpool_3",
}

config_comic_tree = {
    "batch_size": 512,
    "n_epochs": 100,
    "learning_rate": 0.005725,
    "weight_decay": 0.0125,
    "conv_channels": 113,
    "dense_layer_size": 189,
    "n_dense_layers": 4,
    "p_dropout": 0.5,
    "architecture": "4_avgpool_3",
}

# config = {
#     "batch_size": 1024,
#     "n_epochs": 50,
#     "learning_rate": 0.0025,
#     "weight_decay": 0.035,
#     "conv_channels": 64,
#     "lstm_size": 64,
#     "n_lstm_layers": 2,
#     "dense_layer_size": 128,
#     "n_dense_layers": 2,
#     "p_dropout": 0.4,
# }


if __name__ == "__main__":

    main(config_light_rain)
