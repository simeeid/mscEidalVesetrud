import __fix_relative_imports  # noqa: F401
import torch
import torch.nn as nn
import wandb
import numpy as np
from functools import partial
from mscEidalVesetrudUnofficial.data_preprocessing.prepare_load_dataset import (
    load_cross_val,
)
from mscEidalVesetrudUnofficial.deep_learning.neural_nets import (
    DenseNN,
    ConvNet3D,
    LSTMConv2D,
)
from mscEidalVesetrudUnofficial.deep_learning.train_model import (
    train_one_epoch,
    eval_one_epoch,
)
from mscEidalVesetrudUnofficial.global_constants import (
    TRAIN_SIZE,
    VAL_SIZE,
    TRAIN_DATA_PATH,
    SEEDS,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def main(network_type: str, loss_func: str, optimizer_type: str, seeds: list[int]):

    assert network_type in [
        "dense",
        "cnn_4_avgpool_3",
        "cnn_5_4_4",
        "cnn_5_3_3_3",
        "cnn_4_4_4",
        "cnn_5_5",
        "cnn_6_4_3",
        "cnn_7_3_3",
        "lstm",
    ]
    assert loss_func in ["mse", "huber"]
    assert optimizer_type in ["ngd", "adamw"]

    config = {
        "_network_type": network_type,
        "_loss_func": loss_func,
        "_optimizer_type": optimizer_type,
        "_seeds": seeds,
    }

    run = wandb.init(config=config)
    n_epochs = wandb.config.n_epochs

    try:
        lr_scheduler_factor = wandb.config["lr_scheduler_factor"]
        lr_scheduler_patience = wandb.config["lr_scheduler_patience"]
        use_lr_scheduler = True
    except KeyError:
        use_lr_scheduler = False

    datasets = load_cross_val(TRAIN_DATA_PATH, train_size=TRAIN_SIZE, val_size=VAL_SIZE)

    train_loss_history = np.zeros((n_epochs, datasets.n_splits, len(seeds)))
    train_mae_history = np.zeros((n_epochs, datasets.n_splits, len(seeds)))
    val_loss_history = np.zeros((n_epochs, datasets.n_splits, len(seeds)))
    val_mae_history = np.zeros((n_epochs, datasets.n_splits, len(seeds)))

    for seed_idx, seed in enumerate(seeds):

        np.random.seed(seed)
        torch.manual_seed(seed)

        for fold, (train_data, val_data) in enumerate(
            datasets.split(scale=True, device=device)
        ):

            train_dataloader = torch.utils.data.DataLoader(
                dataset=train_data, batch_size=wandb.config.batch_size, shuffle=True
            )
            val_dataloader = torch.utils.data.DataLoader(
                dataset=val_data, batch_size=wandb.config.batch_size, shuffle=True
            )

            if network_type[:3] == "cnn":
                model = ConvNet3D(
                    wandb.config.conv_channels,
                    wandb.config.dense_layer_size,
                    wandb.config.n_dense_layers,
                    wandb.config.p_dropout,
                    network_type[4:],
                    final_activation="hardsigmoid",
                ).to(device)
            elif network_type == "dense":
                model = DenseNN(
                    wandb.config.layer_size,
                    wandb.config.n_layers,
                    wandb.config.p_dropout,
                    final_activation="hardsigmoid",
                ).to(device)
            elif network_type == "lstm":
                model = LSTMConv2D(
                    wandb.config.conv_channels,
                    wandb.config.lstm_size,
                    wandb.config.n_lstm_layers,
                    wandb.config.dense_layer_size,
                    wandb.config.n_dense_layers,
                    wandb.config.p_dropout,
                    final_activation="hardsigmoid",
                ).to(device)
            else:
                assert False

            if loss_func == "mse":
                loss_fn = nn.MSELoss()
            elif loss_func == "huber":
                loss_fn = nn.HuberLoss()
            else:
                assert False

            if optimizer_type == "ngd":
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=wandb.config.learning_rate,
                    momentum=wandb.config.momentum,
                    dampening=0,
                    nesterov=True,
                    weight_decay=wandb.config.weight_decay,
                )
            elif optimizer_type == "adamw":
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=wandb.config.learning_rate,
                    weight_decay=wandb.config.weight_decay,
                )

            if use_lr_scheduler:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    "min",
                    factor=lr_scheduler_factor,
                    patience=lr_scheduler_patience,
                    threshold=0.0001,
                )

            for epoch in range(n_epochs):
                # print(f"Seed: {seed} Fold: {fold} Epoch: {epoch}")

                train_loss, train_mae = train_one_epoch(
                    train_dataloader, model, loss_fn, optimizer
                )
                val_loss, val_mae = eval_one_epoch(val_dataloader, model, loss_fn)

                if use_lr_scheduler:
                    scheduler.step(val_mae)

                # print(f"val_mae: {val_mae*255.6} val_loss: {val_loss}")

                train_loss_history[epoch, fold, seed_idx] = train_loss
                train_mae_history[epoch, fold, seed_idx] = train_mae
                val_loss_history[epoch, fold, seed_idx] = val_loss
                val_mae_history[epoch, fold, seed_idx] = val_mae

                wandb.log(
                    {
                        "fold": fold,
                        "epoch": epoch,
                        f"fold_{fold}_seed_{seed}_train_loss": train_loss_history[
                            epoch, fold, seed_idx
                        ],
                        f"fold_{fold}_seed_{seed}_train_mae": train_mae_history[
                            epoch, fold, seed_idx
                        ],
                        f"fold_{fold}_seed_{seed}_val_loss": val_loss_history[
                            epoch, fold, seed_idx
                        ],
                        f"fold_{fold}_seed_{seed}_val_mae": val_mae_history[
                            epoch, fold, seed_idx
                        ],
                    }
                )
    for epoch in range(n_epochs):
        wandb.log(
            {
                "epoch": epoch,
                "avg_train_loss": np.average(train_loss_history[epoch, :, :]),
                "avg_train_mae": np.average(train_mae_history[epoch, :, :]),
                "avg_val_loss": np.average(val_loss_history[epoch, :, :]),
                "avg_val_mae": np.average(val_mae_history[epoch, :, :]),
            }
        )

    wandb.finish()


sweep_configuration_conv_ngd = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "avg_val_mae"},
    "parameters": {
        "batch_size": {"values": [64, 128, 256, 512, 1024, 1536, 2048]},
        "n_epochs": {"values": [30, 40, 50]},
        "learning_rate": {"max": 0.01, "min": 0.0001, "distribution": "uniform"},
        "momentum": {"max": 0.95, "min": 0.8, "distribution": "uniform"},
        "weight_decay": {"max": 0.1, "min": 0.0},
        "conv_channels": {"max": 128, "min": 8, "distribution": "int_uniform"},
        "dense_layer_size": {"max": 256, "min": 16, "distribution": "int_uniform"},
        "n_dense_layers": {"values": [1, 2, 3, 4, 5, 6]},
        "p_dropout": {"values": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]},
    },
}

sweep_configuration_dense_ngd = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "avg_val_mae"},
    "parameters": {
        "batch_size": {"values": [64, 128, 256, 512, 1024, 1536, 2048]},
        "n_epochs": {"values": [30, 40, 50]},
        "learning_rate": {"max": 0.01, "min": 0.0001, "distribution": "uniform"},
        "momentum": {"max": 0.95, "min": 0.8, "distribution": "uniform"},
        "weight_decay": {"max": 0.1, "min": 0.0},
        "layer_size": {"max": 256, "min": 16, "distribution": "int_uniform"},
        "n_layers": {"values": [1, 2, 3, 4, 5, 6]},
        "p_dropout": {"values": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]},
    },
}

sweep_configuration_conv_adamw = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "avg_val_mae"},
    "parameters": {
        "batch_size": {"values": [64, 128, 256, 512, 1024, 1536, 2048]},
        "n_epochs": {"values": [30, 40, 50]},
        "learning_rate": {"max": 0.01, "min": 0.0001, "distribution": "uniform"},
        "weight_decay": {"max": 0.1, "min": 0.0},
        "conv_channels": {"max": 128, "min": 8, "distribution": "int_uniform"},
        "dense_layer_size": {"max": 256, "min": 16, "distribution": "int_uniform"},
        "n_dense_layers": {"values": [1, 2, 3, 4, 5, 6]},
        "p_dropout": {"values": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]},
    },
}

sweep_configuration_dense_adamw = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "avg_val_mae"},
    "parameters": {
        "batch_size": {"values": [64, 128, 256, 512, 1024, 1536, 2048]},
        "n_epochs": {"values": [30, 40, 50]},
        "learning_rate": {"max": 0.01, "min": 0.0001, "distribution": "uniform"},
        "weight_decay": {"max": 0.1, "min": 0.0},
        "layer_size": {"max": 256, "min": 16, "distribution": "int_uniform"},
        "n_layers": {"values": [1, 2, 3, 4, 5, 6]},
        "p_dropout": {"values": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]},
    },
}

sweep_meta_options = {
    "network_type": ["dense", "cnn_4_avgpool_3"],
    "loss_func": ["mse", "huber"],
    "optimizer_type": ["adamw", "ngd"],
    "runs": 30,
}

if __name__ == "__main__":
    project_name = "exploratory-sweeps"

    for network_type in sweep_meta_options["network_type"]:
        for loss_func in sweep_meta_options["loss_func"]:
            for optimizer_type in sweep_meta_options["optimizer_type"]:

                if network_type == "dense" and optimizer_type == "ngd":
                    sweep_config = sweep_configuration_dense_ngd
                elif network_type == "dense" and optimizer_type == "adamw":
                    sweep_config = sweep_configuration_dense_adamw
                elif network_type[:3] == "cnn" and optimizer_type == "ngd":
                    sweep_config = sweep_configuration_conv_ngd
                elif network_type[:3] == "cnn" and optimizer_type == "adamw":
                    sweep_config = sweep_configuration_conv_adamw
                else:
                    assert False

                intitialized_main = partial(
                    main,
                    network_type,
                    loss_func,
                    optimizer_type,
                    SEEDS,
                )

                sweep_config["name"] = (
                    f"sweep-{network_type}-{loss_func}-{optimizer_type}"
                )

                sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)

                wandb.agent(
                    sweep_id,
                    function=intitialized_main,
                    count=sweep_meta_options["runs"],
                )
