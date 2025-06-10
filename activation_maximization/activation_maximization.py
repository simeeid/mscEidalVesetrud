import __fix_relative_imports  # noqa: F401

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mscEidalVesetrudUnofficial.deep_learning.neural_nets import IndexableModule
from mscEidalVesetrudUnofficial.global_constants import (
    CONTAINING_FOLDER,
    SEEDS,
    TRAIN_DATA_PATH,
    TRAIN_SIZE,
    VAL_SIZE,
)
from mscEidalVesetrudUnofficial.deep_learning.train_model import load_checkpoint
from mscEidalVesetrudUnofficial.data_preprocessing.prepare_load_dataset import (
    load_cross_val,
)
from mscEidalVesetrudUnofficial.deep_learning.visualize import (
    plot_weather_data_medium,
    plot_weather_data_time,
)
from scipy.stats import circstd, circmean


def visualize_wind(
    weather_data: np.ndarray, timestamps: pd.DatetimeIndex, save_as_pdf: bool = False
):

    if save_as_pdf:
        for i in range(11):
            plot_weather_data_medium(
                weather_data[i],
                timestamps[i],
                height="all",
                arrow_length_scale=0.6 / 20,
                arrow_width=0.15,
                show_background=True,
                custom_name=f"act_max_wind_{i}.pdf",
            )

    else:
        plot_weather_data_time(weather_data, timestamps)


def variability_avg_stats(
    weather_data: np.ndarray, times_of_year: np.ndarray, times_of_day: np.ndarray
) -> tuple[np.ndarray, float, float]:
    assert weather_data.shape[1:] == (11, 13, 13, 3, 3)

    weather_degrees = np.rad2deg(
        np.arctan2(weather_data[:, :, :, :, :, 2], weather_data[:, :, :, :, :, 1])
    )
    weather_std = np.std(weather_data[:, :, :, :, :, 0], axis=0)

    weather_circstd = circstd(weather_degrees, axis=0, high=360, low=0)

    print(f"Average circular std (degrees): {np.average(weather_circstd):.4e}")
    print(
        f"Average speed std (m/s)         : {np.average(weather_std):.4e}",
    )

    print(
        f"Average circular std year (days): {np.average(circstd(times_of_year) * 365 / (2 * np.pi)):.4e}"
    )
    print(
        f"Average circular std day (hours): {np.average(circstd(times_of_day) * 24 / (2 * np.pi)):.4e}",
    )

    return (
        np.average(weather_data, axis=0),
        float(circmean(times_of_year)),
        float(circmean(times_of_day)),
    )


def time_of_year_and_day_to_timestamp(
    time_of_year: np.ndarray, time_of_day: np.ndarray
) -> pd.DatetimeIndex:
    # Converts the hour in the day and the day in the year to a pandas date in 2100
    # 2100 is not a leap year
    # The day in the year will not affect the hour at all, that is only affected by the time of day

    hour = ((np.asarray(time_of_day) / (2 * np.pi)) % 1.0) * 24
    day_in_year = ((np.asarray(time_of_year) / (2 * np.pi)) % 1.0) * 365

    year = np.asarray(2100) - 1970
    types = ("<M8[Y]", "<m8[D]", "<m8[h]")
    vals = (year, day_in_year, hour)
    center_time = pd.Timestamp(sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)))

    timestamps = pd.date_range(
        center_time - pd.Timedelta(hours=5),
        center_time + pd.Timedelta(hours=5),
        freq="h",
    )
    return timestamps


class OptimizableWindInput(nn.Module):
    def __init__(
        self,
        wind_speed_mean: np.ndarray | float,
        wind_speed_std: np.ndarray | float,
    ):
        super(OptimizableWindInput, self).__init__()
        WIND_SHAPE = (1, 11, 13, 13, 3, 1)

        def transform_numpy(x: np.ndarray | float):
            if isinstance(x, np.ndarray):
                reshaped_x = x.reshape(WIND_SHAPE)
            elif isinstance(x, float) or isinstance(x, int):
                reshaped_x = np.ones(WIND_SHAPE, dtype=np.float32) * x
            else:
                raise ValueError()

            return torch.tensor(reshaped_x, dtype=torch.float32, requires_grad=False)

        self.wind_speed_mean = transform_numpy(wind_speed_mean)
        self.wind_speed_std = transform_numpy(wind_speed_std)

        self.wind_dir_tensor = nn.Parameter(
            torch.empty(WIND_SHAPE, dtype=torch.float32), requires_grad=True
        )
        self.wind_speed_tensor = nn.Parameter(
            torch.empty(WIND_SHAPE), requires_grad=True
        )
        nn.init.uniform_(self.wind_dir_tensor, 0, 2 * torch.pi)

        # Initialize the wind tensor to be a truncated normal distribution, created so the average and
        # Standard deviation tends to follow the measured wind speeds
        # It gets quite close, but the main difference is how the real wind speeds are less often close to zero

        # avg 5.369504 std 3.3988647
        nn.init.trunc_normal_(
            self.wind_speed_tensor[:, :, :, :, 0, :], mean=4.2, std=4.2, a=0, b=45
        )
        # avg 7.876362 std 4.413766
        nn.init.trunc_normal_(
            self.wind_speed_tensor[:, :, :, :, 1, :], mean=7.1, std=5.0, a=0, b=45
        )
        # avg 8.499261 std 4.7546616
        nn.init.trunc_normal_(
            self.wind_speed_tensor[:, :, :, :, 2, :], mean=7.7, std=5.3, a=0, b=45
        )
        self.time_of_day = nn.Parameter(torch.empty(1), requires_grad=True)
        nn.init.uniform_(self.time_of_day, 0, 2 * torch.pi)
        self.time_of_year = nn.Parameter(torch.empty(1), requires_grad=True)
        nn.init.uniform_(self.time_of_year, 0, 2 * torch.pi)

    # Overrides the _apply function, which underlines the .to() functionality
    # This way, also the self.wind_speed_mean and self.wind_speed_std tensors
    # are also affected by the change
    def _apply(self, fn, recurse: bool = True):
        super(OptimizableWindInput, self)._apply(fn, recurse)
        self.wind_speed_mean = fn(self.wind_speed_mean)
        self.wind_speed_std = fn(self.wind_speed_std)
        return self

    def plot_speed_wind_tensor_hist(self, height_idx: int, compare: np.ndarray):
        # Used to test the inital distribution of the wind tensor
        wind = (
            self.wind_speed_tensor[:, :, :, :, height_idx, :]
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )
        plt.hist(wind, bins=30, alpha=0.5, label="generated")
        plt.hist(compare, bins=30, alpha=0.5, label="true")
        plt.title("wind speed histogram")
        plt.legend()
        plt.show()

    @property
    def time_of_year_day(self) -> tuple[float, float]:
        return (
            float(self.time_of_year.detach().cpu().numpy()[0]),
            float(self.time_of_day.detach().cpu().numpy()[0]),
        )

    @property
    def weather_data(self) -> np.ndarray:
        weather_data = (
            torch.cat(
                (
                    self.wind_speed_tensor,
                    torch.cos(self.wind_dir_tensor),
                    torch.sin(self.wind_dir_tensor),
                ),
                dim=-1,
            )
            .detach()
            .cpu()
            .numpy()
        ).reshape(11, 13, 13, 3, 3)

        return weather_data

    def clip_wind(self):
        with torch.no_grad():
            self.wind_speed_tensor.data = torch.clamp_(
                self.wind_speed_tensor.data, min=0, max=None
            )

    def forward(self, x):

        scaled_wind_speed = (
            self.wind_speed_tensor - self.wind_speed_mean
        ) / self.wind_speed_std

        wind_with_dir = torch.cat(
            (
                scaled_wind_speed,
                torch.cos(self.wind_dir_tensor),
                torch.sin(self.wind_dir_tensor),
            ),
            dim=-1,
        )

        time_features = torch.empty((1, 5), device=self.time_of_day.device)
        time_features[:, 0] = (self.time_of_day / (2 * torch.pi)) % 1.0
        time_features[:, 1] = torch.cos(self.time_of_day)
        time_features[:, 2] = torch.sin(self.time_of_day)
        time_features[:, 3] = torch.cos(self.time_of_year)
        time_features[:, 4] = torch.sin(self.time_of_year)

        return wind_with_dir, time_features


class ActivationMaximization(nn.Module):
    def __init__(
        self,
        input: OptimizableWindInput,
        model: IndexableModule,
        optimized_layer_name: str,
        layer_selection: int | torch.Tensor,
    ):
        super(ActivationMaximization, self).__init__()

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.input = input
        self.model = model
        self.optimized_layer_name = optimized_layer_name
        self.layer_selection = layer_selection

        # Register the hook that tracks the value to be optimized
        self.activation: torch.Tensor = torch.empty(0)

        def hook(
            module: torch.nn.Module, input: tuple[torch.Tensor], output: torch.Tensor
        ):
            self.activation = output

        self.hook_handle = self.model[optimized_layer_name].register_forward_hook(hook)

    def _apply(self, fn, recurse: bool = True):
        # Overrides the _apply function, which underlines the .to() functionality
        super(ActivationMaximization, self)._apply(fn, recurse)

        if isinstance(self.layer_selection, torch.Tensor):
            self.layer_selection = fn(self.layer_selection)
        self.activation = fn(self.activation)
        return self

    def forward(self, x):
        _ = self.model(*self.input(None))

        act_shape = self.activation.shape
        assert (
            len(act_shape) == 2
            and act_shape[0] == 1
            and (
                (
                    isinstance(self.layer_selection, int)
                    and self.layer_selection < act_shape[1]
                )
                or (
                    isinstance(self.layer_selection, torch.Tensor)
                    and len(self.layer_selection.shape) == 1
                    and self.layer_selection.shape[0] == act_shape[1]
                )
            )
        )

        if isinstance(self.layer_selection, int):
            return self.activation[0, self.layer_selection]

        # assume the tensor is on the same device
        return torch.dot(self.activation[0], self.layer_selection)

    def __enter__(self) -> "ActivationMaximization":
        return self

    def finalize(self):
        # Remove the hooks to prevent side effects
        self.hook_handle.remove()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()


def run_activation_maximization(
    model: IndexableModule,
    model_wind_scaling_mean: np.ndarray,
    model_wind_scaling_std: np.ndarray,
    layer_name: str,
    layer_selection: int | torch.Tensor,
    loss_fn,
    device: str,
    visualize_initial_wind: bool = False,
    visualize_end_wind: bool = False,
    save_as_pdf: bool = False,
    n_epochs: int = 500,
    learning_rate: float = 0.2,
) -> tuple[np.ndarray, float, float, float]:

    optimizable_wind_input = OptimizableWindInput(
        model_wind_scaling_mean, model_wind_scaling_std
    )

    if visualize_initial_wind:
        visualize_wind(
            optimizable_wind_input.weather_data,
            time_of_year_and_day_to_timestamp(*optimizable_wind_input.time_of_year_day),
            save_as_pdf,
        )

    with ActivationMaximization(
        optimizable_wind_input, model, layer_name, layer_selection
    ).to(device) as act_max:
        optimizer = torch.optim.Adam(act_max.parameters(), lr=learning_rate)

        for i in range(n_epochs):
            optimizer.zero_grad()
            output = act_max(None)
            (loss_fn(output)).backward()
            optimizer.step()
            act_max.input.clip_wind()
            print(f"target: {output.item()}")

    if visualize_end_wind:
        visualize_wind(
            optimizable_wind_input.weather_data,
            time_of_year_and_day_to_timestamp(*optimizable_wind_input.time_of_year_day),
            save_as_pdf,
        )

    return (
        optimizable_wind_input.weather_data,
        *optimizable_wind_input.time_of_year_day,
        float(output.item()),
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = load_cross_val(TRAIN_DATA_PATH, TRAIN_SIZE, VAL_SIZE, verbose=True)
    model_wind_mean = datasets.wind_speed_scaler_full.mean
    model_wind_std = datasets.wind_speed_scaler_full.std

    np.random.seed(SEEDS[0])
    torch.manual_seed(SEEDS[0])

    MODEL_PATH = f"{CONTAINING_FOLDER}/mscEidalVesetrudUnofficial/models_test/final-model-sith-emperor-57.pth"
    (
        model,
        _,
        _,
        epoch,
        model_fold,
        mae,
    ) = load_checkpoint(MODEL_PATH, device, force_final_activation="sigmoid")
    print(
        f"Loaded model type {model.class_name} with epoch {epoch}, fold {model_fold}, mae {mae}"
    )
    model.summary()

    # layer_selection = torch.zeros(116)
    # layer_selection[42] = 1
    # layer_name = "dense_silu_2"

    layer_name = "dense_layer_final"
    # layer_name = "final_activation"
    layer_selection = 0

    n_seeds = 50

    combined_wind_data = np.zeros((n_seeds, 11, 13, 13, 3, 3))
    times_of_year = np.zeros((n_seeds))
    times_of_day = np.zeros((n_seeds))
    end_targets = np.zeros((n_seeds))
    print(times_of_day.shape)

    for seed in range(n_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        weather_data, time_of_year, time_of_day, end_target = (
            run_activation_maximization(
                model,
                model_wind_mean,
                model_wind_std,
                layer_name,
                layer_selection,
                loss_fn=lambda x: -x,
                # loss_fn=lambda x: (0.99 - x) ** 2, # Assuming layer_name is "final_activation"
                device=device,
                visualize_end_wind=False,
                n_epochs=1000,
                learning_rate=0.015,
            )
        )

        combined_wind_data[seed] = weather_data
        times_of_year[seed] = time_of_year
        times_of_day[seed] = time_of_day
        end_targets[seed] = end_target

    wind_avg, time_of_year_avg, time_of_day_avg = variability_avg_stats(
        combined_wind_data, times_of_year, times_of_day
    )
    visualize_wind(
        np.average(combined_wind_data, axis=0),
        time_of_year_and_day_to_timestamp(time_of_year_avg, time_of_day_avg),
    )
    visualize_wind(
        np.average(combined_wind_data, axis=0),
        time_of_year_and_day_to_timestamp(time_of_year_avg, time_of_day_avg),
        save_as_pdf=True,
    )
    print(f"Average end target {np.average(end_target):.4e}")
    print(
        f"center time: {time_of_year_and_day_to_timestamp(time_of_year_avg, time_of_day_avg)}"
    )
