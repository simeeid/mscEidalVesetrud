import __fix_relative_imports  # noqa: F401
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from IPython.display import HTML, display
from datetime import timedelta

import os

from mscEidalVesetrudUnofficial.data_preprocessing.prepare_load_dataset import (
    load_cross_val,
    load_scale_test_dataset,
)
from mscEidalVesetrudUnofficial.global_constants import (
    TRAIN_DATA_PATH,
    TRAIN_SIZE,
    VAL_SIZE,
    SHADOW_MAP_35_BY_35_PATH,
    TEST_DATA_PATH,
)

import torch
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from sklearn.preprocessing import StandardScaler
import warnings


warnings.filterwarnings("ignore")

plant_name: str = "roan"
plant_coordinates: str = "6414_1038"

node_index = 6


# Perhaps convergence of wind directions is important, what about curl?

# Set the font globally
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Computer Modern Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "medium"
plt.rcParams["axes.titlesize"] = "large"
plt.rcParams["xtick.labelsize"] = "medium"
plt.rcParams["ytick.labelsize"] = "medium"


def plot_weather_data_time(
    weather_data: np.ndarray,
    timestamps: pd.DatetimeIndex,
    height: str = "all",
    grid_spacing=1,
    arrow_length_scale=1 / 20,
    margin=1,
    show_background: bool = True,
):
    # Increasing Lonitude means more east
    # Increasing Latitude means more north

    # weather_data:
    # hour(any), latitude(13), longitude(13), height(3), speed_direction_cos_sin(3)

    # 0 degrees -> north (increasing latitude)
    # 90 degrees -> east (increasing longitude)

    assert weather_data.shape[1:] == (13, 13, 3, 3)
    assert np.all(
        weather_data[:, :, :, :, 0] >= 0
    )  # Assert positive wind speed, should not be scaled
    assert weather_data.shape[0] == timestamps.shape[0]

    match height:
        case "10m":
            weather_data = weather_data[:, :, :, 0, :]
        case "80m":
            weather_data = weather_data[:, :, :, 1, :]
        case "120m":
            weather_data = weather_data[:, :, :, 2, :]
        case "avg":
            weather_data = np.average(weather_data, axis=3)
        case "all":
            pass
        case _:
            raise ValueError()
    # weather_data now has shape (time_steps,13,13,3) with axis 3 being [wind_speed, cos, sin]
    # or it has shape (time_steps,13,13,3,3) with axis 3 as height and axis 4 being [wind_speed, cos, sin]

    num_time_steps = weather_data.shape[0]
    n_latitude, n_longitude = weather_data.shape[1:3]

    fig_height, fig_width = 6.0, 5.7
    (fig, (ax_top, ax, ax_slider)) = plt.subplots(
        3, 1, height_ratios=[0.3, fig_width - 0.5, 0.5], layout="constrained"
    )
    ax: Axes = ax
    # (fig, (ax, ax_slider)) = plt.subplots(2, 1, height_ratios=[fig_width,0.5], layout="constrained")
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)

    # Only used to make space for the slider
    ax_slider.set_axis_off()
    del ax_slider

    x_tails = np.arange(0, n_longitude * grid_spacing, grid_spacing)
    y_tails = np.arange(0, n_latitude * grid_spacing, grid_spacing)

    time_text = ax_top.text(
        0.25,
        0.43,
        f"Time: {timestamps[0]}",
        fontsize="medium",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax_top.transAxes,
    )

    if height == "all":
        u = np.zeros((n_latitude, n_longitude, 3))  # x-component of the arrows
        v = np.zeros((n_latitude, n_longitude, 3))  # y-component of the arrows

        # Create quiver plot (initial state)
        Q10 = ax.quiver(
            x_tails,
            y_tails,
            u[:, :, 0],
            v[:, :, 0],
            angles="xy",
            scale_units="xy",
            scale=1,  # Important for correct scaling
            headwidth=0.15 * grid_spacing / arrow_length_scale,
            headlength=0.275 * grid_spacing / arrow_length_scale,
            fc="red",
            ec="white",
            linewidth=0.02 / arrow_length_scale,
            label="10m",
        )
        # Create quiver plot (initial state)
        Q80 = ax.quiver(
            x_tails,
            y_tails,
            u[:, :, 1],
            v[:, :, 1],
            angles="xy",
            scale_units="xy",
            scale=1,  # Important for correct scaling
            headwidth=0.15 * grid_spacing / arrow_length_scale,
            headlength=0.275 * grid_spacing / arrow_length_scale,
            fc="green",
            ec="white",
            linewidth=0.02 / arrow_length_scale,
            label="80m",
        )
        # Create quiver plot (initial state)
        Q120 = ax.quiver(
            x_tails,
            y_tails,
            u[:, :, 2],
            v[:, :, 2],
            angles="xy",
            scale_units="xy",
            scale=1,  # Important for correct scaling
            headwidth=0.15 * grid_spacing / arrow_length_scale,
            headlength=0.275 * grid_spacing / arrow_length_scale,
            fc="blue",
            ec="white",
            linewidth=0.02 / arrow_length_scale,
            label="120m",
        )

        def update(time_index: float):
            current_data = weather_data[int(time_index)]

            u = (
                current_data[:, :, :, 0] * arrow_length_scale * current_data[:, :, :, 2]
            )  # sine
            v = (
                current_data[:, :, :, 0] * arrow_length_scale * current_data[:, :, :, 1]
            )  # cosine

            Q10.set_UVC(u[:, :, 0], v[:, :, 0])
            Q80.set_UVC(u[:, :, 1], v[:, :, 1])
            Q120.set_UVC(u[:, :, 2], v[:, :, 2])

            time_text.set_text(f"Time: {timestamps[int(time_index)]}")

    else:
        u = np.zeros((n_latitude, n_longitude))  # x-component of the arrows
        v = np.zeros((n_latitude, n_longitude))  # y-component of the arrows

        # Create quiver plot (initial state)
        Q = ax.quiver(
            x_tails,
            y_tails,
            u,
            v,
            angles="xy",
            scale_units="xy",
            scale=1,  # Important for correct scaling
            headwidth=0.15 * grid_spacing / arrow_length_scale,
            headlength=0.275 * grid_spacing / arrow_length_scale,
            fc="blue",
            ec="blue",
            label=height,
        )

        def update(time_index: float):
            current_data = weather_data[int(time_index)]

            u = (
                current_data[:, :, 0] * arrow_length_scale * current_data[:, :, 2]
            )  # sine
            v = (
                current_data[:, :, 0] * arrow_length_scale * current_data[:, :, 1]
            )  # cosine

            Q.set_UVC(u, v)
            time_text.set_text(f"Time: {timestamps[int(time_index)]}")
            # Return only used for FuncAnimation
            # return Q,

    # Set axis limits with margin once
    x_end = -margin, (n_longitude - 1) * grid_spacing + margin
    y_end = -margin, (n_latitude - 1) * grid_spacing + margin
    ax.set_xlim(x_end)
    ax.set_ylim(y_end)

    if show_background:
        background = plt.imread(SHADOW_MAP_35_BY_35_PATH)
        ax.imshow(background, extent=[*x_end, *y_end], alpha=0.4)

    ax.set_xticks(x_tails)
    ax.set_xticklabels(
        np.round(np.linspace(10.07, 10.69, n_longitude), decimals=2), rotation=30
    )

    ax.set_yticks(y_tails)
    ax.set_yticklabels(np.round(np.linspace(64.00, 64.27, n_latitude), decimals=2))

    ax.set_aspect("equal")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.suptitle("Wind plot")
    ax.grid(True)

    # Legend MUST match with the colors and type of the arrows
    ax_top.set_axis_off()
    if height == "all":
        ax_top.legend(
            loc="center right",
            ncol=3,
            bbox_to_anchor=(0.9, 0.5),
            handles=[
                patches.Patch(color="red", label="10m"),
                patches.Patch(color="green", label="80m"),
                patches.Patch(color="blue", label="120m"),
            ],
        )
    else:
        ax_top.legend(
            loc="center right",
            bbox_to_anchor=(0.9, 0.5),
            handles=[patches.Patch(color="blue", label=height)],
        )

    # Create the slider
    # Position the slider 10% from the left, 2% from the bottom 80% wide, and 6% tall
    # This is done separatly from the axes returned by plt.subplot, because the width of the axes around the slider must be absolute,
    # otherwise the entire plot will adjust when the value becomes larger than 10 or 100.
    ax_slider = fig.add_axes((0.1, 0.02, 0.8, 0.06))
    time_slider = Slider(
        ax_slider, "Time", 0, num_time_steps - 1, valinit=0, valstep=1
    )  # Integer steps

    # ani = FuncAnimation(fig, update, frames=num_time_steps, blit=True) # blit=True if no other artists are being animated

    time_slider.on_changed(update)
    update(0)

    plt.show()


def plot_weather_data(
    weather_data: np.ndarray,
    time: str,
    height: str = "all",
    grid_spacing=1,
    arrow_length_scale=1 / 20,
    margin=1,
    show_background: bool = True,
    custom_name: str = None,
):
    assert weather_data.shape == (13, 13, 3, 3)
    assert np.all(
        weather_data[:, :, :, 0] >= 0
    )  # Assert positive wind speed, should not be scaled

    match height:
        case "10m":
            weather_data = weather_data[:, :, 0, :]
        case "80m":
            weather_data = weather_data[:, :, 1, :]
        case "120m":
            weather_data = weather_data[:, :, 2, :]
        case "avg":
            weather_data = np.average(weather_data, axis=3)
        case "all":
            pass
        case _:
            raise ValueError()
    # weather_data now has shape (13,13,3) with axis 3 being [wind_speed, cos, sin]
    # or it has shape (13,13,3,3) with axis 3 as height and axis 4 being [wind_speed, cos, sin]

    n_latitude, n_longitude = weather_data.shape[0:2]

    fig_height, fig_width = 5.7, 5.7
    (fig, ax) = plt.subplots(1, 1, layout="constrained")
    # (fig, (ax, ax_slider)) = plt.subplots(2, 1, height_ratios=[fig_width,0.5], layout="constrained")
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)

    x_tails = np.arange(0, n_longitude * grid_spacing, grid_spacing)
    y_tails = np.arange(0, n_latitude * grid_spacing, grid_spacing)

    if height == "all":
        # x-component of the arrows
        u = (
            weather_data[:, :, :, 0] * arrow_length_scale * weather_data[:, :, :, 2]
        )  # sine
        # y-component of the arrows
        v = (
            weather_data[:, :, :, 0] * arrow_length_scale * weather_data[:, :, :, 1]
        )  # cosine

        # Create quiver plot (initial state)
        ax.quiver(
            x_tails,
            y_tails,
            u[:, :, 0],
            v[:, :, 0],
            angles="xy",
            scale_units="xy",
            scale=1,  # Important for correct scaling
            headwidth=0.15 * grid_spacing / arrow_length_scale,
            headlength=0.275 * grid_spacing / arrow_length_scale,
            fc="red",
            ec="white",
            linewidth=0.025 / arrow_length_scale,
            label="10m",
        )
        # Create quiver plot (initial state)
        ax.quiver(
            x_tails,
            y_tails,
            u[:, :, 1],
            v[:, :, 1],
            angles="xy",
            scale_units="xy",
            scale=1,  # Important for correct scaling
            headwidth=0.15 * grid_spacing / arrow_length_scale,
            headlength=0.275 * grid_spacing / arrow_length_scale,
            fc="green",
            ec="white",
            linewidth=0.02 / arrow_length_scale,
            label="80m",
        )
        # Create quiver plot (initial state)
        ax.quiver(
            x_tails,
            y_tails,
            u[:, :, 2],
            v[:, :, 2],
            angles="xy",
            scale_units="xy",
            scale=1,  # Important for correct scaling
            headwidth=0.15 * grid_spacing / arrow_length_scale,
            headlength=0.275 * grid_spacing / arrow_length_scale,
            fc="blue",
            ec="white",
            linewidth=0.02 / arrow_length_scale,
            label="120m",
        )
    else:
        # x-component of the arrows
        u = weather_data[:, :, 0] * arrow_length_scale * weather_data[:, :, 2]  # sine
        # y-component of the arrows
        v = weather_data[:, :, 0] * arrow_length_scale * weather_data[:, :, 1]  # cosine

        # Create quiver plot (initial state)
        ax.quiver(
            x_tails,
            y_tails,
            u,
            v,
            angles="xy",
            scale_units="xy",
            scale=1,  # Important for correct scaling
            headwidth=0.15 * grid_spacing / arrow_length_scale,
            headlength=0.275 * grid_spacing / arrow_length_scale,
            fc="blue",
            ec="blue",
            label=height,
        )

    # Set axis limits with margin once
    x_end = -margin, (n_longitude - 1) * grid_spacing + margin
    y_end = -margin, (n_latitude - 1) * grid_spacing + margin
    ax.set_xlim(x_end)
    ax.set_ylim(y_end)

    if show_background:
        background = plt.imread(SHADOW_MAP_35_BY_35_PATH)
        ax.imshow(background, extent=[*x_end, *y_end], alpha=0.4)

    ax.set_xticks(x_tails)
    ax.set_xticklabels(
        np.round(np.linspace(10.07, 10.69, n_longitude), decimals=2), rotation=30
    )

    ax.set_yticks(y_tails)
    ax.set_yticklabels(np.round(np.linspace(64.00, 64.27, n_latitude), decimals=2))

    ax.set_aspect("equal")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.suptitle("Wind plot")
    ax.grid(True)

    # Set legend:
    # MUST match with the colors and type of the arrows
    if height == "all":
        ax.legend(
            loc="center right",
            ncol=3,
            bbox_to_anchor=(1.0, 1.05),
            handles=[
                patches.Patch(color="red", label="10m"),
                patches.Patch(color="green", label="80m"),
                patches.Patch(color="blue", label="120m"),
            ],
        )
    else:
        ax.legend(
            loc="center right",
            bbox_to_anchor=(1.0, 1.05),
            handles=[patches.Patch(color="blue", label=height)],
        )

    ax.text(
        0.2,
        1.045,
        f"Time: {time}",
        fontsize="medium",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )

    if custom_name is None:
        custom_name = f"wind_plot_{str(time)[:-6]}.pdf"
    plt.savefig(custom_name, format="pdf")
    plt.show()
    plt.close()


def plot_prediction_timeline(
    true_y: np.ndarray, pred_y: np.ndarray, timestamps: pd.DatetimeIndex
):
    assert true_y.ndim == pred_y.ndim == 1
    assert true_y.shape[0] == pred_y.shape[0] == timestamps.shape[0]

    length = true_y.shape[0]
    x_axis = np.arange(length)

    plt.figure(constrained_layout=True, figsize=(5.5, 4))

    plt.plot(x_axis, true_y, linestyle="-", label="True Prod")
    plt.plot(x_axis, pred_y, linestyle="-", label="Prediction")
    plt.xlabel(f"Hours from {timestamps[0]}")
    plt.ylabel("Production [MW]")

    plt.title("Comparison of model")

    plt.grid(True)
    plt.legend()

    # Set x and y axis limits
    # plt.xlim(-5, 5)
    # plt.ylim(-1.25, 2.5)

    plt.savefig("prediction_plot.pdf", format="pdf")
    plt.show()


class WindAnimation:
    def __init__(
        self,
        grid_data_speed: dict,
        grid_data_direction: dict,
        num_time_steps: int = 10,
        fig_size: tuple = (5, 5),
    ):
        self.grid_data_speed = grid_data_speed  # Dict with keys '10m', '80m', '120m'
        self.grid_data_direction = (
            grid_data_direction  # Dict with keys '10m', '80m', '120m'
        )

        # setting 0 degrees to be north, 90 degrees to be east, 180 degrees to be south and 270 degrees to be west
        scale = -90
        for height in self.grid_data_direction.keys():
            self.grid_data_direction[height] = self.grid_data_direction[height] + scale
            self.grid_data_direction[height] = -self.grid_data_direction[height]
            self.grid_data_direction[height] = self.grid_data_direction[height] % 360

        self.global_min_speed = None
        self.global_max_speed = None
        self.last_valid_speed = {h: None for h in ["10m", "80m", "120m"]}
        self.last_valid_direction = {h: None for h in ["10m", "80m", "120m"]}
        self.num_time_steps = num_time_steps
        self.fig_size = fig_size
        self.heights = ["10m", "80m", "120m"]
        self.colors = {"10m": "red", "80m": "green", "120m": "blue"}

    def safe_concatenate(self, data, time_steps, height):
        all_wind_speeds = []
        for time_step in time_steps:
            try:
                all_wind_speeds.append(data[height].loc[time_step].values)
            except KeyError:
                print(f"Data for {time_step} at {height} is missing.")
                all_wind_speeds.append(np.full(data[height].shape[1], np.nan))
        return np.concatenate(all_wind_speeds)

    def create_quiver_plot(self, ax, time_step):
        quivers = []
        missing_data_flags = []

        coordinates = [
            col.split("_")[-2:] for col in self.grid_data_speed["10m"].columns
        ]
        latitudes = [float(lat) for lat, lon in coordinates]
        longitudes = [float(lon) for lat, lon in coordinates]
        scaling_factor = 0.3

        for height in self.heights:
            try:
                row_speed = self.grid_data_speed[height].loc[time_step]
                row_direction = self.grid_data_direction[height].loc[time_step]
                missing_data = False
            except KeyError:
                row_speed = self.last_valid_speed[height]
                row_direction = self.last_valid_direction[height]
                missing_data = True

            wind_speeds = row_speed.values
            wind_directions = row_direction.values
            wind_directions_rad = np.deg2rad(wind_directions)
            scaled_wind_speeds = wind_speeds * scaling_factor

            quiver = ax.quiver(
                longitudes,
                latitudes,
                np.cos(wind_directions_rad) * scaled_wind_speeds,
                np.sin(wind_directions_rad) * scaled_wind_speeds,
                color=self.colors[height],
                scale=1,
                scale_units="xy",
                width=0.004,
            )

            if missing_data:
                quiver.set_alpha(0.5)

            quivers.append(quiver)
            missing_data_flags.append(missing_data)

        return quivers, any(missing_data_flags)

    def update(
        self,
        frame,
        ax,
        specified_time,
        time_steps,
        save_as_pdf=False,
        pdf_prefix="wind",
    ):
        ax.clear()
        try:
            # Calculate frame index here
            frame_idx = time_steps.index(frame)
            quivers, missing_data = self.create_quiver_plot(ax, frame)
            if quivers:
                # ax.set_xlabel("Longitude", fontsize=8)
                # ax.set_ylabel("Latitude", fontsize=8)
                title = f"Wind (10m: Red, 80m: Green, 120m: Blue) - {frame}"
                if missing_data:
                    title += " (Missing values)"
                ax.set_title(title, fontsize=8)
                ax.grid(True)

                for height in self.heights:
                    try:
                        self.last_valid_speed[height] = self.grid_data_speed[
                            height
                        ].loc[frame]
                        self.last_valid_direction[height] = self.grid_data_direction[
                            height
                        ].loc[frame]
                    except KeyError:
                        pass

                ax.text(
                    0.5,
                    -0.1,
                    f"Forecast: {specified_time}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )

                if save_as_pdf:
                    output_dir = "wind_animation_frames"
                    os.makedirs(output_dir, exist_ok=True)
                    pdf_filename = os.path.join(
                        output_dir, f"{pdf_prefix}_{frame_idx:03d}.pdf"
                    )
                    ax.get_figure().savefig(
                        pdf_filename, format="pdf", bbox_inches="tight"
                    )
                    print(f"Saved frame {frame_idx} as {pdf_filename}")

        except KeyError:
            pass
        return ax.get_children()

    def create_animation(
        self,
        specified_time,
        use_in_grid: bool = False,
        fig_size=None,
        save_as_pdf: bool = False,
        pdf_postfix: str = "wind",
    ):
        time_steps = [
            specified_time + timedelta(hours=i)
            for i in range(-self.num_time_steps, self.num_time_steps + 1)
        ]

        all_speeds = []
        for height in self.heights:
            speeds = self.safe_concatenate(self.grid_data_speed, time_steps, height)
            all_speeds.append(speeds)
        all_speeds = np.concatenate(all_speeds)
        self.global_min_speed = np.nanmin(all_speeds)
        self.global_max_speed = np.nanmax(all_speeds)

        if fig_size:
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            fig, ax = plt.subplots(figsize=self.fig_size)

        for height in self.heights:
            for time_step in time_steps:
                try:
                    self.last_valid_speed[height] = self.grid_data_speed[height].loc[
                        str(time_step)
                    ]
                    self.last_valid_direction[height] = self.grid_data_direction[
                        height
                    ].loc[str(time_step)]
                    break
                except KeyError:
                    pass

        def init():
            quivers, _ = self.create_quiver_plot(ax, time_steps[0])
            return quivers

        # Pass time_steps to update function to compute frame index
        ani = FuncAnimation(
            fig,
            lambda frame, ax, st, ts, sp=save_as_pdf, pp=pdf_postfix: self.update(
                frame, ax, st, ts, sp, pp
            ),
            frames=time_steps,
            fargs=(ax, specified_time, time_steps),
            init_func=init,
            repeat=False,
            blit=False,
        )
        if use_in_grid:
            html = ani.to_jshtml()
            plt.close(fig)
            return html
        else:
            display(HTML(ani.to_jshtml()))
            plt.close(fig)

    def create_animation_grid(
        self,
        time_instances,
        max_columns: int = 3,
        fig_size=None,
        save_as_pdf: bool = False,
        pdf_postfix: str = "wind",
    ):
        n_instances = len(time_instances)
        ncols = min(max_columns, int(np.ceil(np.sqrt(n_instances))))
        nrows = int(np.ceil(n_instances / ncols))

        # Generate HTML for each animation and wrap in a styled div
        html_outputs = []
        for idx, specified_time in enumerate(time_instances):
            # Create unique postfix for each animation in the grid
            unique_postfix = f"{pdf_postfix}_grid_{idx}"
            html = self.create_animation(
                specified_time,
                use_in_grid=True,
                fig_size=fig_size,
                save_as_pdf=save_as_pdf,
                pdf_postfix=unique_postfix,
            )
            html_outputs.append(
                f'<div style="display: inline-block; width: {self.fig_size[0]*100}px; margin: 10px; vertical-align: top;">{html}</div>'
            )

        # Combine all HTML outputs in a grid container
        grid_html = f"""
        <div style="display: grid; grid-template-columns: repeat({ncols}, {self.fig_size[0]*100 + 20}px); gap: 20px;">
            {''.join(html_outputs)}
        </div>
        """

        display(HTML(grid_html))


def test_wind() -> np.ndarray:
    # Increasing Lonitude means more east
    # Increasing Latitude means more north
    # hour(x), latitude(13), longitude(13), height(3), speed_direction_cos_sin(3)

    # 0 degrees -> north (increasing latitude)
    # 90 degrees -> east (increasing longitude)

    # This will create a wind grid with the corners pointing away from the center
    # Lengths:
    # South east = 5
    # South west = 10
    # North east = 20
    # North west = 15
    weather_data = np.zeros((1, 13, 13, 3, 2))
    weather_data[:, 0, 0, :, 0] = 10
    weather_data[:, 0, 0, :, 1] = 225
    weather_data[:, 0, 12, :, 0] = 5
    weather_data[:, 0, 12, :, 1] = 135
    weather_data[:, 12, 0, :, 0] = 15
    weather_data[:, 12, 0, :, 1] = 315
    weather_data[:, 12, 12, :, 0] = 20
    weather_data[:, 12, 12, :, 1] = 45

    result = np.zeros((1, 13, 13, 3, 3))
    result[:, :, :, :, 0] = weather_data[:, :, :, :, 0]

    radians = np.deg2rad(weather_data[:, :, :, :, 1])
    # Calculate cos and sin of wind direction
    result[:, :, :, :, 1] = np.cos(radians)
    result[:, :, :, :, 2] = np.sin(radians)

    return result


def plot_weather_data_medium(
    weather_data: np.ndarray,
    time: str,
    height: str = "all",
    grid_spacing=1,
    arrow_length_scale=1 / 20,
    arrow_width=0.23,
    margin=1,
    show_background: bool = True,
    custom_name: str = None,
):
    assert weather_data.shape == (13, 13, 3, 3)
    assert np.all(
        weather_data[:, :, :, 0] >= 0
    )  # Assert positive wind speed, should not be scaled

    match height:
        case "10m":
            weather_data = weather_data[:, :, 0, :]
        case "80m":
            weather_data = weather_data[:, :, 1, :]
        case "120m":
            weather_data = weather_data[:, :, 2, :]
        case "avg":
            weather_data = np.average(weather_data, axis=3)
        case "all":
            pass
        case _:
            raise ValueError()

    n_latitude, n_longitude = weather_data.shape[0:2]

    # Reduced figure size by half
    fig_height, fig_width = 2.6, 2.6  # 2.0, 2.0
    (fig, ax) = plt.subplots(1, 1, layout="constrained")
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)

    x_tails = np.arange(0, n_longitude * grid_spacing, grid_spacing)
    y_tails = np.arange(0, n_latitude * grid_spacing, grid_spacing)

    if height == "all":
        u = (
            weather_data[:, :, :, 0] * arrow_length_scale * weather_data[:, :, :, 2]
        )  # sine
        v = (
            weather_data[:, :, :, 0] * arrow_length_scale * weather_data[:, :, :, 1]
        )  # cosine

        ax.quiver(
            x_tails,
            y_tails,
            u[:, :, 0],
            v[:, :, 0],
            angles="xy",
            scale_units="xy",
            scale=1,
            headwidth=arrow_width * grid_spacing / arrow_length_scale,
            headlength=0.275 * grid_spacing / arrow_length_scale,
            fc="red",
            ec="none",  # Remove border
            linewidth=0,
            label="10m",
        )
        ax.quiver(
            x_tails,
            y_tails,
            u[:, :, 1],
            v[:, :, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            headwidth=arrow_width * grid_spacing / arrow_length_scale,
            headlength=0.275 * grid_spacing / arrow_length_scale,
            fc="green",
            ec="none",  # Remove border
            linewidth=0,
            label="80m",
        )
        ax.quiver(
            x_tails,
            y_tails,
            u[:, :, 2],
            v[:, :, 2],
            angles="xy",
            scale_units="xy",
            scale=1,
            headwidth=arrow_width * grid_spacing / arrow_length_scale,
            headlength=0.275 * grid_spacing / arrow_length_scale,
            fc="blue",
            ec="none",  # Remove border
            linewidth=0,
            label="120m",
        )
    else:
        u = weather_data[:, :, 0] * arrow_length_scale * weather_data[:, :, 2]  # sine
        v = weather_data[:, :, 0] * arrow_length_scale * weather_data[:, :, 1]  # cosine

        ax.quiver(
            x_tails,
            y_tails,
            u,
            v,
            angles="xy",
            scale_units="xy",
            scale=1,
            headwidth=arrow_width * grid_spacing / arrow_length_scale,
            headlength=0.275 * grid_spacing / arrow_length_scale,
            fc="blue",
            ec="none",  # Remove border
            linewidth=0,
            label=height,
        )

    x_end = -margin, (n_longitude - 1) * grid_spacing + margin
    y_end = -margin, (n_latitude - 1) * grid_spacing + margin
    ax.set_xlim(x_end)
    ax.set_ylim(y_end)

    if show_background:
        background = plt.imread(SHADOW_MAP_35_BY_35_PATH)
        ax.imshow(background, extent=[*x_end, *y_end], alpha=0.4)

    # Set ticks to show first, center, and last values
    x_center_idx = n_longitude // 2
    y_center_idx = n_latitude // 2
    ax.set_xticks([x_tails[0], x_tails[x_center_idx], x_tails[-1]])
    ax.set_xticklabels(
        ["10.07", "10.38", "10.69"],
        rotation=0,
        fontsize=8,
    )

    ax.set_yticks([y_tails[0], y_tails[y_center_idx], y_tails[-1]])
    ax.set_yticklabels(
        ["64.00", "64.13", "64.27"],
        rotation=90,
        fontsize=8,
    )

    ax.set_aspect("equal")

    # Remove axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.grid(True)

    if height == "all":
        ax.legend(
            loc="center right",
            ncol=3,
            bbox_to_anchor=(0.925, 1.055),
            fontsize=6,
            handles=[
                patches.Patch(color="red", label="10m"),
                patches.Patch(color="green", label="80m"),
                patches.Patch(color="blue", label="120m"),
            ],
        )
    else:
        ax.legend(
            loc="center right",
            bbox_to_anchor=(0.925, 1.055),
            fontsize=6,
            handles=[patches.Patch(color="blue", label=height)],
        )

    if custom_name is None:
        custom_name = f"wind_plot_{str(time)[:-6]}.pdf"
    plt.savefig(custom_name, format="pdf")
    plt.close()


def plot_weather_data_small(
    weather_data: np.ndarray,
    time: str,
    height: str = "all",
    grid_spacing=1,
    arrow_length_scale=1 / 20,
    arrow_width=0.15,
    margin=1,
    show_background: bool = False,
    custom_name: str = None,
):
    assert weather_data.shape == (13, 13, 3, 3)
    assert np.all(
        weather_data[:, :, :, 0] >= 0
    )  # Assert positive wind speed, should not be scaled

    match height:
        case "10m":
            weather_data = weather_data[:, :, 0, :]
        case "80m":
            weather_data = weather_data[:, :, 1, :]
        case "120m":
            weather_data = weather_data[:, :, 2, :]
        case "avg":
            weather_data = np.average(weather_data, axis=3)
        case "all":
            pass
        case _:
            raise ValueError()

    n_latitude, n_longitude = weather_data.shape[0:2]

    # Reduced figure size by half
    fig_height, fig_width = 2.0, 2.0  # 2.5, 2.5
    (fig, ax) = plt.subplots(1, 1, layout="constrained")
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)

    x_tails = np.arange(0, n_longitude * grid_spacing, grid_spacing)
    y_tails = np.arange(0, n_latitude * grid_spacing, grid_spacing)

    if height == "all":
        u = (
            weather_data[:, :, :, 0] * arrow_length_scale * weather_data[:, :, :, 2]
        )  # sine
        v = (
            weather_data[:, :, :, 0] * arrow_length_scale * weather_data[:, :, :, 1]
        )  # cosine

        ax.quiver(
            x_tails,
            y_tails,
            u[:, :, 0],
            v[:, :, 0],
            angles="xy",
            scale_units="xy",
            scale=1,
            headwidth=arrow_width * grid_spacing / arrow_length_scale,
            headlength=0.275 * grid_spacing / arrow_length_scale,
            fc="red",
            ec="none",  # Remove border
            linewidth=0,
            label="10m",
        )
        ax.quiver(
            x_tails,
            y_tails,
            u[:, :, 1],
            v[:, :, 1],
            angles="xy",
            scale_units="xy",
            scale=1,
            headwidth=arrow_width * grid_spacing / arrow_length_scale,
            headlength=0.275 * grid_spacing / arrow_length_scale,
            fc="green",
            ec="none",  # Remove border
            linewidth=0,
            label="80m",
        )
        ax.quiver(
            x_tails,
            y_tails,
            u[:, :, 2],
            v[:, :, 2],
            angles="xy",
            scale_units="xy",
            scale=1,
            headwidth=arrow_width * grid_spacing / arrow_length_scale,
            headlength=0.275 * grid_spacing / arrow_length_scale,
            fc="blue",
            ec="none",  # Remove border
            linewidth=0,
            label="120m",
        )
    else:
        u = weather_data[:, :, 0] * arrow_length_scale * weather_data[:, :, 2]  # sine
        v = weather_data[:, :, 0] * arrow_length_scale * weather_data[:, :, 1]  # cosine

        ax.quiver(
            x_tails,
            y_tails,
            u,
            v,
            angles="xy",
            scale_units="xy",
            scale=1,
            headwidth=arrow_width * grid_spacing / arrow_length_scale,
            headlength=0.275 * grid_spacing / arrow_length_scale,
            fc="blue",
            ec="none",  # Remove border
            linewidth=0,
            label=height,
        )

    x_end = -margin, (n_longitude - 1) * grid_spacing + margin
    y_end = -margin, (n_latitude - 1) * grid_spacing + margin
    ax.set_xlim(x_end)
    ax.set_ylim(y_end)

    # Set ticks to show first, center, and last values
    x_center_idx = n_longitude // 2
    y_center_idx = n_latitude // 2
    ax.set_xticks([x_tails[0], x_tails[x_center_idx], x_tails[-1]])
    ax.set_xticklabels(
        [
            f"{np.linspace(10.07, 10.69, n_longitude)[0]:.2f}",
            f"{np.linspace(10.07, 10.69, n_longitude)[x_center_idx]:.2f}",
            f"{np.linspace(10.07, 10.69, n_longitude)[-1]:.2f}",
        ],
        rotation=0,
        fontsize=8,
    )

    ax.set_yticks([y_tails[0], y_tails[y_center_idx], y_tails[-1]])
    ax.set_yticklabels(
        [
            f"{np.linspace(64.00, 64.27, n_latitude)[0]:.2f}",
            f"{np.linspace(64.00, 64.27, n_latitude)[y_center_idx]:.2f}",
            f"{np.linspace(64.00, 64.27, n_latitude)[-1]:.2f}",
        ],
        rotation=90,
        fontsize=8,
    )

    ax.set_aspect("equal")

    # Remove axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.grid(True)

    # if height == "all":
    #     ax.legend(
    #         loc="center right",
    #         ncol=3,
    #         bbox_to_anchor=(1.0, 1.05),
    #         fontsize=6,
    #         handles=[
    #             patches.Patch(color="red", label="10m"),
    #             patches.Patch(color="green", label="80m"),
    #             patches.Patch(color="blue", label="120m"),
    #         ],
    #     )
    # else:
    #     ax.legend(
    #         loc="center right",
    #         bbox_to_anchor=(1.0, 1.05),
    #         fontsize=6,
    #         handles=[patches.Patch(color="blue", label=height)],
    #     )

    if custom_name is None:
        custom_name = f"wind_plot_{str(time)[:-6]}.pdf"
    plt.savefig(custom_name, format="pdf")
    plt.close()


if __name__ == "__main__":

    """datasets = load_cross_val(TRAIN_DATA_PATH, TRAIN_SIZE, VAL_SIZE)

    weather_data = datasets.x
    timestamps = datasets.timestamps

    time_index = timestamps.searchsorted("2024-01-20 17:00:00")
    assert (
        time_index != timestamps.shape[0] and time_index != 0
    ), "Timestamp not in range"

    local_timestamps = pd.date_range(
        timestamps[time_index] - pd.Timedelta(hours=5),  # type: ignore[arg-type]
        timestamps[time_index] + pd.Timedelta(hours=5),
        freq="h",
    )  # type: ignore[arg-type]
    print(local_timestamps)

    # plot_weather_data_time(weather_data[time_index, :], local_timestamps, height="all")

    temp_data, _ = datasets.get_train_val(0, scale=False)
    temp = temp_data.x.numpy()
    # local_timestamps = pd.date_range(
    #     timestamps[temp_data.timestamps[123]] - pd.Timedelta(hours=5),  # type: ignore[arg-type]
    #     timestamps[temp_data.timestamps[123]] + pd.Timedelta(hours=5),
    #     freq="h",
    # )  # type: ignore[arg-type]
    print(temp_data.timestamps[123])

    plot_weather_data_new(temp[123, 0], local_timestamps[0], height="all")
    # plot_weather_data_time(temp[123, :], local_timestamps, height="all")"""

    batch_size = 7007
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = load_cross_val(TRAIN_DATA_PATH, TRAIN_SIZE, VAL_SIZE)
    test_data = load_scale_test_dataset(None, TEST_DATA_PATH, device)
    # test_data = load_scale_test_dataset(
    #     datasets.wind_speed_scaler_full, TEST_DATA_PATH, device
    # )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=False
    )
    for x, _, _ in test_dataloader:
        x_tensor = x.to(device)
    print(x_tensor.shape)

    flag = False

    for date_time_stamp in [
        "2024-07-22 03",
    ]:
        time_stamp_ex = pd.to_datetime(date_time_stamp)
        time_index = test_data.timestamps.searchsorted(time_stamp_ex)

        # if flag:
        #     x_tensor[time_index, 5, :, :, :, 0] = 15
        #     x_tensor[time_index, 5, :, :, :, 1] = -np.sqrt(2) / 2
        #     x_tensor[time_index, 5, :, :, :, 2] = np.sqrt(2) / 2

        plot_weather_data_small(
            x_tensor[time_index][5].detach().cpu().numpy(),
            time_stamp_ex,
            height="all",
            arrow_length_scale=1.5 / 20,
            arrow_width=0.3,
        )
        flag = True
