import __fix_relative_imports  # noqa: F401
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox
from matplotlib import cm
from mscEidalVesetrud.TCAV.cav_calculation import CAVData

from mscEidalVesetrud.deep_learning.train_model import (
    load_checkpoint,
)  # noqa: F401


matplotlib.use("QtAgg")

# Set the font globally
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Computer Modern Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "medium"
plt.rcParams["axes.titlesize"] = "medium"
plt.rcParams["xtick.labelsize"] = "medium"
plt.rcParams["ytick.labelsize"] = "medium"


def gridify_cav_data(cav_data: dict[str, dict[str, dict[str, CAVData]]]):
    model_epochs = list(cav_data.keys())
    concepts = list(cav_data[model_epochs[0]].keys())
    layers = list(cav_data[model_epochs[0]][concepts[0]].keys())

    grid_coefficient_of_determinations: dict[str, np.ndarray] = {
        concept: np.zeros((len(model_epochs), len(layers)), dtype=np.float32)
        for concept in concepts
    }

    for model_epoch_idx, model_result in enumerate(cav_data.values()):
        assert list(model_result.keys()) == concepts
        for concept, concept_result in model_result.items():
            assert list(concept_result.keys()) == layers
            for layer_idx, result in enumerate(concept_result.values()):

                grid_coefficient_of_determinations[concept][
                    model_epoch_idx, layer_idx
                ] = result.coefficient_of_determination

    return grid_coefficient_of_determinations, model_epochs, layers


def prettyfy_baseline_feature_name(name: str) -> str:
    if name == "hour":
        return name

    if name[:18] == "wind_direction_cos":
        name = "cos" + name[18:]
    elif name[:18] == "wind_direction_sin":
        name = "sin" + name[18:]

    name = name.replace("wind_direction", "wind dir")
    name = name.replace("wind_speed", "wind speed")
    name = name.replace("temporal_variance", "Temporal variance")
    name = name.replace("spatial_smoothing", "Spatial smoothing")
    name = name.replace("spatial_std", "Spatial std")

    names = name.split("_")

    try:
        names.remove("roan")
    except ValueError:
        pass

    if names[0] == "PC":
        names.pop(0)
        names.insert(2, "PC")
    if names[0] == "u":
        names[0] = "Azimuthal"
    elif names[0] == "v":
        names[0] = "Meridional"

    if names[-1].isdigit():
        names[-2] = names[-2] + " " + names[-1]
        names.pop()

    assert len(names) == 3

    return names[2] + "; " + names[0] + "; " + names[1]


def plot_single_tcav_concept_3d(
    concept: str,
    cod_data: np.ndarray,
    model_epochs: list[str],
    layers: list[str],
    save_name: str,
):
    assert (
        len(cod_data.shape) == 2
        and cod_data.shape[0] == len(model_epochs)
        and cod_data.shape[1] == len(layers)
    )

    # X is model_epochs
    # Y is layers
    minimum_z = min(-0.1, np.min(cod_data))

    # Cannot use constrained layout because the tick labels are long text and not short numbers
    fig = plt.figure(figsize=(2.37, 3.9))  # , constrained_layout=True)
    ax: Axes = fig.add_subplot(111, projection="3d")

    X = np.arange(0, len(model_epochs), 1, dtype=np.float32)
    Y = np.arange(0, len(layers), 1, dtype=np.float32)
    Y, X = np.meshgrid(Y, X)

    surface = ax.plot_surface(
        X,
        Y,
        cod_data,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
        vmin=minimum_z,
        vmax=1,
    )
    layer_nums = [int(layer[-1]) for layer in layers]

    ax.azim, ax.elev = -25, 25
    # ax.xaxis.set_rotate_label(False)
    # ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel("Training epochs")  # , labelpad=8)
    ax.set_ylabel("Model Layer")  # , labelpad=4)
    ax.set_zlabel("$R^2$")
    ax.set_xlim(0, len(model_epochs) - 0.75)
    ax.set_ylim(-0.25, len(layers) - 1)
    ax.set_zlim(minimum_z, 1)
    ax.set_xticks(range(len(model_epochs)))
    ax.set_xticklabels(model_epochs)  # , rotation=-25, ha="center")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layer_nums)  # , rotation=25, ha="center")
    ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])  # , rotation=25, ha="center")
    ax.zaxis.set_ticks_position("lower")
    ax.zaxis.set_label_position("lower")

    ax.set_box_aspect(
        [len(model_epochs) + 0.25, len(layers) + 0.25, (1 - minimum_z) * 3]
    )
    ax.set_title(prettyfy_baseline_feature_name(concept))

    # fig.colorbar(
    #     surface,
    #     ax=ax,
    #     shrink=0.75,
    #     aspect=20,
    #     location="bottom",
    #     label="Coefficient of Determination",
    # )

    # print(fig.get_tightbbox())
    plt.savefig(
        save_name,
        format="pdf",
        bbox_inches=Bbox(np.array([[-0.1, 0.655], [2.2, 2.655]])),
    )
    plt.close()
    # plt.show()


def plot_tcav_concepts_3d(
    cav_data: dict[str, dict[str, dict[str, CAVData]]], save_folder: str
):

    assert save_folder[-1] == "/"

    grid_coefficient_of_determinations, model_epochs, layers = gridify_cav_data(
        cav_data
    )

    model_epochs = ["0", "10", "20", "30", "41"]

    for concept, cod_data in grid_coefficient_of_determinations.items():
        plot_single_tcav_concept_3d(
            concept,
            cod_data,
            model_epochs,
            layers,
            save_folder + concept + ".pdf",
        )
