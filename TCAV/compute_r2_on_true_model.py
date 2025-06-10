import __fix_relative_imports  # noqa: F401
import torch
import numpy as np
import pandas as pd
import pickle
import sys
from scipy.stats import pearsonr

from mscEidalVesetrud.TCAV.vizualize_coefficient_of_determination import (
    plot_tcav_concepts_3d,
)
from mscEidalVesetrud.data_preprocessing.cross_validation import (
    TimestampDataset,
)
from mscEidalVesetrud.deep_learning.neural_nets import ConvNet3D, DenseNN
from mscEidalVesetrud.global_constants import (
    CONTAINING_FOLDER,
    SEEDS,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    TRAIN_SIZE,
    VAL_SIZE,
    BASELINE_FEATURES_TRAIN_PATH,
    BASELINE_FEATURES_TEST_PATH,
)

from mscEidalVesetrud.TCAV.cav_calculation import (
    ActivationTracker,
    CAVData,
    CAVSignTestR2Data,
    calculate_cavs,
    sign_test_cavs,
)
from mscEidalVesetrud.deep_learning.train_model import load_checkpoint
from mscEidalVesetrud.data_preprocessing.prepare_load_dataset import (
    load_cross_val,
    load_scale_test_dataset,
)


def load_baseline_features(timestamps: pd.DatetimeIndex) -> dict[str, np.ndarray]:

    df_train = pd.read_csv(BASELINE_FEATURES_TRAIN_PATH, index_col=0)
    df_train.index = pd.to_datetime(df_train.index)
    df_test = pd.read_csv(BASELINE_FEATURES_TEST_PATH, index_col=0)
    df_test.index = pd.to_datetime(df_test.index)

    df = pd.concat([df_train, df_test])

    df = df.reindex(timestamps)

    result: dict[str, np.ndarray] = {}

    for column in df.columns:
        result[column] = df[column].to_numpy(dtype=np.float32)
    return result


def add_random_concepts(
    concept_data: dict[str, np.ndarray],
    x: np.ndarray,
    n_randoms: int,
    seed_end: int,
    with_adjustment: bool,
) -> tuple[dict[str, np.ndarray], list[str]]:

    random_linear_combination_names = [
        f"random_linear_combination_{seed}" for seed in range(n_randoms)
    ]
    for seed in range(n_randoms):
        np.random.seed(seed)

        # 1 as size on first dimension to ensure correct broadcasting
        random_linear_combination = np.random.normal(
            loc=0, scale=1, size=(1, 11, 13, 13, 3, 3)
        )

        if with_adjustment:
            # The sum over the speed, cosine and sine dimension for all heights should be zero
            # If it is not, this linear combination will be correlated with the average wind speed for example
            for i in range(3):
                for j in range(3):
                    random_linear_combination[:, :, :, :, i, j] -= np.mean(
                        random_linear_combination[:, :, :, :, i, j]
                    )

        # Broadcast random_linear_combination over the first axis of x,
        # do elementwise multiplication, and sum over the rest of the axes
        concept_data[random_linear_combination_names[seed]] = np.sum(
            x * random_linear_combination, axis=(1, 2, 3, 4, 5)
        )

    np.random.seed(seed_end)
    # concept_data["random"] = np.random.normal(0, 1, size=x.shape[0])

    return concept_data, random_linear_combination_names


def save_concept_data(
    train_data: TimestampDataset,
    test_data: TimestampDataset,
    wind_speed_scaler,
    n_randoms: int,
    with_adjustment: bool,
    save_name: str,
):

    concept_data_train = load_baseline_features(train_data.timestamps)
    concept_data_test = load_baseline_features(test_data.timestamps)

    concept_data_train, random_names = add_random_concepts(
        concept_data_train,
        train_data.x.detach().cpu().numpy(),
        n_randoms,
        SEEDS[0],
        with_adjustment,
    )
    print("Train", with_adjustment)
    concept_data_test, _ = add_random_concepts(
        concept_data_test,
        test_data.x.detach().cpu().numpy(),
        n_randoms,
        SEEDS[0],
        with_adjustment,
    )
    print("Test", with_adjustment)

    # This confirms the concept matches the input data
    check_spatial_smoothing_120m(
        concept_data_test,
        wind_speed_scaler.inverse_transform(test_data.x.detach().cpu().numpy()),
    )
    check_spatial_smoothing_120m(
        concept_data_train,
        wind_speed_scaler.inverse_transform(train_data.x.detach().cpu().numpy()),
    )

    with open(
        f"{CONTAINING_FOLDER}/mscEidalVesetrud/TCAV/{save_name}.pkl",
        "wb",
    ) as file:
        pickle.dump(concept_data_train, file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(concept_data_test, file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(random_names, file, pickle.HIGHEST_PROTOCOL)


def load_concept_data(
    save_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    with open(
        f"{CONTAINING_FOLDER}/mscEidalVesetrud/TCAV/{save_name}.pkl", "rb"
    ) as file:
        concept_data_train = pickle.load(file)
        concept_data_test = pickle.load(file)
        random_names = pickle.load(file)

    return concept_data_train, concept_data_test, random_names


def check_spatial_smoothing_120m(
    concept_data: dict[str, np.ndarray], wind_input: np.ndarray
):
    smooth_concept_data = concept_data["wind_speed_120m_spatial_smoothing"]
    input_smoothed = np.mean(wind_input[:, 5, :, :, 2, 0], axis=(1, 2))

    assert np.sum(np.abs(smooth_concept_data - input_smoothed)) < 0.01


def load_best_model(device: str):
    model_path = f"{CONTAINING_FOLDER}/mscEidalVesetrud/models_test/final-model-sith-emperor-57.pth"

    (
        model,
        _,
        _,
        epoch,
        _,
        mae,
    ) = load_checkpoint(model_path, device)
    print(f"Loaded model type {model.class_name} with epoch {epoch}, mae {mae}")
    model.eval()
    return model


def load_model(device: str, epoch: int):

    np.random.seed(SEEDS[0])
    torch.manual_seed(SEEDS[0])
    model_path = (
        f"{CONTAINING_FOLDER}/models/final-model-proud-monkey-59-epoch-{epoch}.pth"
    )

    (
        model,
        _,
        _,
        epoch,
        _,
        mae,
    ) = load_checkpoint(model_path, device)
    print(f"Loaded model type {model.class_name} with epoch {epoch}, mae {mae}")
    model.eval()
    return model


def load_scaled_data(device: str):

    train_datasets = load_cross_val(TRAIN_DATA_PATH, TRAIN_SIZE, VAL_SIZE)
    wind_speed_scaler = train_datasets.wind_speed_scaler_full

    test_data = load_scale_test_dataset(wind_speed_scaler, TEST_DATA_PATH, device)

    train_data = TimestampDataset(
        wind_speed_scaler.transform(train_datasets.x),
        train_datasets.y,
        train_datasets.timestamps,
        device,
    )
    return train_data, test_data, wind_speed_scaler


def print_correlation_randoms(
    concept_data: dict[str, np.ndarray], random_names: list[str]
):

    height_avg_abs_correlation = {
        10: 0,
        80: 0,
        120: 0,
    }

    avg_max_abs_correlation = 0

    for name in random_names:
        max_abs_correlation = 0
        for height in [10, 80, 120]:
            result = pearsonr(
                concept_data[f"wind_speed_{height}m_spatial_std"],
                concept_data[name],
            )
            height_avg_abs_correlation[height] += abs(result.statistic)
            max_abs_correlation = max(max_abs_correlation, abs(result.statistic))
        avg_max_abs_correlation += max_abs_correlation

    print(avg_max_abs_correlation / len(random_names))

    for height in height_avg_abs_correlation.keys():
        height_avg_abs_correlation[height] /= len(random_names)
    print(height_avg_abs_correlation)


def calculate_and_save_cav_r2_sign_test(
    train_data: TimestampDataset,
    test_data: TimestampDataset,
    concept_data_train: dict[str, np.ndarray],
    concept_data_test: dict[str, np.ndarray],
    random_names: list[str],
    device: str,
    l2_regularization: float,
):
    model = load_best_model(device)

    input_data_train: tuple[torch.Tensor, torch.Tensor] = (
        train_data.x,
        train_data.x_time,
    )
    input_data_test: tuple[torch.Tensor, torch.Tensor] = (
        test_data.x,
        test_data.x_time,
    )

    # [
    #     "dense_layer_0",
    #     "dense_batchnorm_0",
    #     "dense_silu_0",
    #     "dense_dropout_0",
    #     "dense_layer_1",
    #     "dense_batchnorm_1",
    #     "dense_silu_1",
    #     "dense_dropout_1",
    #     "dense_layer_2",
    #     "dense_batchnorm_2",
    #     "dense_silu_2",
    #     "dense_dropout_2",
    #     "dense_layer_3",
    #     "dense_batchnorm_3",
    #     "dense_silu_3",
    #     "dense_dropout_3",
    #     "dense_layer_final",
    #     "final_activation",
    # ]

    layer_names = ["dense_silu_0", "dense_silu_1", "dense_silu_2", "dense_silu_3"]

    sign_test_result, cav_datas, num_tests, random_r2_scores = sign_test_cavs(
        model,
        layer_names,
        input_data_train,
        input_data_test,
        concept_data_train,
        concept_data_test,
        random_names,
        batch_size=4096,
        l2_regularization=l2_regularization,
        use_sgd=False,
        normalize_cav=False,
        verbose=True,
    )

    with open(f"{CONTAINING_FOLDER}/mscEidalVesetrud/TCAV/sign_test.pkl", "wb") as file:
        pickle.dump(sign_test_result, file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cav_datas, file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(random_r2_scores, file, pickle.HIGHEST_PROTOCOL)


def calculate_and_save_cav_r2_during_training(
    train_data: TimestampDataset,
    test_data: TimestampDataset,
    concept_data_train: dict[str, np.ndarray],
    concept_data_test: dict[str, np.ndarray],
    device: str,
    l2_regularization: float,
):

    layer_names = ["dense_silu_0", "dense_silu_1", "dense_silu_2", "dense_silu_3"]

    input_data_train: tuple[torch.Tensor, torch.Tensor] = (
        train_data.x,
        train_data.x_time,
    )
    input_data_test: tuple[torch.Tensor, torch.Tensor] = (
        test_data.x,
        test_data.x_time,
    )

    cav_datas_all_models: dict[str, dict[str, dict[str, CAVData]]] = {}

    epochs = [0, 10, 20, 30, 41]
    for i, epoch in enumerate(epochs):
        print(f"Run [{i}/{len(epochs)}]")
        model = load_model(device, epoch)

        with ActivationTracker(
            model, layer_names, batch_size=4096
        ) as activation_tracker:
            cav_datas_all_models[f"{epoch}"] = calculate_cavs(
                activation_tracker,
                layer_names,
                input_data_train,
                concept_data_train,
                input_data_test,
                concept_data_test,
                l2_regularization=l2_regularization,
                use_sgd=False,
                normalize_cav=False,
                verbose=False,
            )

    save_name = "cav_during_training"
    with open(
        f"{CONTAINING_FOLDER}/mscEidalVesetrud/TCAV/{save_name}.pkl",
        "wb",
    ) as file:
        pickle.dump(cav_datas_all_models, file, pickle.HIGHEST_PROTOCOL)

    return save_name


def calculate_and_save_cav_r2_random_model_weights(
    train_data: TimestampDataset,
    test_data: TimestampDataset,
    concept_data_train: dict[str, np.ndarray],
    concept_data_test: dict[str, np.ndarray],
    device: str,
    l2_regularization: float,
) -> str:

    # "PC_wind_speed_120m_1"
    # "PC_wind_speed_120m_2"
    # "PC_wind_speed_120m_3"
    # "wind_speed_120m_roan_temporal_variance_7"
    # "wind_speed_120m_spatial_std"
    # "wind_speed_120m_roan_lag_0"
    # "hour"

    layer_names = ["dense_silu_0", "dense_silu_1", "dense_silu_2", "dense_silu_3"]

    input_data_train: tuple[torch.Tensor, torch.Tensor] = (
        train_data.x,
        train_data.x_time,
    )
    input_data_test: tuple[torch.Tensor, torch.Tensor] = (
        test_data.x,
        test_data.x_time,
    )

    concepts = list(concept_data_test.keys())

    cav_datas_all_models: dict[str, dict[str, dict[str, CAVData]]] = {}

    model = ConvNet3D(
        **{
            "conv_channels": 77,
            "dense_layer_size": 80,
            "n_dense_layers": 4,
            "p_dropout": 0.3,
            "architecture": "4_avgpool_3",
            "final_activation": "hardsigmoid",
        }
    ).to(device)
    model.eval()

    with ActivationTracker(model, layer_names, batch_size=4096) as activation_tracker:
        temp = calculate_cavs(
            activation_tracker,
            layer_names,
            input_data_train,
            concept_data_train,
            input_data_test,
            concept_data_test,
            l2_regularization=l2_regularization,
            use_sgd=False,
            normalize_cav=False,
            verbose=False,
        )
        for key in concepts:
            for layer in layer_names:
                temp[key][layer[-1]] = temp[key].pop(layer)
        cav_datas_all_models["a"] = temp

    model = ConvNet3D(
        **{
            "conv_channels": 77,
            "dense_layer_size": 80,
            "n_dense_layers": 4,
            "p_dropout": 0.3,
            "architecture": "5_5",
            "final_activation": "hardsigmoid",
        }
    ).to(device)
    model.eval()

    with ActivationTracker(model, layer_names, batch_size=4096) as activation_tracker:
        temp = calculate_cavs(
            activation_tracker,
            layer_names,
            input_data_train,
            concept_data_train,
            input_data_test,
            concept_data_test,
            l2_regularization=l2_regularization,
            use_sgd=False,
            normalize_cav=False,
            verbose=False,
        )
        for key in concepts:
            for layer in layer_names:
                temp[key][layer[-1]] = temp[key].pop(layer)
        cav_datas_all_models["b"] = temp

    model = DenseNN(128, 4, 0.2, "hardsigmoid").to(device).to(device)
    model.eval()
    layer_names = ["silu_0", "silu_1", "silu_2", "silu_3"]

    with ActivationTracker(model, layer_names, batch_size=4096) as activation_tracker:
        temp = calculate_cavs(
            activation_tracker,
            layer_names,
            input_data_train,
            concept_data_train,
            input_data_test,
            concept_data_test,
            l2_regularization=l2_regularization,
            use_sgd=False,
            normalize_cav=False,
            verbose=False,
        )
        for key in concepts:
            for layer in layer_names:
                temp[key][layer[-1]] = temp[key].pop(layer)
        cav_datas_all_models["c"] = temp

    save_name = "cav_random_models"
    with open(
        f"{CONTAINING_FOLDER}/mscEidalVesetrud/TCAV/{save_name}.pkl",
        "wb",
    ) as file:
        pickle.dump(cav_datas_all_models, file, pickle.HIGHEST_PROTOCOL)
    return save_name


def load_r2_sign_test() -> tuple[
    dict[str, dict[str, CAVSignTestR2Data]],
    dict[str, dict[str, list[CAVData]]],
    dict[str, np.ndarray],
]:
    with open(f"{CONTAINING_FOLDER}/mscEidalVesetrud/TCAV/sign_test.pkl", "rb") as file:
        sign_test_result = pickle.load(file)
        cav_datas = pickle.load(file)
        random_r2_scores = pickle.load(file)
    return sign_test_result, cav_datas, random_r2_scores


def display_r2(save_name: str):
    with open(
        f"{CONTAINING_FOLDER}/mscEidalVesetrud/TCAV/{save_name}.pkl",
        "rb",
    ) as file:
        cav_datas = pickle.load(file)

    # cav_datas is on the wrong format, containing a list of R² values
    plot_tcav_concepts_3d(cav_datas, save_folder=f"{CONTAINING_FOLDER}/TCAV_plots/")


def print_sign_test_results(significance_level):

    sign_test_result, cav_datas, random_r2_scores = load_r2_sign_test()
    r2_for_layers = {}
    pval_for_layers = {}

    sign_test_result_list: list[tuple[str, str, CAVSignTestR2Data]] = []
    for concept, value in sign_test_result.items():
        for layer, result in value.items():
            sign_test_result_list.append((concept, layer, result))

            if layer not in r2_for_layers:
                r2_for_layers[layer] = []
                pval_for_layers[layer] = []
            r2_for_layers[layer].append(result.coefficient_of_determination)
            pval_for_layers[layer].append(result.r2_greater_than_random_pvalue)

    sign_test_result_list.sort(key=lambda x: -x[2].coefficient_of_determination)
    sign_test_result_list.sort(key=lambda x: x[1])

    old_stdout = sys.stdout
    log_file = open(f"{CONTAINING_FOLDER}/mscEidalVesetrud/TCAV/test_output.txt", "w")
    sys.stdout = log_file

    for concept, layer, result in sign_test_result_list:
        print(concept, ":", layer)
        print(
            f"R² {result.coefficient_of_determination:.04f} Sign-test P-val: {result.r2_greater_than_random_pvalue:.4e} [{result.num_random_r2_less}, {result.num_random_r2_greater}]\n"
        )

    print("Random score distributions")
    for layer, scores in random_r2_scores.items():
        print(layer)
        print(
            f"Avg {np.mean(scores)} Std {np.std(scores)} Max {np.max(scores)} Min {np.min(scores)}"
        )

    print(
        '\nLayer: "Number of significant features" "Number of insignificant features" "Average R2 value"'
    )
    for layer, pval in pval_for_layers.items():
        print(
            f"{layer}: {np.sum(np.array(pval) < significance_level)} {np.sum(np.array(pval) >= significance_level)}, Avg {np.average(r2_for_layers[layer]):.04f}"
        )

    grouped_results_height: dict[str, dict[str, list[CAVSignTestR2Data]]] = {
        "10m": {name: [] for name in r2_for_layers.keys()},
        "80m": {name: [] for name in r2_for_layers.keys()},
        "120m": {name: [] for name in r2_for_layers.keys()},
    }
    grouped_results_type: dict[str, dict[str, list[CAVSignTestR2Data]]] = {
        "spatial_std": {name: [] for name in r2_for_layers.keys()},
        "spatial_smoothing": {name: [] for name in r2_for_layers.keys()},
        "temporal_variance": {name: [] for name in r2_for_layers.keys()},
        "PC_1": {name: [] for name in r2_for_layers.keys()},
        "PC_rest": {name: [] for name in r2_for_layers.keys()},
        "lags_and_leads": {name: [] for name in r2_for_layers.keys()},
        "hour": {name: [] for name in r2_for_layers.keys()},
    }

    for i, grouped_results in enumerate([grouped_results_height, grouped_results_type]):
        if i == 0:
            print("\nResults grouped on height")
        else:
            print("\nResults grouped on feature type")

        for concept, layer, result in sign_test_result_list:
            for group in grouped_results.keys():
                if group[:2] != "PC" and group != "lags_and_leads":
                    if concept.find(group) != -1:
                        grouped_results[group][layer].append(result)
                        break
                elif group == "lags_and_leads":
                    if concept.find("lag") != -1 or concept.find("lead") != -1:
                        grouped_results[group][layer].append(result)
                        break
                elif group[-1] == "1":
                    if concept.find("PC") != -1 and (
                        concept[-2:] == "_1" or concept[-2:] == "_2"
                    ):
                        grouped_results[group][layer].append(result)
                        break
                else:
                    if concept.find("PC") != -1:
                        grouped_results[group][layer].append(result)
                        break
        grouped_results_summary: dict[str, dict[str, str]] = {
            name: {} for name in grouped_results.keys()
        }

        for group, grouped_result in grouped_results.items():
            for layer, result in grouped_result.items():
                pvalues = np.array([r.r2_greater_than_random_pvalue for r in result])
                r2 = np.array([r.coefficient_of_determination for r in result])
                grouped_results_summary[group][
                    layer
                ] = f"n_significant: {np.sum(pvalues < significance_level)} n_insignificant: {np.sum(pvalues >= significance_level)} Avg R² {np.average(r2):.04f} [{np.min(r2):.04f}, {np.max(r2):.04f}] Std R² {np.std(r2):.04f}"

        for group, grouped_result in grouped_results_summary.items():
            for layer, result in grouped_result.items():
                print()
                print(group, ":", layer)
                print(result)

    print("\nDecreasing score results")
    for concept, value in sign_test_result.items():
        prev_layer = list(value.keys())[0]
        decrease = True
        for layer in list(value.keys())[1:]:
            if (
                value[prev_layer].coefficient_of_determination
                <= value[layer].coefficient_of_determination
            ):
                decrease = False
                print(
                    "###",
                    concept,
                    [r.coefficient_of_determination for r in value.items()],
                )
                break
        print(concept, decrease)

    sys.stdout = old_stdout
    log_file.close()


def get_lags_leads():
    sign_test_result, cav_datas, random_r2_scores = load_r2_sign_test()

    lags_and_leads = {}
    for concept, value in sign_test_result.items():
        for layer, result in value.items():
            if layer not in lags_and_leads.keys():
                lags_and_leads[layer] = []

            if concept[-5:-1] == "lag_":
                lags_and_leads[layer].append(
                    (
                        -int(concept[-1]),
                        result.coefficient_of_determination,
                    )
                )
            if concept[-6:-1] == "lead_":
                lags_and_leads[layer].append(
                    (
                        int(concept[-1]),
                        result.coefficient_of_determination,
                    )
                )

    merged_r2: dict[str, dict[int, (float, float)]] = {
        layer: {} for layer in lags_and_leads.keys()
    }

    for layer, values in lags_and_leads.items():
        merged_r2[layer] = {i: [] for i in range(-3, 4)}
        for t, r2 in values:
            merged_r2[layer][t].append(r2)

        for t in range(-3, 4):
            merged_r2[layer][t] = (
                np.average(merged_r2[layer][t]),
                np.std(merged_r2[layer][t]),
            )

    return lags_and_leads, merged_r2


def get_r2_per_layer():
    sign_test_result, cav_datas, random_r2_scores = load_r2_sign_test()
    r2_for_layers: dict[str, list[float]] = {}

    for concept, value in sign_test_result.items():
        for layer, result in value.items():
            if layer not in r2_for_layers:
                r2_for_layers[layer] = []
            r2_for_layers[layer].append(result.coefficient_of_determination)

    return r2_for_layers


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    np.random.seed(SEEDS[0])
    torch.manual_seed(SEEDS[0])

    train_data, test_data, wind_speed_scaler = load_scaled_data(device)

    n_randoms = 1000
    save_concept_data(
        train_data,
        test_data,
        wind_speed_scaler,
        n_randoms,
        with_adjustment=True,
        save_name="concept_data",
    )
    # save_concept_data(
    #     train_data,
    #     test_data,
    #     wind_speed_scaler,
    #     n_randoms,
    #     with_adjustment=False,
    #     save_name="concept_data_no_adjust",
    # )

    concept_data_train, concept_data_test, random_names = load_concept_data(
        "concept_data"
    )
    # concept_data_train_no_adjust, concept_data_test_no_adjust, _ = load_concept_data(
    #     "concept_data_no_adjust"
    # )

    l2_regularization = 0.5
    calculate_and_save_cav_r2_sign_test(
        train_data,
        test_data,
        concept_data_train,
        concept_data_test,
        random_names,
        device,
        l2_regularization,
    )

    print_sign_test_results(significance_level=0.0001)

    # print("Correlation adjust")
    # print_correlation_randoms(concept_data_test, random_names)
    # print("Correlation no adjust")
    # print_correlation_randoms(concept_data_test_no_adjust, random_names)
    """
    Correlation adjust
    0.12770361340650002
    {10: np.float64(0.08867896945286637), 80: np.float64(0.09860711072847095), 120: np.float64(0.0939552004895361)}
    Correlation no adjust
    0.32258803495453203
    {10: np.float64(0.25149551651707247), 80: np.float64(0.22978797362848133), 120: np.float64(0.22800784048972672)}
    """
    # Only using concept data R² from this point
    concept_names = list(set(concept_data_train.keys()) - set(random_names))
    concept_data_train_no_random = {
        name: concept_data_train[name] for name in concept_names
    }
    concept_data_test_no_random = {
        name: concept_data_test[name] for name in concept_names
    }

    save_name_during_training = calculate_and_save_cav_r2_during_training(
        train_data,
        test_data,
        concept_data_train_no_random,
        concept_data_test_no_random,
        device,
        l2_regularization,
    )
    # save_name_random_model_weights = calculate_and_save_cav_r2_random_model_weights(
    #     train_data,
    #     test_data,
    #     concept_data_train_no_random,
    #     concept_data_test_no_random,
    #     device,
    #     l2_regularization,
    # )

    # display_r2("cav_random_models")  # save_name_random_model_weights
    display_r2("cav_during_training")  # save_name_during_training
