import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import Any


class IndexableModule(nn.Module):

    def __init__(self, final_activation: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        match final_activation:
            case "identity":
                self.final_activation = nn.Identity()
            case "sigmoid":
                self.final_activation = nn.Sigmoid()
            case "hardsigmoid":
                self.final_activation = nn.Hardsigmoid()
            case _:
                raise ValueError("Invalid final activation")

    def __getitem__(self, layer_name: str) -> nn.Module:
        raise NotImplementedError("Please Implement this method")

    def __setitem__(self, layer_name: str, module: nn.Module) -> nn.Module:
        raise NotImplementedError("Please Implement this method")

    @property
    def layers(self) -> list[str]:
        raise NotImplementedError("Please Implement this method")

    configuration: dict[str, Any]
    class_name: str

    def summary(
        self,
        input_size: tuple[int, ...] | list[tuple[int, ...]] = [
            (11, 13, 13, 3, 3),
            (5,),
        ],
        batch_size: int = -1,
        device: torch.device = "cpu",
    ):
        """
        Adapted from the torchsummary package, but made to work for LSTM layers, multiple inputs, and
        showing the layer names we give
        """

        summary_data = OrderedDict()
        handles: dict[str, torch.utils.hooks.RemovableHandle] = {}

        for layer_name in self.layers:
            # We take the layer_name as input to force python to save the current value of layer_name
            # as a default parameter value, rather than just referencing the layer_name variable directly
            # which will end up being the last value in layer_names by the end of the loop
            def hook(
                module: torch.nn.Module,
                input: tuple[torch.Tensor],
                output: torch.Tensor,
                layer_name=layer_name,
            ):
                summary_data[layer_name] = OrderedDict()

                summary_data[layer_name]["class_name"] = (
                    str(module.__class__).split(".")[-1].split("'")[0]
                )

                summary_data[layer_name]["input_shape"] = list(input[0].size())
                summary_data[layer_name]["input_shape"][0] = batch_size

                if isinstance(module, nn.LSTM):
                    summary_data[layer_name]["output_shape"] = list(output[0].size())
                    summary_data[layer_name]["output_shape"][0] = batch_size
                elif isinstance(output, (list, tuple)):
                    summary_data[layer_name]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary_data[layer_name]["output_shape"] = list(output.size())
                    summary_data[layer_name]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary_data[layer_name]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary_data[layer_name]["nb_params"] = params

            handles[layer_name] = self[layer_name].register_forward_hook(hook)

        if isinstance(input_size, tuple):
            input_size = [input_size]
        x = [
            torch.rand(2, *in_size, dtype=torch.float32, device=device)
            for in_size in input_size
        ]
        _ = self(*x)

        # Remove the hooks to prevent side effects
        for handle in handles.values():
            handle.remove()

        print(
            "---------------------------------------------------------------------------------------------"
        )
        line_new = "{:>5}  {:>20}  {:>20}  {:>25} {:>15}".format(
            "Order", "Layer name", "Layer type", "Output Shape", "Param #"
        )
        print(line_new)
        print(
            "============================================================================================="
        )
        total_params = 0
        total_output = 0
        trainable_params = 0
        contains_lstm = False
        for i, layer in enumerate(summary_data):
            if summary_data[layer]["class_name"] == "LSTM":
                contains_lstm = True

            line_new = "{:>5}  {:>20}  {:>20}  {:>25} {:>15}".format(
                i,
                layer,
                summary_data[layer]["class_name"],
                str(summary_data[layer]["output_shape"]),
                "{0:,}".format(summary_data[layer]["nb_params"]),
            )
            total_params += summary_data[layer]["nb_params"]
            total_output += np.prod(summary_data[layer]["output_shape"])
            if "trainable" in summary_data[layer] and summary_data[layer]["trainable"]:
                trainable_params += summary_data[layer]["nb_params"]
            print(line_new)

        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(input_size[0]) * batch_size * 4.0 / (1024**2.0))
        total_output_size = abs(
            2.0 * total_output * 4.0 / (1024**2.0)
        )  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4.0 / (1024**2.0))
        total_size = total_params_size + total_output_size + total_input_size

        print(
            "============================================================================================="
        )
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print(
            "---------------------------------------------------------------------------------------------"
        )
        if contains_lstm:
            print(
                "Size estimations does not consider LSTM layers correctly (underestimated)"
            )
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print(
            "---------------------------------------------------------------------------------------------"
        )


class DenseNN(IndexableModule):
    def __init__(
        self,
        layer_size: int,
        n_hidden_layers: int,
        p_dropout: float,
        final_activation: str,
    ):
        super(DenseNN, self).__init__(final_activation)

        self.configuration = {
            "layer_size": layer_size,
            "n_hidden_layers": n_hidden_layers,
            "p_dropout": p_dropout,
            "final_activation": final_activation,
        }
        self.class_name = "dense"

        layer_size = int(layer_size)
        n_hidden_layers = int(n_hidden_layers)

        self.p_dropout = p_dropout

        if n_hidden_layers == 0:
            dense_layers = []
            final = nn.Linear(13 * 13 * 3 * 3 + 10 * 3 * 3 + 5, 1)
        else:
            dense_layers = [nn.Linear(13 * 13 * 3 * 3 + 10 * 3 * 3 + 5, layer_size)]
            for _ in range(n_hidden_layers - 1):
                dense_layers.append(nn.Linear(layer_size, layer_size))
            final = nn.Linear(layer_size, 1)

        # Good initialization for layers ending in a ReLU like activation:
        for layer in dense_layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        # Good initialization for layers ending in a sigmoid or unit activation:
        nn.init.xavier_normal_(final.weight)

        # Not applying dropout to input features, as these are very correlated
        self.ordered_layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i, layer in enumerate(dense_layers):
            self.ordered_layers[f"layer_{i}"] = layer
            self.ordered_layers[f"batchnorm_{i}"] = nn.BatchNorm1d(layer_size)
            self.ordered_layers[f"silu_{i}"] = nn.SiLU()
            self.ordered_layers[f"dropout_{i}"] = nn.Dropout(p=p_dropout)

        self.ordered_layers["layer_final"] = final
        self.ordered_layers["final_activation"] = self.final_activation

        self.net = nn.Sequential(self.ordered_layers)

    def __getitem__(self, layer_name: str) -> nn.Module:
        return self.ordered_layers[layer_name]

    def __setitem__(self, layer_name: str, module: nn.Module) -> nn.Module:
        self.ordered_layers[layer_name] = module

    @property
    def layers(self) -> list[str]:
        return list(self.ordered_layers.keys())

    def forward(self, x: torch.Tensor, x_time: torch.Tensor):
        # The whole grid for the current time, flattened
        x_spatial = x[:, 5, :, :, :].view(x.shape[0], -1)

        # The center point of the grid for the previous and future times, flattened
        x_temporal = torch.cat([x[:, :5, 6, 6, :], x[:, 6:, 6, 6, :]], dim=1).view(
            x.shape[0], -1
        )

        # Not applying dropout to input features, as these are very correlated
        x = torch.cat([x_spatial, x_temporal, x_time], dim=1)

        x = self.net(x)
        return x


class ConvNet3D(IndexableModule):
    def __init__(
        self,
        conv_channels: int,
        dense_layer_size: int,
        n_dense_layers: int,
        p_dropout: float,
        architecture: str,
        final_activation: str,
    ):
        assert architecture in [
            "4_avgpool_3",
            "5_4_4",
            "5_3_3_3",
            "4_4_4",
            "5_5",
            "6_4_3",
            "7_3_3",
        ]
        assert n_dense_layers >= 1
        super(ConvNet3D, self).__init__(final_activation)

        self.configuration = {
            "conv_channels": conv_channels,
            "dense_layer_size": dense_layer_size,
            "n_dense_layers": n_dense_layers,
            "p_dropout": p_dropout,
            "architecture": architecture,
            "final_activation": final_activation,
        }
        self.class_name = "cnn"

        def __calc_size(*kernel_sizes: int):
            reduction = sum(kernel_sizes) - len(kernel_sizes)
            # + 5 correspond to x_time
            return (11 - reduction) * (13 - reduction) ** 2 * conv_channels + 5

        # Convolution layers
        match architecture:
            case "4_avgpool_3":
                # This architecture also includes AvgPooling, inserted later
                conv_layers = [
                    nn.Conv3d(9, conv_channels, kernel_size=4),
                    nn.Conv3d(conv_channels, conv_channels, kernel_size=3),
                ]
                conv_end_size = 18 * conv_channels + 5

            case "5_4_4":
                conv_layers = [
                    nn.Conv3d(9, conv_channels, kernel_size=5),
                    nn.Conv3d(conv_channels, conv_channels, kernel_size=4),
                    nn.Conv3d(conv_channels, conv_channels, kernel_size=4),
                ]
                conv_end_size = __calc_size(5, 4, 4)

            case "5_3_3_3":
                conv_layers = [
                    nn.Conv3d(9, conv_channels, kernel_size=5),
                    nn.Conv3d(conv_channels, conv_channels, kernel_size=3),
                    nn.Conv3d(conv_channels, conv_channels, kernel_size=3),
                    nn.Conv3d(conv_channels, conv_channels, kernel_size=3),
                ]
                conv_end_size = __calc_size(5, 3, 3, 3)

            case "4_4_4":
                conv_layers = [
                    nn.Conv3d(9, conv_channels, kernel_size=4),
                    nn.Conv3d(conv_channels, conv_channels, kernel_size=4),
                    nn.Conv3d(conv_channels, conv_channels, kernel_size=4),
                ]
                conv_end_size = __calc_size(4, 4, 4)

            case "5_5":
                conv_layers = [
                    nn.Conv3d(9, conv_channels, kernel_size=5),
                    nn.Conv3d(conv_channels, conv_channels, kernel_size=5),
                ]
                conv_end_size = __calc_size(5, 5)

            case "6_4_3":
                conv_layers = [
                    nn.Conv3d(9, conv_channels, kernel_size=6),
                    nn.Conv3d(conv_channels, conv_channels, kernel_size=4),
                    nn.Conv3d(conv_channels, conv_channels, kernel_size=3),
                ]
                conv_end_size = __calc_size(6, 4, 3)

            case "7_3_3":
                conv_layers = [
                    nn.Conv3d(9, conv_channels, kernel_size=7),
                    nn.Conv3d(conv_channels, conv_channels, kernel_size=3),
                    nn.Conv3d(conv_channels, conv_channels, kernel_size=3),
                ]
                conv_end_size = __calc_size(7, 3, 3)

        # Dense layers
        dense_layers = [nn.Linear(conv_end_size, dense_layer_size)]
        for _ in range(n_dense_layers - 1):
            dense_layers.append(nn.Linear(dense_layer_size, dense_layer_size))
        final_layer = nn.Linear(dense_layer_size, 1)

        # Good initialization for layers ending in a ReLU like activation:
        for layer in (*conv_layers, *dense_layers):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")  # type: ignore[arg-type]

        # Good initialization for layers ending in a sigmoid or unit activation:
        nn.init.xavier_normal_(final_layer.weight)

        # As described in the paper "Efficient Object Localization Using Convolutional Networks",
        # if adjacent pixels within feature maps are strongly correlated (as is normally the case
        # in early convolution layers) then i.i.d. dropout will not regularize the activations and
        # will otherwise just result in an effective learning rate decrease. -> Using Dropout3D

        # We are not applying dropout to the input layer, because dropout3D would not work as an
        # entire channel would be removed and the network should at least get all channels every time
        ordered_conv_layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i, layer in enumerate(conv_layers):
            ordered_conv_layers[f"conv_layer_{i}"] = layer
            ordered_conv_layers[f"conv_batchnorm_{i}"] = nn.BatchNorm3d(conv_channels)
            ordered_conv_layers[f"conv_silu_{i}"] = nn.SiLU()
            ordered_conv_layers[f"conv_dropout_{i}"] = nn.Dropout3d(p=p_dropout)

            # Inserting AvgPooling after the first layer
            if i == 0 and architecture == "4_avgpool_3":
                ordered_conv_layers[f"conv_avgpool_{0}"] = nn.AvgPool3d(kernel_size=2)
        ordered_conv_layers["flatten"] = nn.Flatten()

        ordered_dense_layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i, layer in enumerate(dense_layers):
            ordered_dense_layers[f"dense_layer_{i}"] = layer
            ordered_dense_layers[f"dense_batchnorm_{i}"] = nn.BatchNorm1d(
                dense_layer_size
            )
            ordered_dense_layers[f"dense_silu_{i}"] = nn.SiLU()
            ordered_dense_layers[f"dense_dropout_{i}"] = nn.Dropout(p=p_dropout)

        ordered_dense_layers["dense_layer_final"] = final_layer
        ordered_dense_layers["final_activation"] = self.final_activation

        self.conv = nn.Sequential(ordered_conv_layers)
        self.dense = nn.Sequential(ordered_dense_layers)

        # All layers is only used to access layers later on
        self.all_layers = ordered_conv_layers
        for name, layer in ordered_dense_layers.items():
            self.all_layers[name] = layer

    def __getitem__(self, layer_name: str) -> nn.Module:
        return self.all_layers[layer_name]

    def __setitem__(self, layer_name: str, module: nn.Module) -> nn.Module:
        self.all_layers[layer_name] = module

    @property
    def layers(self) -> list[str]:
        return list(self.all_layers.keys())

    def forward(self, x: torch.Tensor, x_time: torch.Tensor):

        # Combine height and type dimension to a channel dim with lenght 9
        # And move the channel dim to the second dimension as expected by Conv3D
        x = x.view((*x.shape[:4], 9)).permute(0, 4, 1, 2, 3)

        x = self.conv(x)
        x = torch.cat([x, x_time], dim=1)
        x = self.dense(x)

        return x


class LSTMConv2D(IndexableModule):
    """
    1. Apply the same 2D convolutional network to each of the 11 time steps individually
    2. Apply a bidirectional LSTM layer
    3. Extract the middle timestep of the LSTM output
    4. Pass it through a final dense network
    """

    def __init__(
        self,
        conv_channels: int,
        lstm_size: int,
        n_lstm_layers: int,
        dense_layer_size: int,
        n_dense_layers: int,
        p_dropout: float,
        final_activation: str,
    ):
        assert n_dense_layers >= 1
        super(LSTMConv2D, self).__init__(final_activation)

        self.configuration = {
            "conv_channels": conv_channels,
            "lstm_size": lstm_size,
            "n_lstm_layers": n_lstm_layers,
            "dense_layer_size": dense_layer_size,
            "n_dense_layers": n_dense_layers,
            "p_dropout": p_dropout,
            "final_activation": final_activation,
        }
        self.class_name = "lstm"

        conv_layers = [
            nn.Conv2d(9, conv_channels, kernel_size=4),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3),
        ]
        self.conv_out_size = conv_channels * 3 * 3

        self.lstm_layers = nn.LSTM(
            self.conv_out_size,
            lstm_size,
            n_lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        # Times 3 because of three center values
        # Times 2 because of bidirectional LSTM
        # Add 5 because of time features
        dense_layers = [nn.Linear(3 * 2 * lstm_size + 5, dense_layer_size)]
        for _ in range(n_dense_layers - 1):
            dense_layers.append(nn.Linear(dense_layer_size, dense_layer_size))
        final_layer = nn.Linear(dense_layer_size, 1)

        # Good initialization for layers ending in a ReLU like activation:
        for layer in conv_layers + dense_layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        # Good initialization for layers ending in a sigmoid or unit activation:
        for name, param in self.lstm_layers.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(final_layer.weight)

        ordered_conv_layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i, layer in enumerate(conv_layers):
            ordered_conv_layers[f"conv_layer_{i}"] = layer
            ordered_conv_layers[f"conv_batchnorm_{i}"] = nn.BatchNorm2d(conv_channels)
            ordered_conv_layers[f"conv_silu_{i}"] = nn.SiLU()
            ordered_conv_layers[f"conv_dropout_{i}"] = nn.Dropout2d(p=p_dropout)

            # Inserting AvgPooling after the first layer
            if i == 0:
                ordered_conv_layers[f"conv_avgpool_{0}"] = nn.AvgPool2d(kernel_size=2)
        ordered_conv_layers["flatten"] = nn.Flatten()

        ordered_dense_layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i, layer in enumerate(dense_layers):
            ordered_dense_layers[f"dense_layer_{i}"] = layer
            ordered_dense_layers[f"dense_batchnorm_{i}"] = nn.BatchNorm1d(
                dense_layer_size
            )
            ordered_dense_layers[f"dense_silu_{i}"] = nn.SiLU()
            ordered_dense_layers[f"dense_dropout_{i}"] = nn.Dropout(p=p_dropout)

        ordered_dense_layers["dense_layer_final"] = final_layer
        ordered_dense_layers["final_activation"] = self.final_activation

        self.conv = nn.Sequential(ordered_conv_layers)
        self.dense = nn.Sequential(ordered_dense_layers)

        # All layers is only used to access layers later on
        self.all_layers = ordered_conv_layers
        self.all_layers["lstm"] = self.lstm_layers
        for name, layer in ordered_dense_layers.items():
            self.all_layers[name] = layer

    def __getitem__(self, layer_name: str) -> nn.Module:
        return self.all_layers[layer_name]

    def __setitem__(self, layer_name: str, module: nn.Module) -> nn.Module:
        self.all_layers[layer_name] = module

    @property
    def layers(self) -> list[str]:
        return list(self.all_layers.keys())

    def forward(self, x: torch.Tensor, x_time: torch.Tensor):
        x = x.view((*x.shape[:4], 9)).permute(0, 4, 1, 2, 3)

        # Apply the same 2D conv layer to each of the 11 time steps individually:
        conv_result = torch.zeros(
            x.shape[0],
            11,
            self.conv_out_size,
            dtype=torch.float32,
            requires_grad=False,
            device=x.device,
        )
        for i in range(11):
            conv_result[:, i, :] = self.conv(x[:, :, i, :, :])

        # The Bidirectional LSTM is initialized with random hidden states
        # Meaning it will probably take a few time steps in each direction
        # for the recurrent layers to contain more useful information
        lstm_result, (h_n, c_n) = self.lstm_layers(conv_result)

        # Extracts only the three center values
        center_values = lstm_result[:, 4:7, :].flatten(start_dim=1)
        dense_input = torch.cat([center_values, x_time], dim=1)

        return self.dense(dense_input)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    dense_model = DenseNN(256, 4, 0, "hardsigmoid").to(device)
    conv_model = ConvNet3D(77, 80, 4, 0.5, "4_avgpool_3", "hardsigmoid").to(device)
    lstm_model = LSTMConv2D(
        conv_channels=128,
        lstm_size=256,
        n_lstm_layers=3,
        dense_layer_size=256,
        n_dense_layers=2,
        p_dropout=0.4,
        final_activation="hardsigmoid",
    ).to(device)

    # dense_model.summary(
    #     input_size=[(11, 13, 13, 3, 3), (5,)], batch_size=1024, device=device
    # )
    conv_model.summary(
        input_size=[(11, 13, 13, 3, 3), (5,)], batch_size=1024, device=device
    )
    # lstm_model.summary(
    #     input_size=[(11, 13, 13, 3, 3), (5,)], batch_size=2048, device=device
    # )
