import __fix_relative_imports  # noqa: F401
import torch.nn as nn
import torch
import numpy as np
from mscEidalVesetrud.deep_learning.neural_nets import IndexableModule
from collections import OrderedDict


class SimpleNN(IndexableModule):
    def __init__(self):
        super(SimpleNN, self).__init__()

        self.dense_layers = {
            "fc11": nn.Linear(10, 10),
            "fc12": nn.Linear(2, 2),
            "fc2": nn.Linear(12, 1),
        }

        for layer in self.dense_layers.values():
            nn.init.xavier_normal_(layer.weight)

        self.fc11 = self.dense_layers["fc11"]
        self.fc12 = self.dense_layers["fc12"]
        self.fc2 = self.dense_layers["fc2"]
        self.sigmoid = nn.Sigmoid()

    def __getitem__(self, layer_name: str) -> nn.Module:
        return self.dense_layers[layer_name]

    def __setitem__(self, layer_name: str, module: nn.Module) -> nn.Module:
        self.dense_layers[layer_name] = module

    def forward(self, x: torch.Tensor, x_time: torch.Tensor):
        assert x.shape[1:] == (10,)
        assert x_time.shape[1:] == (2,)

        x = self.fc11(x)
        x = self.sigmoid(x)

        x_time = self.fc12(x_time)
        x_time = self.sigmoid(x_time)

        x = torch.cat([x, x_time], dim=1)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class ShapeRecognizer3000(IndexableModule):
    def __init__(
        self,
        hidden_layer_sizes: list[int] = [2],
        activation: str = "relu",
        preset_weights=False,
    ):
        super(ShapeRecognizer3000, self).__init__(final_activation="identity")

        assert activation in ["relu", "silu"]

        if preset_weights:
            # Force ReLU activtion
            activation = "relu"

            layers = [nn.Linear(2, 3), nn.Linear(3, 1)]
            layers[0].weight = nn.Parameter(
                torch.tensor(
                    [
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [-1.0, 1.0],
                    ]
                ),
                requires_grad=False,
            )
            layers[0].bias = nn.Parameter(
                torch.tensor([1.0, 0.0, 0.0]), requires_grad=False
            )
            layers[1].weight = nn.Parameter(
                torch.tensor([[1.0, -1.0, -1.0]]), requires_grad=False
            )
            layers[1].bias = nn.Parameter(torch.tensor([0.0]), requires_grad=False)

        else:
            if len(hidden_layer_sizes) == 0:
                layers = [nn.Linear(2, 1)]
            else:
                layers = [nn.Linear(2, hidden_layer_sizes[0])]
                for i in range(1, len(hidden_layer_sizes)):
                    layers.append(
                        nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i])
                    )
                layers.append(nn.Linear(hidden_layer_sizes[-1], 1))

            for layer in layers:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        self.ordered_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.ordered_layers["input"] = nn.Identity()
        for i, layer in enumerate(layers):
            self.ordered_layers[f"linear_{i}"] = layer
            if activation == "relu":
                self.ordered_layers[f"relu_{i}"] = nn.ReLU()
            elif activation == "silu":
                self.ordered_layers[f"silu_{i}"] = nn.SiLU()
            else:
                raise ValueError("Unexpected activation")

        self.net = nn.Sequential(self.ordered_layers)

        if preset_weights:
            self.linear_0 = self["linear_0"]
            self.relu_0 = self["relu_0"]
            self.linear_1 = self["linear_1"]
            self.relu_1 = self["relu_1"]

    def __getitem__(self, layer_name: str) -> nn.Module:
        return self.ordered_layers[layer_name]

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 2
        return self.net(x)

    def print_net(self):
        for layer_id, layer in self.ordered_layers.items():
            if not isinstance(layer, nn.Linear):
                continue

            print(layer_id)
            print(layer.weight)
            print(layer.bias)
            print()

    def get_activation(self, layer_name: str) -> np.ndarray:
        return self.activations[layer_name].detach().cpu().numpy()
