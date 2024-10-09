# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from backbone import xavier, num_flat_features
from backbone.utils.k_winners import KWinners
import numpy as np
from backbone import MammothBackbone, num_flat_features, xavier


class MNISTMLP(MammothBackbone):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Instantiates the layers of the network.

        Args:
            input_size: the size of the input data
            output_size: the size of the output
        """
        super(MNISTMLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, 100)
        self.fc2 = nn.Linear(100, 100)

        self._features = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
        )
        self.classifier = nn.Linear(100, self.output_size)
        self.net = nn.Sequential(self._features, self.classifier)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.net.apply(xavier)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, input_size)

        Returns:
            output tensor (output_size)
        """
        x = x.view(-1, num_flat_features(x))

        feats = self._features(x)

        if returnt == 'features':
            return feats

        out = self.classifier(feats)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feats)

        raise NotImplementedError("Unknown return type")

    def to(self, device):
        super().to(device)
        self.device = device
        return self
class SparseMNISTMLP(nn.Module):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """

    def __init__(self, input_size: int, output_size: int, kw_percent_on: float=0.99) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(SparseMNISTMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kw_percent_on = kw_percent_on

        self.apply_heterogeneous_dropout = False
        self._layers = nn.ModuleList()
        self._activations = nn.ModuleList()
        self._activation_counts = {}
        self._keep_probs = {}

        # Init activation counts
        self._activation_counts[f'layer_1'] = torch.zeros(100)
        self._activation_counts[f'layer_2'] = torch.zeros(100)
        self._activation_counts[f'layer_1_classwise'] = torch.zeros(self.output_size, 100)
        self._activation_counts[f'layer_2_classwise'] = torch.zeros(self.output_size, 100)

        fc1 = nn.Linear(self.input_size, 100)
        fc2 = nn.Linear(100, 100)

        kw1 = KWinners(
            n=100, percent_on=kw_percent_on[0],
            k_inference_factor=1.0,
            boost_strength=0.0,
            boost_strength_factor=0.0
        )

        kw2 = KWinners(
            n=100, percent_on=kw_percent_on[1],
            k_inference_factor=1.0,
            boost_strength=0.0,
            boost_strength_factor=0.0
        )

        self._layers = nn.ModuleList([fc1, fc2])
        self._activations = nn.ModuleList([kw1, kw2])
        self.classifier = nn.Linear(100, self.output_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self._layers.apply(xavier)
        self.classifier.apply(xavier)

    def forward(self, x, y=None, disable_dropout=False, return_activations=False, mode='train', update_act_counts=True) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        activations = {}
        x = x.view(-1, num_flat_features(x))
        for i, (layer, activation) in enumerate(zip(self._layers, self._activations)):
            orig_act = layer(x)

            # Apply Dropout
            if self.apply_heterogeneous_dropout and not disable_dropout and self.layerwise_dropout[i]:
                if self.training and self._keep_probs:

                    if self.disable_heterogeneous_mode:
                        heterogeneous_mask = (torch.ones(orig_act.shape[1]) > 0).repeat(len(y), 1)
                    else:
                        prob = self._keep_probs[f'layer_{i + 1}']
                        prob = np.asarray(prob).astype('float64')
                        prob = prob / np.sum(prob)
                        heterogeneous_mask = torch.zeros(orig_act.shape[1])
                        try:
                            idx = np.random.choice(
                                orig_act.shape[1],
                                int(self.kw_percent_on[i] * 1.1 * orig_act.shape[1]),
                                p=prob,
                                replace=False
                            )
                            heterogeneous_mask[idx] = 1
                            heterogeneous_mask = (heterogeneous_mask > 0).repeat(len(y), 1)
                        except:
                            print('Fix it!')

                    classwise_mask = torch.rand(orig_act.shape[1]) < self._keep_probs[f'layer_{i + 1}_classwise'][y]

                    # If classwise probabilities available, use them instead use heterogeneous dropout
                    layer_mask = torch.where(
                        (self._keep_probs[f'layer_{i + 1}_classwise'].sum(1)[y] > 0)[:, None].repeat(1, heterogeneous_mask.shape[1]),
                        classwise_mask,
                        heterogeneous_mask
                    )
                    layer_mask = layer_mask.to(x.device)
                    orig_act *= layer_mask

            x = activation(orig_act)

            if self.training and update_act_counts:
                self._activation_counts[f'layer_{i + 1}'] += (x > 0).sum(dim=0).cpu()
                if y is not None:
                    for class_idx in range(self.output_size):
                        sel_idx = y == class_idx
                        self._activation_counts[f'layer_{i + 1}_classwise'][class_idx] += (x[sel_idx] > 0).sum(dim=0).cpu()

            activations[f'layer_{i}'] = x

        out = self.classifier(x)

        if return_activations:
            return out, activations
        else:
            return out

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (input_size * 100 + 100 + 100 * 100 + 100 +
                                    + 100 * output_size + output_size)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_sparsity(self, kw_percent_on, device='cuda'):
        print(f'Resetting Activation Sparsity to {kw_percent_on}')
        self.kw_percent_on = kw_percent_on
        kw1 = KWinners(
            n=100, percent_on=kw_percent_on,
            k_inference_factor=1.0,
            boost_strength=0.0,
            boost_strength_factor=0.0
        )

        kw2 = KWinners(
            n=100, percent_on=kw_percent_on,
            k_inference_factor=1.0,
            boost_strength=0.0,
            boost_strength_factor=0.0
        )

        self._activations = nn.ModuleList([kw1, kw2])
        self._activations = self._activations.to(device)
