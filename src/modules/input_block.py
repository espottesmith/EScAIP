import torch
import torch.nn as nn

from ..configs import (
    GlobalConfigs,
    GraphNeuralNetworksConfigs,
    MolecularGraphConfigs,
    RegularizationConfigs,
)
from ..custom_types import GraphAttentionData
from .base_block import BaseGraphNeuralNetworkLayer
from ..utils.nn_utils import get_feedforward, get_linear, get_normalization_layer


class InputLayer(BaseGraphNeuralNetworkLayer):
    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
        use_global_charge_spin: bool = False,
        use_atomic_charge_spin: bool = False
    ):
        super().__init__(
            global_cfg,
            molecular_graph_cfg,
            gnn_cfg,
            reg_cfg,
            use_global_charge_spin=use_global_charge_spin,
            use_atomic_charge_spin=use_atomic_charge_spin
        )

        # Edge linear layer
        self.edge_linear = self.get_edge_linear(gnn_cfg, global_cfg, reg_cfg)

        # ffn for edge features
        self.edge_ffn = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=global_cfg.activation,
            hidden_layer_multiplier=1,
            dropout=reg_cfg.mlp_dropout,
            bias=True,
        )

        # normalization
        self.norm = get_normalization_layer(reg_cfg.normalization, is_graph=False)(
            global_cfg.hidden_size
        )

        # datatype
        if global_cfg.use_fp16_backbone:
            self.backbone_dtype = torch.float16
            self.edge_ffn = self.edge_ffn.half()
            self.norm = self.norm.half()
        else:
            self.backbone_dtype = torch.float32

    def forward(self,
                inputs: GraphAttentionData,
                atomic_partial_charges: nn.Tensor | None = None,
                atomic_partial_spins:  nn.Tensor | None = None
            ):
        # Get edge features
        edge_features = self.get_edge_features(
            inputs,
            atomic_partial_charges=atomic_partial_charges,
            atomic_partial_spins=atomic_partial_spins
        )

        # Edge processing
        edge_hidden = self.edge_linear(edge_features)
        # Convert data type
        edge_hidden = edge_hidden.to(self.backbone_dtype)
        edge_output = edge_hidden + self.edge_ffn(self.norm(edge_hidden))

        # Aggregation
        node_output = self.aggregate(edge_output, inputs.neighbor_mask)

        # Update inputs
        return node_output, edge_output


class InputBlock(nn.Module):
    """
    Wrapper of InputLayer for adding normalization
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
        use_global_charge_spin: bool = False,
        use_atomic_charge_spin: bool = False
    ):
        super().__init__()

        self.input_layer = InputLayer(
            global_cfg,
            molecular_graph_cfg,
            gnn_cfg,
            reg_cfg,
            use_global_charge_spin=use_global_charge_spin,
            use_atomic_charge_spin=use_atomic_charge_spin
        )

        self.norm = get_normalization_layer(reg_cfg.normalization)(
            global_cfg.hidden_size
        )

        if global_cfg.use_fp16_backbone:
            self.norm = self.norm.half()

    def forward(self, inputs: GraphAttentionData, atomic_partial_charges: nn.Tensor | None = None, atomic_partial_spins:  nn.Tensor | None = None):
        node_features, edge_features = self.input_layer(inputs, atomic_partial_charges, atomic_partial_spins)
        return self.norm(node_features, edge_features)
