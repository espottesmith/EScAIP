from torch import nn

from ..utils.nn_utils import get_feedforward, get_normalization_layer

from ..configs import (
    GlobalConfigs,
    GraphNeuralNetworksConfigs,
    RegularizationConfigs,
)


class ReadoutBlock(nn.Module):
    """
    Readout from each graph attention block for energy and force output
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        self.regress_forces = global_cfg.regress_forces
        self.direct_force = global_cfg.direct_force

        # energy read out
        self.energy_ffn = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=global_cfg.activation,
            hidden_layer_multiplier=gnn_cfg.readout_hidden_layer_multiplier,
            dropout=reg_cfg.mlp_dropout,
            bias=True,
        )
        self.energy_norm = get_normalization_layer(
            reg_cfg.normalization, is_graph=False
        )(global_cfg.hidden_size)

        # forces read out
        if self.regress_forces and self.direct_force:
            self.force_ffn = get_feedforward(
                hidden_dim=global_cfg.hidden_size,
                activation=global_cfg.activation,
                hidden_layer_multiplier=gnn_cfg.readout_hidden_layer_multiplier,
                dropout=reg_cfg.mlp_dropout,
                bias=True,
            )
            self.force_norm = get_normalization_layer(
                reg_cfg.normalization, is_graph=False
            )(global_cfg.hidden_size)
        else:
            self.force_ffn = nn.Identity()
            self.force_norm = nn.Identity()

        if global_cfg.use_fp16_backbone:
            self.energy_ffn = self.energy_ffn.half()
            self.energy_norm = self.energy_norm.half()
            self.force_ffn = self.force_ffn.half()
            self.force_norm = self.force_norm.half()

    def forward(self, node_features, edge_features):
        """
        Output: Node Readout (N, H); Edge Readout (N, max_nei, H)
        """
        energy_readout = node_features + self.energy_ffn(
            self.energy_norm(node_features)
        )
        force_readout = edge_features + self.force_ffn(self.force_norm(edge_features))

        return energy_readout, force_readout


class ChargeSpinReadoutBlock(nn.Module):
    """
    Readout from each graph attention block for partial charge and spin output
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        
        # node read out
        self.node_ffn = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=global_cfg.activation,
            hidden_layer_multiplier=gnn_cfg.readout_hidden_layer_multiplier,
            dropout=reg_cfg.mlp_dropout,
            bias=True,
        )
        self.node_norm = get_normalization_layer(
            reg_cfg.normalization, is_graph=False
        )(global_cfg.hidden_size)

        # edges read out
        self.edge_ffn = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=global_cfg.activation,
            hidden_layer_multiplier=gnn_cfg.readout_hidden_layer_multiplier,
            dropout=reg_cfg.mlp_dropout,
            bias=True,
        )
        self.edge_norm = get_normalization_layer(
            reg_cfg.normalization, is_graph=False
        )(global_cfg.hidden_size)

        if global_cfg.use_fp16_backbone:
            self.node_ffn = self.node_ffn.half()
            self.node_norm = self.node_norm.half()
            self.edge_ffn = self.edge_ffn.half()
            self.edge_norm = self.edge_norm.half()

    def forward(self, node_features, edge_features):
        """
        Output: Node Readout (N, H); Edge Readout (N, max_nei, H)
        """
        node_readout = node_features + self.node_ffn(
            self.node_norm(node_features)
        )
        edge_readout = edge_features + self.edge_ffn(self.edge_norm(edge_features))

        return node_readout, edge_readout

