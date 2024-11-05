from typing import Literal

import torch
import torch.nn as nn

from ..utils.nn_utils import get_linear, get_feedforward, get_normalization_layer

from ..configs import (
    GlobalConfigs,
    MolecularGraphConfigs,
    GraphNeuralNetworksConfigs,
    RegularizationConfigs,
)

#TODO: magmoms, partial_charges, partial_spins
class OutputLayer(nn.Module):
    """
    Get the final prediction from the readouts (force or energy)
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
        output_type: Literal["Energy", "ForceDirection", "ForceMagnitude"],
    ):
        super().__init__()

        self.output_type = output_type
        output_type_dict = {
            "Energy": 1,
            "ForceDirection": 3,
            "ForceMagnitude": 1,
        }
        assert (
            output_type in output_type_dict.keys()
        ), f"Invalid output type {output_type}"

        # mlp
        self.ffn = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=global_cfg.activation,
            hidden_layer_multiplier=gnn_cfg.output_hidden_layer_multiplier,
            dropout=reg_cfg.mlp_dropout,
            bias=True,
        )

        # normalization
        self.pre_norm = get_normalization_layer(reg_cfg.normalization, is_graph=False)(
            global_cfg.hidden_size
        )
        self.post_norm = get_normalization_layer(reg_cfg.normalization, is_graph=False)(
            global_cfg.hidden_size
        )

        # final output layer
        self.final_output = get_linear(
            in_features=global_cfg.hidden_size,
            out_features=output_type_dict[output_type],
            activation=None,
        )

    def forward(self, readouts: torch.Tensor) -> torch.Tensor:
        """
        readouts: concated readout features
        Shape ([num_nodes, hidden_size] or [num_nodes, max_neighbor, hidden_size])
        """
        # mlp
        readouts = readouts + self.ffn(self.pre_norm(readouts))

        # final output layer
        return self.final_output(self.post_norm(readouts))


class OutputBlock(nn.Module):
    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        self.regress_forces = global_cfg.regress_forces
        self.direct_force = global_cfg.direct_force
        self.register_buffer(
            "avg_num_nodes", torch.tensor(molecular_graph_cfg.avg_num_nodes)
        )

        # map concatenated readout features to hidden size
        self.node_input_projection = get_linear(
            in_features=global_cfg.hidden_size * (gnn_cfg.num_layers + 1),
            out_features=global_cfg.hidden_size,
            activation=global_cfg.activation,
            bias=True,
        )

        # energy output layer
        self.energy_layer = OutputLayer(
            global_cfg=global_cfg,
            gnn_cfg=gnn_cfg,
            reg_cfg=reg_cfg,
            output_type="Energy",
        )

        # force output layer
        if self.regress_forces and self.direct_force:
            # map concatenated readout features to hidden size
            self.edge_input_projection = get_linear(
                in_features=global_cfg.hidden_size * (gnn_cfg.num_layers + 1),
                out_features=global_cfg.hidden_size,
                activation=global_cfg.activation,
                bias=True,
            )
            self.force_direction_layer = OutputLayer(
                global_cfg=global_cfg,
                gnn_cfg=gnn_cfg,
                reg_cfg=reg_cfg,
                output_type="ForceDirection",
            )
            self.force_magnitude_layer = OutputLayer(
                global_cfg=global_cfg,
                gnn_cfg=gnn_cfg,
                reg_cfg=reg_cfg,
                output_type="ForceMagnitude",
            )
            self.force_output_fn = self.get_force_output
        else:
            self.force_output_fn = lambda *_, **__: None

    def get_force_output(
        self, node_readouts, edge_readouts, edge_direction, neighbor_mask
    ):
        edge_readouts = self.edge_input_projection(edge_readouts)
        # get force direction from edge readouts
        force_direction = self.force_direction_layer(
            edge_readouts
        )  # (num_nodes, max_neighbor, 3)
        force_direction = (
            force_direction * edge_direction
        )  # (num_nodes, max_neighbor, 3)
        force_direction = (force_direction * neighbor_mask.unsqueeze(-1)).sum(
            dim=1
        )  # (num_nodes, 3)
        # get force magnitude from node readouts
        force_magnitude = self.force_magnitude_layer(node_readouts)  # (num_nodes, 1)
        # get output force
        force_output = force_direction * force_magnitude  # (num_nodes, 3)
        return force_output

    def forward(
        self,
        node_readouts,
        edge_readouts,
        edge_direction,
        node_batch,
        neighbor_mask,
        num_graphs,
    ):
        # map to float32
        node_readouts = node_readouts.to(torch.float32)
        edge_readouts = edge_readouts.to(torch.float32)

        # get energy from node readouts
        node_readouts = self.node_input_projection(node_readouts)
        energy_output = self.energy_layer(node_readouts)

        # the following not compatible with torch.compile
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")

        # implement with torch.Tensor.scatter_reduce_ instead
        output = torch.zeros(
            (num_graphs, 1), device=energy_output.device, dtype=energy_output.dtype
        )
        energy_output = (
            output.scatter_reduce(
                dim=0,
                index=node_batch.reshape(-1, 1),
                src=energy_output,
                reduce="sum",
                include_self=False,
            )
            / self.avg_num_nodes
        )

        # get force from edge readouts and node readouts
        force_output = self.force_output_fn(
            node_readouts, edge_readouts, edge_direction, neighbor_mask
        )

        return energy_output, force_output
