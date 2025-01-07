import torch
from torch import nn

from ..configs import (
    GlobalConfigs,
    GraphNeuralNetworksConfigs,
    MolecularGraphConfigs,
    RegularizationConfigs,
)
from ..utils.graph_utils import map_sender_receiver_feature
from ..utils.nn_utils import get_linear
from ..custom_types import GraphAttentionData, GraphAttentionDataChargeSpin


class BaseGraphNeuralNetworkLayer(nn.Module):
    """
    Base class for Graph Neural Network layers.
    Used in InputLayer and EfficientGraphAttention.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        # Atomic number embeddings
        # ref: escn https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/escn/escn.py#L823
        self.source_atomic_embedding = nn.Embedding(
            molecular_graph_cfg.max_num_elements, gnn_cfg.atom_embedding_size
        )
        self.target_atomic_embedding = nn.Embedding(
            molecular_graph_cfg.max_num_elements, gnn_cfg.atom_embedding_size
        )
        nn.init.uniform_(self.source_atomic_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_atomic_embedding.weight.data, -0.001, 0.001)

        # Node direction embedding
        self.source_direction_embedding = get_linear(
            in_features=gnn_cfg.node_direction_expansion_size,
            out_features=gnn_cfg.node_direction_embedding_size,
            activation=global_cfg.activation,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )
        self.target_direction_embedding = get_linear(
            in_features=gnn_cfg.node_direction_expansion_size,
            out_features=gnn_cfg.node_direction_embedding_size,
            activation=global_cfg.activation,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )

        # Edge distance embedding
        self.edge_distance_embedding = get_linear(
            in_features=gnn_cfg.edge_distance_expansion_size,
            out_features=gnn_cfg.edge_distance_embedding_size,
            activation=global_cfg.activation,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )

    def get_edge_linear(
        self,
        gnn_cfg: GraphNeuralNetworksConfigs,
        global_cfg: GlobalConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        return get_linear(
            in_features=gnn_cfg.edge_distance_embedding_size
            + 2 * gnn_cfg.node_direction_embedding_size
            + 2 * gnn_cfg.atom_embedding_size,
            out_features=global_cfg.hidden_size,
            activation=global_cfg.activation,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )

    def get_node_linear(
        self, global_cfg: GlobalConfigs, reg_cfg: RegularizationConfigs
    ):
        return get_linear(
            in_features=2 * global_cfg.hidden_size,
            out_features=global_cfg.hidden_size,
            activation=global_cfg.activation,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )

    def get_edge_features(self, x: GraphAttentionData) -> torch.Tensor:
        # Atomic number embeddings (ref: escn)
        source_atomic_embedding = self.source_atomic_embedding(x.atomic_numbers)
        target_atomic_embedding = self.target_atomic_embedding(x.atomic_numbers)
        source_atomic_embedding, target_atomic_embedding = map_sender_receiver_feature(
            source_atomic_embedding, target_atomic_embedding, x.neighbor_list
        )

        # Node direction embedding
        source_direction_embedding = self.source_direction_embedding(
            x.node_direction_expansion
        )
        target_direction_embedding = self.target_direction_embedding(
            x.node_direction_expansion
        )
        source_direction_embedding, target_direction_embedding = (
            map_sender_receiver_feature(
                source_direction_embedding, target_direction_embedding, x.neighbor_list
            )
        )

        # Edge distance embedding
        edge_distance_embedding = self.edge_distance_embedding(
            x.edge_distance_expansion
        )

        # Concatenate edge features
        return torch.cat(
            [
                edge_distance_embedding,
                source_direction_embedding,
                source_atomic_embedding,
                target_direction_embedding,
                target_atomic_embedding,
            ],
            dim=-1,
        )

    def get_node_features(
        self, node_features: torch.Tensor, neighbor_list: torch.Tensor
    ) -> torch.Tensor:
        sender_feature, receiver_feature = map_sender_receiver_feature(
            node_features, node_features, neighbor_list
        )
        return torch.cat([sender_feature, receiver_feature], dim=-1)

    def aggregate(self, edge_features, neighbor_mask):
        neighbor_count = neighbor_mask.sum(dim=1, keepdim=True) + 1e-5
        neighbor_count = neighbor_count.to(edge_features.dtype)
        return (edge_features * neighbor_mask.unsqueeze(-1)).sum(dim=1) / neighbor_count

    def forward(self):
        raise NotImplementedError


class BaseGraphNeuralNetworkLayerChargeSpin(BaseGraphNeuralNetworkLayer):
    """
    Base class for Graph Neural Network layers.
    Used in InputLayerChargeSpin and EfficientGraphAttentionChargeSpin.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        # Charge and spin embeddings
        if global_cfg.use_global_charge:
            self.charge_embedding = nn.Embedding(
                molecular_graph_cfg.max_charges, gnn_cfg.global_embedding_size
            )
            nn.init.uniform_(self.charge_embedding.weight.data, -0.001, 0.001)
        else:
            self.charge_embedding = nn.Identity()
        
        if global_cfg.use_global_spin and self.use_global_charge_spin:
            self.spin_embedding = nn.Embedding(
                molecular_graph_cfg.max_spin_multiplicities, gnn_cfg.global_embedding_size
            )
            nn.init.uniform_(self.spin_embedding.weight.data, -0.001, 0.001)
        else:
            self.spin_embedding = nn.Identity()

        # Embeddings for atomic partial charges/spins
        if global_cfg.regress_charges:
            self.source_partial_charge_embedding = get_linear(
                in_features=1,
                out_features=gnn_cfg.atomic_charge_spin_embedding_size,
                activation=global_cfg.activation,
                bias=True,
                dropout=reg_cfg.mlp_dropout
            )
            self.target_partial_charge_embedding = get_linear(
                in_features=1,
                out_features=gnn_cfg.atomic_charge_spin_embedding_size,
                activation=global_cfg.activation,
                bias=True,
                dropout=reg_cfg.mlp_dropout
            )
        else:
            self.source_partial_charge_embedding = nn.Identity()
            self.target_partial_charge_embedding = nn.Identity()

        if global_cfg.regress_spins:
            self.source_partial_spin_embedding = get_linear(
                in_features=1,
                out_features=gnn_cfg.atomic_charge_spin_embedding_size,
                activation=global_cfg.activation,
                bias=True,
                dropout=reg_cfg.mlp_dropout
            )
            self.target_partial_spin_embedding = get_linear(
                in_features=1,
                out_features=gnn_cfg.atomic_charge_spin_embedding_size,
                activation=global_cfg.activation,
                bias=True,
                dropout=reg_cfg.mlp_dropout
            )
        else:
            self.source_partial_spin_embedding = nn.Identity()
            self.target_partial_spin_embedding = nn.Identity()

    def get_edge_linear(
        self,
        gnn_cfg: GraphNeuralNetworksConfigs,
        global_cfg: GlobalConfigs,
        reg_cfg: RegularizationConfigs,
    ):

        in_features = gnn_cfg.edge_distance_embedding_size
            + 2 * gnn_cfg.node_direction_embedding_size
            + 2 * gnn_cfg.atom_embedding_size
            + 2 * gnn_cfg.global_embedding_size
            + 4 * gnn_cfg.atomic_charge_spin_embedding_size

        return get_linear(
            in_features=in_features,
            out_features=global_cfg.hidden_size,
            activation=global_cfg.activation,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )

    def get_edge_features(self,
        x: GraphAttentionDataChargeSpin,
    ) -> torch.Tensor:
        # Atomic number embeddings (ref: escn)
        source_atomic_embedding = self.source_atomic_embedding(x.atomic_numbers)
        target_atomic_embedding = self.target_atomic_embedding(x.atomic_numbers)
        source_atomic_embedding, target_atomic_embedding = map_sender_receiver_feature(
            source_atomic_embedding, target_atomic_embedding, x.neighbor_list
        )

        # Node direction embedding
        source_direction_embedding = self.source_direction_embedding(
            x.node_direction_expansion
        )
        target_direction_embedding = self.target_direction_embedding(
            x.node_direction_expansion
        )
        source_direction_embedding, target_direction_embedding = (
            map_sender_receiver_feature(
                source_direction_embedding, target_direction_embedding, x.neighbor_list
            )
        )

        # Edge distance embedding
        edge_distance_embedding = self.edge_distance_embedding(
            x.edge_distance_expansion
        )

        # Charge and spin embeddings
        # TODO: check dimensions
        charge_embedding = self.charge_embedding(x.charge)
        size_1, size_3 = charge_embedding.shape
        charge_embedding = charge_embedding.repeat(1, max_neighbors).reshape(size_1, max_neighbors, size_3)

        spin_embedding = self.spin_embedding(x.spin_multiplicity)
        size_1, size_3 = spin_embedding.shape
        spin_embedding = spin_embedding.repeat(1, max_neighbors).reshape(size_1, max_neighbors, size_3)

        source_partial_charge_embedding = self.source_partial_charge_embedding(atomic_partial_charges)
        target_partial_charge_embedding = self.target_partial_charge_embedding(atomic_partial_charges)
        source_partial_charge_embedding, target_partial_charge_embedding = (
            map_sender_receiver_feature(
                source_partial_charge_embedding, target_partial_charge_embedding, x.neighbor_list
            )
        )

        source_partial_spin_embedding = self.source_partial_spin_embedding(atomic_partial_spins)
        target_partial_spin_embedding = self.target_partial_spin_embedding(atomic_partial_spins)
        source_partial_spin_embedding, target_partial_spin_embedding = (
            map_sender_receiver_feature(
                source_partial_spin_embedding, target_partial_spin_embedding, x.neighbor_list
            )
        )

        features_to_stack = [
            edge_distance_embedding,
            source_direction_embedding,
            source_atomic_embedding,
            target_direction_embedding,
            target_atomic_embedding,
            charge_embedding,
            spin_embedding,
            source_partial_charge_embedding,
            target_partial_charge_embedding,
            source_partial_spin_embedding,
            target_partial_spin_embedding
        ]

        max_neighbors = x.neighbor_list.shape[1]

        # Concatenate edge features
        return torch.cat(
            features_to_stack,
            dim=-1,
        )
