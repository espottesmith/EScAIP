import dataclasses
import torch


@dataclasses.dataclass
class GraphAttentionData:
    """
    Custom dataclass for storing graph data for Graph Attention Networks

    atomic_numbers: (N)
    edge_distance_expansion: (N, max_nei, edge_distance_expansion_size)
    edge_direction: (N, max_nei, 3)
    node_direction_expansion: (N, node_direction_expansion_size)
    attn_mask: (N * num_head, max_nei, max_nei) Attention mask with angle embeddings
    angle_embedding: (N * num_head, max_nei, max_nei) Angle embeddings (cosine)
    neighbor_list: (N, max_nei)
    neighbor_mask: (N, max_nei)
    node_batch: (N)
    node_padding_mask: (N)
    graph_padding_mask: (num_graphs)
    """

    atomic_numbers: torch.Tensor
    edge_distance_expansion: torch.Tensor
    edge_direction: torch.Tensor
    node_direction_expansion: torch.Tensor
    attn_mask: torch.Tensor
    angle_embedding: torch.Tensor
    neighbor_list: torch.Tensor
    neighbor_mask: torch.Tensor
    node_batch: torch.Tensor
    node_padding_mask: torch.Tensor
    graph_padding_mask: torch.Tensor


def flatten_graph_attention_data_with_spec(data, spec):
    # Flatten based on the in_spec structure
    flat_data = []
    for field_name in spec.context[0]:
        field_value = getattr(data, field_name)
        if isinstance(field_value, torch.Tensor):
            flat_data.append(field_value)
        elif field_value is None:
            flat_data.append(None)
        else:
            # Handle custom types like AttentionBias
            flat_data.extend(field_value.tree_flatten())
    return tuple(flat_data)


# TODO: alter instantiation
# Charge/spin info needs to be optional; defaults zeros? Defaults False?
@dataclasses.dataclass
class GraphAttentionDataChargeSpin:
    """
    Custom dataclass for storing graph data for Graph Attention Networks
    TODO: make sure to repeat charge and spin appropriately
        (current scheme, for simplicity: repeat global charge m times,
        where m is the number of atoms in molecule M)
    charge: (N)
    spin_multiplicity: (N)
    atomic_numbers: (N)
    atomic_partial_charges: (N)
    atomic_partial_spins: (N)
    edge_distance_expansion: (N, max_nei, edge_distance_expansion_size)
    edge_direction: (N, max_nei, 3)
    node_direction_expansion: (N, node_direction_expansion_size)
    neighbor_list: (N, max_nei)
    neighbor_mask: (N, max_nei)
    node_batch: (N)
    node_padding_mask: (N)
    graph_padding_mask: (num_graphs)
    attn_bias: AttentionBias for xformers kernel
    """

    charge: torch.Tensor
    spin_multiplicity: torch.Tensor
    atomic_numbers: torch.Tensor
    atomic_partial_charges: torch.Tensor
    atomic_partial_spins: torch.Tensor
    edge_distance_expansion: torch.Tensor
    edge_direction: torch.Tensor
    node_direction_expansion: torch.Tensor
    neighbor_list: torch.Tensor
    neighbor_mask: torch.Tensor
    node_batch: torch.Tensor
    node_padding_mask: torch.Tensor
    graph_padding_mask: torch.Tensor
    attn_bias: Union[AttentionBias, None]


torch.export.register_dataclass(
    GraphAttentionData, serialized_type_name="GraphAttentionData"
)
torch.fx._pytree.register_pytree_flatten_spec(
    GraphAttentionData, flatten_fn_spec=flatten_graph_attention_data_with_spec
)

torch.export.register_dataclass(
    GraphAttentionDataChargeSpin, serialized_type_name="GraphAttentionDataChargeSpin"
)
torch.fx._pytree.register_pytree_flatten_spec(
    GraphAttentionDataChargeSpin, flatten_fn_spec=flatten_graph_attention_data_with_spec
)