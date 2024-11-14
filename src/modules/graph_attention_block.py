from functools import partial

import torch
from torch import nn

from xformers.ops import memory_efficient_attention

from ..configs import (
    GlobalConfigs,
    GraphNeuralNetworksConfigs,
    MolecularGraphConfigs,
    RegularizationConfigs,
)
from ..custom_types import GraphAttentionData
from ..utils.stochastic_depth import StochasticDepth, SkipStochasticDepth
from ..utils.nn_utils import (
    NormalizationType,
    get_normalization_layer,
    get_linear,
    get_feedforward,
)
from .base_block import BaseGraphNeuralNetworkLayer


class EfficientGraphAttention(BaseGraphNeuralNetworkLayer):
    """
    Efficient Graph Attention module.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
        use_global_charge_spin: bool = False,
    ):
        super().__init__(
            global_cfg,
            molecular_graph_cfg,
            gnn_cfg,
            reg_cfg,
            use_global_charge_spin=use_global_charge_spin
        )

        # Edge linear layer
        self.edge_linear = self.get_edge_linear(gnn_cfg, global_cfg, reg_cfg)

        # Node hidden layer
        self.node_linear = self.get_node_linear(global_cfg, reg_cfg)

        # message linear
        self.message_linear = get_linear(
            in_features=global_cfg.hidden_size * 2,
            out_features=global_cfg.hidden_size,
            activation=global_cfg.activation,
            bias=True,
        )

        # datatype
        if global_cfg.use_fp16_backbone:
            self.backbone_dtype = torch.float16
            self.node_linear = self.node_linear.half()
            self.message_linear = self.message_linear.half()
        else:
            self.backbone_dtype = torch.float32

        # Multi-head attention
        self.is_xformers = gnn_cfg.atten_name == "xformers"
        if self.is_xformers:
            self.in_proj = nn.Linear(global_cfg.hidden_size, 3 * global_cfg.hidden_size)
            self.num_heads = gnn_cfg.atten_num_heads
            if global_cfg.hidden_size % self.num_heads != 0:
                raise ValueError(
                    f"Hidden size {global_cfg.hidden_size} is not divisible by the number of heads {self.num_heads}"
                )
            self.head_dim = global_cfg.hidden_size // self.num_heads
            self.out_proj = nn.Linear(global_cfg.hidden_size, global_cfg.hidden_size)
            self.multi_head_attention = partial(
                memory_efficient_attention, p=reg_cfg.atten_dropout
            )
            self.compute_attention = self.compute_attention_xformers
            if global_cfg.use_fp16_backbone:
                self.in_proj = self.in_proj.half()
                self.out_proj = self.out_proj.half()
        else:
            self.multi_head_attention = nn.MultiheadAttention(
                embed_dim=global_cfg.hidden_size,
                num_heads=gnn_cfg.atten_num_heads,
                dropout=reg_cfg.atten_dropout,
                bias=True,
                batch_first=True,
                dtype=self.backbone_dtype,
            )
            self.compute_attention = self.compute_attention_pytorch

    def compute_attention_pytorch(
        self, message: torch.Tensor, data: GraphAttentionData
    ):
        output = self.multi_head_attention(
            query=message,
            key=message,
            value=message,
            key_padding_mask=~data.neighbor_mask,
            need_weights=False,
        )
        return output[0]

    def compute_attention_xformers(
        self, message: torch.Tensor, data: GraphAttentionData
    ):
        attn_bias = data.attn_bias
        num_nodes, num_neighbors, _ = message.shape
        # Get query, key, value
        # ref: swin transformer https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py#L179-L181
        qkv = self.in_proj(message)
        # merge to a long sequence for block diagonal bias
        qkv = qkv.reshape(num_nodes * num_neighbors, qkv.shape[-1])
        # split heads and qkv
        qkv = qkv.reshape(num_nodes * num_neighbors, 3, self.num_heads, self.head_dim)
        # add batch dimension
        qkv = qkv.unsqueeze(0).permute(2, 0, 1, 3, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        # shape: (1, num_nodes * num_neighbors, num_heads, head_dim)

        # Compute attention
        output = self.multi_head_attention(
            query=query,
            key=key,
            value=value,
            attn_bias=attn_bias,
        )

        # Reshape output
        output = output.squeeze(0).reshape(num_nodes, num_neighbors, -1)

        return self.out_proj(output)

    def forward(
        self,
        data: GraphAttentionData,
        node_features: torch.Tensor,
        edge_features: torch.Tensor
    ):
        # Get edge features
        edge_features = self.get_edge_features(data)
        edge_hidden = self.edge_linear(edge_features)

        # Get node features
        node_features = self.get_node_features(node_features, data.neighbor_list)
        node_hidden = self.node_linear(node_features)

        # Concatenate edge and node features (num_nodes, num_neighbors, hidden_size)
        message = self.message_linear(
            torch.cat([edge_hidden.to(self.backbone_dtype), node_hidden], dim=-1)
        )

        # Multi-head self-attention
        edge_output = self.compute_attention(message, data)

        # Aggregation
        node_output = self.aggregate(edge_output, data.neighbor_mask)

        return node_output, edge_output


class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network module. Wrapper around xformers' MLP module.
    Ref: https://github.com/facebookresearch/xformers/tree/main/xformers/components/feedforward
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()
        self.mlp_node = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=global_cfg.activation,
            hidden_layer_multiplier=gnn_cfg.ffn_hidden_layer_multiplier,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )
        self.mlp_edge = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=global_cfg.activation,
            hidden_layer_multiplier=gnn_cfg.ffn_hidden_layer_multiplier,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )
        if global_cfg.use_fp16_backbone:
            self.mlp_node = self.mlp_node.half()
            self.mlp_edge = self.mlp_edge.half()

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor):
        return self.mlp_node(node_features), self.mlp_edge(edge_features)


class EfficientGraphAttentionBlock(nn.Module):
    """
    Efficient Graph Attention Block module.
    Modified from https://github.com/facebookresearch/xformers/blob/main/xformers/factory/block_factory.py#L96
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
        use_global_charge_spin: bool = False,
    ):
        super().__init__()

        # Graph attention
        self.graph_attention = EfficientGraphAttention(
            global_cfg=global_cfg,
            molecular_graph_cfg=molecular_graph_cfg,
            gnn_cfg=gnn_cfg,
            reg_cfg=reg_cfg,
            use_global_charge_spin=use_global_charge_spin
        )

        # Feed forward network
        self.feedforward = FeedForwardNetwork(
            global_cfg=global_cfg,
            gnn_cfg=gnn_cfg,
            reg_cfg=reg_cfg,
        )

        # Normalization
        normalization = NormalizationType(reg_cfg.normalization)
        self.norm_attn = get_normalization_layer(normalization, is_graph=False)(
            global_cfg.hidden_size
        )
        self.norm_ffn = get_normalization_layer(normalization)(global_cfg.hidden_size)
        if global_cfg.use_fp16_backbone:
            self.norm_attn = self.norm_attn.half()
            self.norm_ffn = self.norm_ffn.half()

        # Stochastic depth
        self.stochastic_depth = (
            StochasticDepth(reg_cfg.stochastic_depth_prob)
            if reg_cfg.stochastic_depth_prob > 0.0
            else SkipStochasticDepth()
        )

    def forward(
        self,
        data: GraphAttentionData,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
    ):
        # ref: swin transformer https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py#L452
        # x = x + self.stochastic_depth(self.graph_attention(self.norm_attn(x)))
        # x = x + self.stochastic_depth(self.feedforward(self.norm_ffn(x)))

        # attention
        node_hidden = self.norm_attn(node_features)
        node_hidden, edge_hidden = self.graph_attention(data, node_hidden)
        node_hidden, edge_hidden = self.stochastic_depth(
            node_hidden, edge_hidden, data.node_batch
        )
        node_features, edge_features = (
            node_hidden + node_features,
            edge_hidden + edge_features,
        )

        # feedforward
        node_hidden, edge_hidden = self.norm_ffn(node_features, edge_features)
        node_hidden, edge_hidden = self.feedforward(node_hidden, edge_hidden)
        node_hidden, edge_hidden = self.stochastic_depth(
            node_hidden, edge_hidden, data.node_batch
        )
        node_features, edge_features = (
            node_hidden + node_features,
            edge_hidden + edge_features,
        )
        return node_features, edge_features
