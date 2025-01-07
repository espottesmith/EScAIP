from functools import partial

import torch
import torch.nn as nn
import torch_geometric

from e3nn import o3

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import GraphModelMixin, HeadInterface

from .configs import EScAIPConfigs, init_configs
from .custom_types import GraphAttentionData, GraphAttentionDataChargeSpin
from .modules import (
    EfficientGraphAttentionBlock,
    EfficientGraphAttentionBlockChargeSpin,
    InputBlock,
    InputBlockChargeSpin,
    ReadoutBlock,
    ChargeSpinReadoutBlock,
    OutputProjection,
    OutputLayer,
)
from .utils.data_preprocess import data_preprocess
from .utils.nn_utils import no_weight_decay, init_linear_weights
from .utils.graph_utils import unpad_results, compilable_scatter


@registry.register_model("EScAIP_backbone")
class EScAIPBackbone(nn.Module, GraphModelMixin):
    """
    Efficiently Scaled Attention Interactomic Potential (EScAIP) backbone model.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        # load configs
        cfg = init_configs(EScAIPConfigs, kwargs)
        self.global_cfg = cfg.global_cfg
        self.molecular_graph_cfg = cfg.molecular_graph_cfg
        self.gnn_cfg = cfg.gnn_cfg
        self.reg_cfg = cfg.reg_cfg

        # for trainer
        self.regress_forces = cfg.global_cfg.regress_forces
        self.use_pbc = cfg.molecular_graph_cfg.use_pbc

        # graph generation
        self.use_pbc_single = (
            self.molecular_graph_cfg.use_pbc_single
        )  # TODO: remove this when FairChem fixes the bug
        generate_graph_fn = partial(
            self.generate_graph,
            cutoff=self.molecular_graph_cfg.max_radius,
            max_neighbors=self.molecular_graph_cfg.max_neighbors,
            use_pbc=self.molecular_graph_cfg.use_pbc,
            otf_graph=self.molecular_graph_cfg.otf_graph,
            enforce_max_neighbors_strictly=self.molecular_graph_cfg.enforce_max_neighbors_strictly,
            use_pbc_single=self.molecular_graph_cfg.use_pbc_single,
        )

        # data preprocess
        self.data_preprocess = partial(
            data_preprocess,
            generate_graph_fn=generate_graph_fn,
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
        )

        ## Model Components

        # Input Block
        self.input_block = InputBlock(
            global_cfg=self.global_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
        )

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                EfficientGraphAttentionBlock(
                    global_cfg=self.global_cfg,
                    molecular_graph_cfg=self.molecular_graph_cfg,
                    gnn_cfg=self.gnn_cfg,
                    reg_cfg=self.reg_cfg,
                )
                for _ in range(self.gnn_cfg.num_layers)
            ]
        )

        # Readout Layer
        self.readout_layers = nn.ModuleList(
            [
                ReadoutBlock(
                    global_cfg=self.global_cfg,
                    gnn_cfg=self.gnn_cfg,
                    reg_cfg=self.reg_cfg,
                )
                for _ in range(self.gnn_cfg.num_layers + 1)
            ]
        )

        # Output Projection
        self.output_projection = OutputProjection(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
        )

        # init weights
        self.apply(init_linear_weights)

        # enable torch.set_float32_matmul_precision('high') if not using fp16 backbone
        if not self.global_cfg.use_fp16_backbone:
            torch.set_float32_matmul_precision("high")
        torch._logging.set_logs(recompiles=True)

        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )


    def compiled_forward(self, data: GraphAttentionData):
        # input block
        node_features, edge_features = self.input_block(data)

        # input readout
        readouts = self.readout_layers[0](node_features, edge_features)
        node_readouts = [readouts[0]]
        edge_readouts = [readouts[1]]

        # transformer blocks
        for idx in range(self.gnn_cfg.num_layers):
            node_features, edge_features = self.transformer_blocks[idx](
                data, node_features, edge_features
            )
            readouts = self.readout_layers[idx + 1](node_features, edge_features)
            node_readouts.append(readouts[0])
            edge_readouts.append(readouts[1])

        node_features, edge_features = self.output_projection(
            node_readouts=torch.cat(node_readouts, dim=-1),
            edge_readouts=torch.cat(edge_readouts, dim=-1),
        )

        return {
            "data": data,
            "node_features": node_features,
            "edge_features": edge_features,
        }

    @conditional_grad(torch.enable_grad())
    def forward(self, data: torch_geometric.data.Batch):
        # gradient force
        if self.regress_forces and not self.global_cfg.direct_force:
            data.pos.requires_grad_(True)

        # preprocess data
        x = self.data_preprocess(data)

        return self.forward_fn(x)

    @torch.jit.ignore
    def no_weight_decay(self):
        return no_weight_decay(self)


class EScAIPHeadBase(nn.Module, HeadInterface):
    def __init__(self, backbone: EScAIPBackbone):
        super().__init__()
        self.global_cfg = backbone.global_cfg
        self.molecular_graph_cfg = backbone.molecular_graph_cfg
        self.gnn_cfg = backbone.gnn_cfg
        self.reg_cfg = backbone.reg_cfg

    def post_init(self, gain=1.0):
        # init weights
        self.apply(partial(init_linear_weights, gain=gain))

        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return no_weight_decay(self)


@registry.register_model("EScAIP_direct_force_head")
class EScAIPDirectForceHead(EScAIPHeadBase):
    def __init__(self, backbone: EScAIPBackbone):
        super().__init__(backbone)
        self.force_direction_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Vector",
        )
        self.force_magnitude_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        self.post_init()

    def compiled_forward(self, edge_features, node_features, data: GraphAttentionData):
        # get force direction from edge features
        force_direction = self.force_direction_layer(
            edge_features
        )  # (num_nodes, max_neighbor, 3)
        force_direction = (
            force_direction * data.edge_direction
        )  # (num_nodes, max_neighbor, 3)
        force_direction = (force_direction * data.neighbor_mask.unsqueeze(-1)).sum(
            dim=1
        )  # (num_nodes, 3)
        # get force magnitude from node readouts
        force_magnitude = self.force_magnitude_layer(node_features)  # (num_nodes, 1)
        # get output force
        return force_direction * force_magnitude  # (num_nodes, 3)

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        force_output = self.forward_fn(
            edge_features=emb["edge_features"],
            node_features=emb["node_features"],
            data=emb["data"],
        )

        return unpad_results(
            results={"forces": force_output},
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )


@registry.register_model("EScAIP_energy_head")
class EScAIPEnergyHead(EScAIPHeadBase):
    def __init__(self, backbone: EScAIPBackbone):
        super().__init__(backbone)
        self.energy_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        self.post_init(gain=0.01)

    def compiled_forward(self, node_features, data: GraphAttentionData):
        energy_output = self.energy_layer(node_features)

        # the following not compatible with torch.compile (grpah break)
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")

        energy_output = compilable_scatter(
            src=energy_output,
            index=data.node_batch,
            dim_size=data.graph_padding_mask.shape[0],
            dim=0,
            reduce="sum",
        )
        return energy_output

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_output = self.forward_fn(
            node_features=emb["node_features"],
            data=emb["data"],
        )
        return unpad_results(
            results={"energy": energy_output},
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )


@registry.register_model("EScAIP_grad_energy_force_head")
class EScAIPGradientEnergyForceHead(EScAIPEnergyHead):
    """
    Do not support torch.compile
    """

    def __init__(self, backbone: EScAIPBackbone):
        super().__init__(backbone)

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_output = self.energy_layer(emb["node_features"])

        # the following not compatible with torch.compile (grpah break)
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")

        energy_output = compilable_scatter(
            src=energy_output,
            index=emb["data"].node_batch,
            dim_size=emb["data"].graph_padding_mask.shape[0],
            dim=0,
            reduce="mean",
        )

        forces_output = (
            -1
            * torch.autograd.grad(
                energy_output.sum(), data.pos, create_graph=self.training
            )[0]
        )

        return unpad_results(
            results={"energy": energy_output, "forces": forces_output},
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )


@registry.register_model("EScAIP_rank2_head")
class EScAIPRank2Head(EScAIPHeadBase):
    """
    Rank-2 head for EScAIP model. Modified from the Rank2Block for Equiformer V2.
    """

    def __init__(
        self,
        backbone: EScAIPBackbone,
        output_name: str = "stress",
    ):
        super().__init__(backbone)
        self.output_name = output_name
        self.scalar_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )
        self.irreps2_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        self.post_init()

    def compiled_forward(self, node_features, edge_features, data: GraphAttentionData):
        sphere_irrep2 = o3.spherical_harmonics(
            2, data.edge_direction, True
        ).detach()  # (num_nodes, max_neighbor, 5)

        # map from invariant to irrep2
        edge_irrep2 = (
            sphere_irrep2[:, :, :, None] * edge_features[:, :, None, :]
        )  # (num_nodes, max_neighbor, 5, h)

        # sum over neighbors
        neighbor_count = data.neighbor_mask.sum(dim=1, keepdim=True) + 1e-5
        neighbor_count = neighbor_count.to(edge_irrep2.dtype)
        node_irrep2 = (
            edge_irrep2 * data.neighbor_mask.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1) / neighbor_count.unsqueeze(-1)  # (num_nodes, 5, h)

        irrep2_output = self.irreps2_layer(node_irrep2)  # (num_nodes, 5, 1)
        scalar_output = self.scalar_layer(node_features)  # (num_nodes, 1)

        # get graph level output
        irrep2_output = compilable_scatter(
            src=irrep2_output.view(-1, 5),
            index=data.node_batch,
            dim_size=data.graph_padding_mask.shape[0],
            dim=0,
            reduce="mean",
        )
        scalar_output = compilable_scatter(
            src=scalar_output.view(-1, 1),
            index=data.node_batch,
            dim_size=data.graph_padding_mask.shape[0],
            dim=0,
            reduce="mean",
        )
        return irrep2_output, scalar_output.view(-1)

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        irrep2_output, scalar_output = self.forward_fn(
            node_features=emb["node_features"],
            edge_features=emb["edge_features"],
            data=emb["data"],
        )
        output = {
            f"{self.output_name}_isotropic": scalar_output.unsqueeze(1),
            f"{self.output_name}_anisotropic": irrep2_output,
        }

        return unpad_results(
            results=output,
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )

###

@registry.register_model("EScAIP_ChargeS_backbone")
class EScAIPChargeSBackbone(nn.Module, GraphModelMixin):
    """
    Efficiently Scaled Attention Interactomic Potential with Charge and Spin (EScAIP-ChargeS) backbone model.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        # load configs
        cfg = init_configs(EScAIPConfigs, kwargs)
        self.global_cfg = cfg.global_cfg
        self.molecular_graph_cfg = cfg.molecular_graph_cfg
        self.gnn_cfg = cfg.gnn_cfg
        self.reg_cfg = cfg.reg_cfg

        # for trainer
        self.regress_forces = cfg.global_cfg.regress_forces
        self.use_pbc = cfg.molecular_graph_cfg.use_pbc

        self.use_global_charge = cfg.global_cfg.use_global_charge
        self.use_global_spin = cfg.global_cfg.use_global_spin
        self.regress_charges = cfg.global_cfg.regress_charges
        self.regress_spins = cfg.global_cfg.regress_spins
        self.electronic_intermediate = cfg.global_cfg.electronic_intermediate

        # graph generation
        self.use_pbc_single = (
            self.molecular_graph_cfg.use_pbc_single
        )  # TODO: remove this when FairChem fixes the bug
        generate_graph_fn = partial(
            self.generate_graph,
            cutoff=self.molecular_graph_cfg.max_radius,
            max_neighbors=self.molecular_graph_cfg.max_neighbors,
            use_pbc=self.molecular_graph_cfg.use_pbc,
            otf_graph=self.molecular_graph_cfg.otf_graph,
            enforce_max_neighbors_strictly=self.molecular_graph_cfg.enforce_max_neighbors_strictly,
            use_pbc_single=self.molecular_graph_cfg.use_pbc_single,
        )

        # data preprocess
        self.data_preprocess = partial(
            data_preprocess,
            generate_graph_fn=generate_graph_fn,
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
        )

        ## Model Components

        # Input Block
        self.input_block = InputBlockChargeSpin(
            global_cfg=self.global_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
        )

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                EfficientGraphAttentionBlockChargeSpin(
                    global_cfg=self.global_cfg,
                    molecular_graph_cfg=self.molecular_graph_cfg,
                    gnn_cfg=self.gnn_cfg,
                    reg_cfg=self.reg_cfg,
                )
                for _ in range(self.gnn_cfg.num_layers)
            ]
        )

        # Readout Layer
        self.readout_layers = nn.ModuleList(
            [
                ReadoutBlock(
                    global_cfg=self.global_cfg,
                    gnn_cfg=self.gnn_cfg,
                    reg_cfg=self.reg_cfg,
                )
                for _ in range(self.gnn_cfg.num_layers + 1)
            ]
        )

        # Output Projection
        self.output_projection = OutputProjection(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
        )

        # init weights
        self.apply(init_linear_weights)

        # enable torch.set_float32_matmul_precision('high') if not using fp16 backbone
        if not self.global_cfg.use_fp16_backbone:
            torch.set_float32_matmul_precision("high")
        torch._logging.set_logs(recompiles=True)

        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )


    def compiled_forward(self, data: GraphAttentionDataChargeSpin):
        # input block
        node_features, edge_features = self.input_block(data)

        # input readout
        readouts = self.readout_layers[0](node_features, edge_features)
        node_readouts = [readouts[0]]
        edge_readouts = [readouts[1]]

        # transformer blocks
        for idx in range(self.gnn_cfg.num_layers):
            node_features, edge_features = self.transformer_blocks[idx](
                data, node_features, edge_features
            )
            readouts = self.readout_layers[idx + 1](node_features, edge_features)
            node_readouts.append(readouts[0])
            edge_readouts.append(readouts[1])

        node_features, edge_features = self.output_projection(
            node_readouts=torch.cat(node_readouts, dim=-1),
            edge_readouts=torch.cat(edge_readouts, dim=-1),
        )

        return {
            "data": data,
            "node_features": node_features,
            "edge_features": edge_features,
        }

    @conditional_grad(torch.enable_grad())
    def forward(self, data: torch_geometric.data.Batch):
        # gradient force
        if self.regress_forces and not self.global_cfg.direct_force:
            data.pos.requires_grad_(True)

        # preprocess data
        x = self.data_preprocess(data)

        return self.forward_fn(x)

    @torch.jit.ignore
    def no_weight_decay(self):
        return no_weight_decay(self)


class EScAIPChargeSHeadBase(nn.Module, HeadInterface):
    def __init__(self, backbone: EScAIPChargeSBackbone):
        super().__init__()
        self.global_cfg = backbone.global_cfg
        self.molecular_graph_cfg = backbone.molecular_graph_cfg
        self.gnn_cfg = backbone.gnn_cfg
        self.reg_cfg = backbone.reg_cfg

    def post_init(self, gain=1.0):
        # init weights
        self.apply(partial(init_linear_weights, gain=gain))

        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return no_weight_decay(self)


@registry.register_model("EScAIP_ChargeS_direct_force_head")
class EScAIPChargeSDirectForceHead(EScAIPChargeSHeadBase):
    def __init__(self, backbone: EScAIPChargeSBackbone):
        super().__init__(backbone)
        self.force_direction_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Vector",
        )
        self.force_magnitude_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        self.post_init()

    def compiled_forward(self, edge_features, node_features, data: GraphAttentionDataChargeSpin):
        # get force direction from edge features
        force_direction = self.force_direction_layer(
            edge_features
        )  # (num_nodes, max_neighbor, 3)
        force_direction = (
            force_direction * data.edge_direction
        )  # (num_nodes, max_neighbor, 3)
        force_direction = (force_direction * data.neighbor_mask.unsqueeze(-1)).sum(
            dim=1
        )  # (num_nodes, 3)
        # get force magnitude from node readouts
        force_magnitude = self.force_magnitude_layer(node_features)  # (num_nodes, 1)
        # get output force
        return force_direction * force_magnitude  # (num_nodes, 3)

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        force_output = self.forward_fn(
            edge_features=emb["edge_features"],
            node_features=emb["node_features"],
            data=emb["data"],
        )

        return unpad_results(
            results={"forces": force_output},
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )


@registry.register_model("EScAIP_ChargeS_energy_head")
class EScAIPChargeSEnergyHead(EScAIPChargeSHeadBase):
    def __init__(self, backbone: EScAIPChargeSBackbone):
        super().__init__(backbone)
        self.energy_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        self.post_init(gain=0.01)

    def compiled_forward(self, node_features, data: GraphAttentionDataChargeSpin):
        energy_output = self.energy_layer(node_features)

        # the following not compatible with torch.compile (grpah break)
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")

        energy_output = compilable_scatter(
            src=energy_output,
            index=data.node_batch,
            dim_size=data.graph_padding_mask.shape[0],
            dim=0,
            reduce="sum",
        )
        return energy_output

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_output = self.forward_fn(
            node_features=emb["node_features"],
            data=emb["data"],
        )
        return unpad_results(
            results={"energy": energy_output},
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )


@registry.register_model("EScAIP_ChargeS_grad_energy_force_head")
class EScAIPChargeSGradientEnergyForceHead(EScAIPChargeSEnergyHead):
    """
    Do not support torch.compile
    """

    def __init__(self, backbone: EScAIPChargeSBackbone):
        super().__init__(backbone)

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        energy_output = self.energy_layer(emb["node_features"])

        # the following not compatible with torch.compile (graph break)
        # energy_output = torch_scatter.scatter(energy_output, node_batch, dim=0, reduce="sum")

        energy_output = compilable_scatter(
            src=energy_output,
            index=emb["data"].node_batch,
            dim_size=emb["data"].graph_padding_mask.shape[0],
            dim=0,
            reduce="mean",  # Mean here, sum elsewhere?
        )

        forces_output = (
            -1
            * torch.autograd.grad(
                energy_output.sum(), data.pos, create_graph=self.training
            )[0]
        )

        return unpad_results(
            results={"energy": energy_output, "forces": forces_output},
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )


@registry.register_model("EScAIP_ChargeS_rank2_head")
class EScAIPChargeSRank2Head(EScAIPChargeSHeadBase):
    """
    Rank-2 head for EScAIP-ChargeS model. Modified from the Rank2Block for Equiformer V2.
    """

    def __init__(
        self,
        backbone: EScAIPChargeSBackbone,
        output_name: str = "stress",
    ):
        super().__init__(backbone)
        self.output_name = output_name
        self.scalar_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )
        self.irreps2_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        self.post_init()

    def compiled_forward(self, node_features, edge_features, data: GraphAttentionDataChargeSpin):
        sphere_irrep2 = o3.spherical_harmonics(
            2, data.edge_direction, True
        ).detach()  # (num_nodes, max_neighbor, 5)

        # map from invariant to irrep2
        edge_irrep2 = (
            sphere_irrep2[:, :, :, None] * edge_features[:, :, None, :]
        )  # (num_nodes, max_neighbor, 5, h)

        # sum over neighbors
        neighbor_count = data.neighbor_mask.sum(dim=1, keepdim=True) + 1e-5
        neighbor_count = neighbor_count.to(edge_irrep2.dtype)
        node_irrep2 = (
            edge_irrep2 * data.neighbor_mask.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1) / neighbor_count.unsqueeze(-1)  # (num_nodes, 5, h)

        irrep2_output = self.irreps2_layer(node_irrep2)  # (num_nodes, 5, 1)
        scalar_output = self.scalar_layer(node_features)  # (num_nodes, 1)

        # get graph level output
        irrep2_output = compilable_scatter(
            src=irrep2_output.view(-1, 5),
            index=data.node_batch,
            dim_size=data.graph_padding_mask.shape[0],
            dim=0,
            reduce="mean",
        )
        scalar_output = compilable_scatter(
            src=scalar_output.view(-1, 1),
            index=data.node_batch,
            dim_size=data.graph_padding_mask.shape[0],
            dim=0,
            reduce="mean",
        )
        return irrep2_output, scalar_output.view(-1)

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        irrep2_output, scalar_output = self.forward_fn(
            node_features=emb["node_features"],
            edge_features=emb["edge_features"],
            data=emb["data"],
        )
        output = {
            f"{self.output_name}_isotropic": scalar_output.unsqueeze(1),
            f"{self.output_name}_anisotropic": irrep2_output,
        }

        return unpad_results(
            results=output,
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )

@registry.register_model("EScAIP_ChargeS_charge_spin_head")
class EScAIPChargeSChargeSpinHead(EScAIPChargeSHeadBase):
    def __init__(self, backbone: EScAIPChargeSBackbone, key: str = "atomic_partial_charges"):
        super().__init__(backbone)
        self.cs_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        self.post_init(gain=0.01)

        self.key = key

    def compiled_forward(self, node_features, data: GraphAttentionDataChargeSpin):
        cs_output = self.cs_layer(node_features)

        return cs_output

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cs_output = self.forward_fn(
            node_features=emb["node_features"],
            data=emb["data"],
        )
        return unpad_results(
            results={self.key: cs_output},
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )
