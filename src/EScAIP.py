from functools import partial
import logging
import time
import os

import torch
import torch.nn as nn
import torch_geometric

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import GraphModelMixin

from fairchem.core.models.gemnet_oc.layers.force_scaler import ForceScaler
from fairchem.core.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

from .configs import (
    EScAIPConfigs,
    GlobalConfigs,
    MolecularGraphConfigs,
    GraphNeuralNetworksConfigs,
    RegularizationConfigs,
    init_configs,
)
from .custom_types import GraphAttentionData
from .modules import (
    EfficientGraphAttentionBlock,
    InputBlock,
    OutputBlock,
    ReadoutBlock,
)
from .utils.graph_utils import (
    get_node_direction_expansion,
    convert_neighbor_list,
    map_neighbor_list,
    patch_singleton_atom,
    pad_batch,
    unpad_results,
)
from .utils.xformers_utils import (
    attn_bias_for_memory_efficient_attention,
)


@registry.register_model("EScAIP")
class EfficientlyScaledAttentionInteratomicPotential(nn.Module, GraphModelMixin):
    """ """

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

        # Electronic information
        self.use_global_charge = cfg.global_cfg.use_global_charge
        self.use_global_spin = cfg.global_cfg.use_global_spin
        self.regress_charges = cfg.global_cfg.regress_charges
        self.regress_spins = cfg.global_cfg.regress_spins
        self.electronic_intermediate = cfg.global_cfg.electronic_intermediate

        # edge distance expansion
        expansion_func = {
            "gaussian": GaussianSmearing,
            "sigmoid": SigmoidSmearing,
            "linear_sigmoid": LinearSigmoidSmearing,
            "silu": SiLUSmearing,
        }[self.molecular_graph_cfg.distance_function]

        self.edge_distance_expansion_func = expansion_func(
            0.0,
            self.molecular_graph_cfg.max_radius,
            self.gnn_cfg.edge_distance_expansion_size,
            basis_width_scalar=2.0,
        )

        self.exportable_model = EScAIPExportable(
            global_cfg=self.global_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
        )

        # gradient force
        if self.regress_forces and not self.global_cfg.direct_force:
            self.force_scaler = ForceScaler()

        # enable torch.set_float32_matmul_precision('high') if not using fp16 backbone
        if not self.global_cfg.use_fp16_backbone:
            torch.set_float32_matmul_precision("high")
        torch._logging.set_logs(recompiles=True)

    def export_and_compile_model(
        self,
        data: torch_geometric.data.Batch,
        export_dir: str = "./",  # TODO: set to checkpoint_dir, but no access now
    ):
        start_time = time.time()
        x = self.data_preprocess(data)
        export_path = os.path.join(export_dir, "exported_model.pt2")
        if not os.path.exists(export_path):
            logging.info("Exporting model...")
            exported_model = torch.export.export(self.exportable_model, (x,))
            torch.export.save(exported_model, export_path)
        else:
            logging.info(f"Loding model from {export_path}")
            exported_model = torch.export.load(export_path)
        logging.info("Compiling model...")
        compiled_exported_model = torch.compile(exported_model.module())
        logging.info("Warmup...")
        _ = compiled_exported_model(x)
        logging.info("Success")
        logging.info(f"Time elapsed: {time.time() - start_time:.2f}s")
        return compiled_exported_model

    def data_preprocess(self, data) -> GraphAttentionData:
        # atomic numbers
        atomic_numbers = data.atomic_numbers.long()

        # generate graph
        self.use_pbc_single = (
            self.molecular_graph_cfg.use_pbc_single
        )  # TODO: remove this when FairChem fixes the bug
        graph = self.generate_graph(
            data=data,
            cutoff=self.molecular_graph_cfg.max_radius,
            max_neighbors=self.molecular_graph_cfg.max_neighbors,
            use_pbc=self.molecular_graph_cfg.use_pbc,
            otf_graph=self.molecular_graph_cfg.otf_graph,
            enforce_max_neighbors_strictly=self.molecular_graph_cfg.enforce_max_neighbors_strictly,
            use_pbc_single=self.molecular_graph_cfg.use_pbc_single,
        )

        # sort edge index according to receiver node
        edge_index, edge_attr = torch_geometric.utils.sort_edge_index(
            graph.edge_index,
            [graph.edge_distance, graph.edge_distance_vec],
            sort_by_row=False,
        )
        edge_distance, edge_distance_vec = edge_attr[0], edge_attr[1]

        # edge directions (for direct force prediction, ref: gemnet)
        edge_direction = -edge_distance_vec / edge_distance[:, None]

        # edge distance expansion (ref: scn)
        edge_distance_expansion = self.edge_distance_expansion_func(edge_distance)

        # node direction expansion
        node_direction_expansion = get_node_direction_expansion(
            distance_vec=edge_distance_vec,
            edge_index=edge_index,
            lmax=self.gnn_cfg.node_direction_expansion_size - 1,
            num_nodes=data.num_nodes,
        )

        # convert to neighbor list
        neighbor_list, neighbor_mask, index_mapping = convert_neighbor_list(
            edge_index, self.molecular_graph_cfg.max_neighbors, data.num_nodes
        )

        # Charge and spin information
        # TODO: Check this
        num_nodes, _ = neighbor_list.shape
        if self.global_cfg.use_global_charge:
            charge = data.charge.long()
        else:
            charge = torch.zeros(num_nodes, dtype=torch.long, device=atomic_numbers.device)
        
        if self.global_cfg.use_global_spin:
            spin_multiplicity = data.spin_multiplicity.long()
        else:
            spin_multiplicity = torch.zeros(num_nodes, dtype=torch.long, device=atomic_numbers.device)

        try:
            atomic_partial_charges = data.atomic_partial_charges
            atomic_partial_spins = data.atomic_partial_spins
        except AttributeError:
            atomic_partial_charges = torch.zeros_like(atomic_numbers, dtype=torch.float)
            atomic_partial_spins = torch.zeros_like(atomic_partial_charges)

        # map neighbor list
        map_neighbor_list_ = partial(
            map_neighbor_list,
            index_mapping=index_mapping,
            max_neighbors=self.molecular_graph_cfg.max_neighbors,
            num_nodes=data.num_nodes,
        )
        edge_direction = map_neighbor_list_(edge_direction)
        edge_distance_expansion = map_neighbor_list_(edge_distance_expansion)

        # TODO: check that this works for datasets without charge/spin info
        # Probably there will need to be checks and empty variables will need to be filled
        # pad batch
        (
            atomic_numbers,
            charge,
            spin_multiplicity,
            atomic_partial_charges,
            atomic_partial_spins,
            node_direction_expansion,
            edge_distance_expansion,
            edge_direction,
            neighbor_list,
            neighbor_mask,
            node_batch,
            node_padding_mask,
            graph_padding_mask,
        ) = pad_batch(
            max_num_nodes_per_batch=self.molecular_graph_cfg.max_num_nodes_per_batch,
            atomic_numbers=atomic_numbers,
            charge=charge,
            spin_multiplicity=spin_multiplicity,
            atomic_partial_charges=atomic_partial_charges,
            atomic_partial_spins=atomic_partial_spins,
            node_direction_expansion=node_direction_expansion,
            edge_distance_expansion=edge_distance_expansion,
            edge_direction=edge_direction,
            neighbor_list=neighbor_list,
            neighbor_mask=neighbor_mask,
            node_batch=data.batch,
            num_graphs=data.num_graphs,
            batch_size=self.global_cfg.batch_size,
        )

        # patch singleton atom
        edge_direction, neighbor_list, neighbor_mask = patch_singleton_atom(
            edge_direction, neighbor_list, neighbor_mask
        )

        if self.gnn_cfg.atten_name == "xformers":
            attn_bias = attn_bias_for_memory_efficient_attention(neighbor_mask)
        elif self.gnn_cfg.atten_name in ["memory_efficient", "flash", "math"]:
            attn_bias = None
            torch.backends.cuda.enable_flash_sdp(self.gnn_cfg.atten_name == "flash")
            torch.backends.cuda.enable_mem_efficient_sdp(
                self.gnn_cfg.atten_name == "memory_efficient"
            )
            torch.backends.cuda.enable_math_sdp(self.gnn_cfg.atten_name == "math")
        else:
            raise NotImplementedError(
                f"Attention name {self.gnn_cfg.atten_name} not implemented"
            )

        # construct input data
        x = GraphAttentionData(
            charge=charge,
            spin_multiplicity=spin_multiplicity,
            atomic_numbers=atomic_numbers,
            atomic_partial_charges=atomic_partial_charges,
            atomic_partial_spins=atomic_partial_spins,
            node_direction_expansion=node_direction_expansion,
            edge_distance_expansion=edge_distance_expansion,
            edge_direction=edge_direction,
            neighbor_list=neighbor_list,
            neighbor_mask=neighbor_mask,
            node_batch=node_batch,
            node_padding_mask=node_padding_mask,
            graph_padding_mask=graph_padding_mask,
            attn_bias=attn_bias,
        )
        return x

    @conditional_grad(torch.enable_grad())
    def forward(self, data: torch_geometric.data.Batch):
        # gradient force
        if self.regress_forces and not self.global_cfg.direct_force:
            data.pos.requires_grad_(True)

        # preprocess data
        x = self.data_preprocess(data)

        if self.global_cfg.use_export:
            if not hasattr(self, "compiled_exported_model"):
                self.compiled_exported_model = self.export_and_compile_model(data)
            # forward pass
            energy_output, force_output, charge_output, spin_output = self.compiled_exported_model(x)
        elif self.global_cfg.use_compile:
            energy_output, force_output, charge_output, spin_output = torch.compile(self.exportable_model)(x)
        else:
            energy_output, force_output, charge_output, spin_output = self.exportable_model(x)

        outputs = {"energy": energy_output}

        if self.regress_forces:
            if not self.global_cfg.direct_force:
                force_output = self.force_scaler.calc_forces_and_update(
                    energy_output, data.pos
                )
            outputs["forces"] = force_output

        if self.regress_charges:
            outputs["charges"] = charge_output
        if self.regress_spins:
            outputs["spins"] = spin_output

        outputs = unpad_results(
            results=outputs,
            node_padding_mask=x.node_padding_mask,
            graph_padding_mask=x.graph_padding_mask,
        )

        return outputs

    @torch.jit.ignore
    def no_weight_decay(self):
        # no weight decay on layer norms and embeddings
        # ref: https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm, nn.RMSNorm)):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear):
                        if "weight" in parameter_name:
                            continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)


class EScAIPExportable(nn.Module):
    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()
        # load configs
        self.global_cfg = global_cfg
        self.molecular_graph_cfg = molecular_graph_cfg
        self.gnn_cfg = gnn_cfg
        self.reg_cfg = reg_cfg

        # Input Block
        self.input_block = InputBlock(
            global_cfg=self.global_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
        )

        # (Optional) layers to predict atomic partial charges & spins
        # The predicted partial charges/spins can then be used to inform energy/force prediction
        if (self.global_cfg.electronic_intermediate
            and self.gnn_cfg.num_layers_qmu > 0
            and (self.global_cfg.regress_charges or self.global_cfg.regress_spins)
        ):
            self.transformer_blocks_qmu = nn.ModuleList(
                [
                    # TODO: Make version that only uses node features?
                    EfficientGraphAttentionBlock(
                        global_cfg=self.global_cfg,
                        molecular_graph_cfg=self.molecular_graph_cfg,
                        gnn_cfg=self.gnn_cfg,
                        reg_cfg=self.reg_cfg,
                )
                for _ in range(self.gnn_cfg.num_layers_qmu)
                ]
            )

            self.readout_layers_qmu = nn.ModuleList(
                [
                    # TODO: need to make a version that only takes in node features
                    ReadoutBlock(
                        global_cfg=self.global_cfg,
                        gnn_cfg=self.gnn_cfg,
                        reg_cfg=self.reg_cfg,
                    )
                    for _ in range(self.gnn_cfg.num_layers_qmu + 1)
                ]
            )

            # TODO: version that only takes in node features?
            self.output_block_qmu = OutputBlock(
                global_cfg=self.global_cfg,
                molecular_graph_cfg=self.molecular_graph_cfg,
                gnn_cfg=self.gnn_cfg,
                reg_cfg=self.reg_cfg,
                energy=False,
                forces=False,
                charges=self.global_cfg.regress_charges,
                spins=self.global_cfg.regress_spins
            )

            # TODO: do we need another class here?
            self.input_block_with_qmu = InputBlock(
                global_cfg=self.global_cfg,
                molecular_graph_cfg=self.molecular_graph_cfg,
                gnn_cfg=self.gnn_cfg,
                reg_cfg=self.reg_cfg,
                use_charge_spin=True
            )
        else:
            self.transformer_blocks_qmu = None
            self.readout_layers_qmu = None
            self.output_block_qmu = None
            self.input_block_with_qmu = None

        # TODO: how to pass outputs from output_block_pre to new input block/self.transformer_blocks

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                EfficientGraphAttentionBlock(
                    global_cfg=self.global_cfg,
                    molecular_graph_cfg=self.molecular_graph_cfg,
                    gnn_cfg=self.gnn_cfg,
                    reg_cfg=self.reg_cfg,
                )
                for _ in range(self.gnn_cfg.num_layers_ef)
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
                for _ in range(self.gnn_cfg.num_layers_ef + 1)
            ]
        )

        # Output Block
        self.output_block = OutputBlock(
            global_cfg=self.global_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            charges=self.global_cfg.regress_charges,
            spins=self.global_cfg.regress_spins
        )

        # Init weights
        # self.linear_initializer = get_initializer("heorthogonal")
        self.linear_initializer = nn.init.xavier_uniform_
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            self.linear_initializer(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    # @torch.compile()
    # @torch.compile(mode='max-autotune')
    def forward(self, x: GraphAttentionData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        #TODO: (optional) charge and spin predictions
        # input block
        node_features, edge_features = self.input_block(x)

        charge_output, spin_output = None

        # (Optional) layers to predict charge (q) and spin (mu)
        if all(
            x is not None for x in [
                self.transformer_blocks_qmu,
                self.readout_layers_qmu,
                self.output_block_qmu,
                self.input_block_with_qmu
            ]
        ):
            # input readout
            readouts = self.readout_layers_qmu[0](node_features)
            node_readouts_qmu = [readouts[0]]

            # transformer blocks
            for idx in range(self.gnn_cfg.num_layers_qmu):
                node_features = self.transformer_blocks_qmu[idx](
                    x, node_features
                )
                readouts = self.readout_layers_qmu[idx + 1](node_features)
                node_readouts_qmu.append(readouts[0])

            # output block
            # TODO: make sure this makes sense (see layer TODO above)
            charge_output, spin_output = self.output_block_qmu(
                node_readouts=torch.cat(node_readouts, dim=-1),
                # edge_readouts=torch.cat(edge_readouts, dim=-1),
                node_batch=x.node_batch,
                # edge_direction=x.edge_direction,
                neighbor_mask=x.neighbor_mask,
                num_graphs=x.graph_padding_mask.shape[0],
            )

            node_features, edge_features = self.input_block_with_qmu(x, atomic_partial_charges=charge_output, atomic_partial_spins=spin_output)

        # input readout
        readouts = self.readout_layers[0](node_features, edge_features)
        node_readouts = [readouts[0]]
        edge_readouts = [readouts[1]]

        # transformer blocks
        for idx in range(self.gnn_cfg.num_layers):
            node_features, edge_features = self.transformer_blocks[idx](
                x, node_features, edge_features
            )
            readouts = self.readout_layers[idx + 1](node_features, edge_features)
            node_readouts.append(readouts[0])
            edge_readouts.append(readouts[1])

        # output block
        energy_output, force_output, charge_output_temp, spin_output_temp = self.output_block(
            node_readouts=torch.cat(node_readouts, dim=-1),
            edge_readouts=torch.cat(edge_readouts, dim=-1),
            node_batch=x.node_batch,
            edge_direction=x.edge_direction,
            neighbor_mask=x.neighbor_mask,
            num_graphs=x.graph_padding_mask.shape[0],
        )

        if charge_output is None:
            charge_output = charge_output_temp
        if spin_output is None:
            spin_output = spin_output_temp

        return energy_output, force_output, charge_output, spin_output
