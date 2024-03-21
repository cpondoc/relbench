from typing import List

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_geometric.nn import MLP
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType

from relgym.config import cfg
from relgym.models.feature_encoder import HeteroEncoder, HeteroTemporalEncoder
from relgym.models.gnn import HeteroGNN


def create_model(data, col_stats_dict, task, to_device=True, shallow_list: List[NodeType] = []):
    r"""
    Create model for graph machine learning

    Args:
        to_device (string): The device that the model will be transferred to
    """

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder = HeteroEncoder(
                channels=cfg.model.channels,
                node_to_col_names_dict={
                    node_type: data[node_type].tf.col_names_dict
                    for node_type in data.node_types
                },
                node_to_col_stats=col_stats_dict,
                torch_frame_model_cls=cfg.torch_frame_model.torch_frame_model_cls,
                torch_frame_model_kwargs={
                    "channels": cfg.torch_frame_model.channels,
                    "num_layers": cfg.torch_frame_model.num_layers,
                },
            )
            self.temporal_encoder = HeteroTemporalEncoder(
                node_types=[
                    node_type
                    for node_type in data.node_types
                    if "time" in data[node_type]
                ],
                channels=cfg.model.channels,
            )
            selfjoin_kwargs = {
                "use_self_join": cfg.selfjoin.use_self_join,
                "node_type_considered": cfg.selfjoin.node_type_considered,
                "num_filtered": cfg.selfjoin.num_filtered,
                "normalize_score": cfg.selfjoin.normalize_score,
                "selfjoin_aggr": cfg.selfjoin.aggr,
                "retrieve_label": cfg.selfjoin.retrieve_label,
            }
            self.gnn = HeteroGNN(
                conv=cfg.model.conv,
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=cfg.model.channels,
                aggr=cfg.model.aggr,
                hetero_aggr=cfg.model.hetero_aggr,
                num_layers=cfg.model.num_layers,
                feature_dropout=cfg.model.feature_dropout,
                **selfjoin_kwargs,
            )
            self.head = MLP(
                cfg.model.channels,
                out_channels=cfg.model.out_channels,
                norm=cfg.model.norm,
                num_layers=1,
            )
            self.embedding_dict = ModuleDict(
                {
                    node: Embedding(data.num_nodes_dict[node], cfg.model.channels)
                    for node in shallow_list
                }
            )

        def forward(
                self,
                batch: HeteroData,
                entity_table,
                **kwargs,
        ) -> Tensor:
            x_dict = self.encoder(batch.tf_dict)

            if cfg.model.use_time_emb:
                rel_time_dict = self.temporal_encoder(batch[entity_table].seed_time, batch.time_dict, batch.batch_dict)
            else:
                rel_time_dict = {}

            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

            for node_type, embedding in self.embedding_dict.items():
                x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

            # Perturb the edges
            if cfg.model.perturb_edges == "rand_perm":
                for key in batch.edge_index_dict:
                    rand_perm = torch.randperm(batch.edge_index_dict[key].size(1)).to(batch.edge_index_dict[key].device)
                    batch.edge_index_dict[key][1] = batch.edge_index_dict[key][1][rand_perm]

            # Mask input features
            if cfg.model.mask_features:
                for node_type, feature in x_dict.items():
                    x_dict[node_type] = torch.zeros_like(feature)

            # Get the embeddings of the bank
            bank_batch = kwargs.get("bank_batch", None)
            if bank_batch is not None:
                bank_x_dict = self.encoder(bank_batch.tf_dict)
                bank_rel_time_dict = self.temporal_encoder(bank_batch[entity_table].seed_time,
                                                           bank_batch.time_dict, bank_batch.batch_dict)
                for node_type, rel_time in bank_rel_time_dict.items():
                    bank_x_dict[node_type] = bank_x_dict[node_type] + rel_time
                bank_y = bank_batch[entity_table].y
                bank_seed_time = bank_batch[entity_table].seed_time
            else:
                bank_x_dict = None
                bank_y = None
                bank_seed_time = None

            if cfg.model.perturb_edges != "drop_all":
                kwargs = {
                    'bank_x_dict': bank_x_dict,
                    'bank_y': bank_y,
                    'seed_time': batch[entity_table].seed_time,
                    'bank_seed_time': bank_seed_time,
                }
                x_dict = self.gnn(
                    x_dict,
                    batch.edge_index_dict,
                    batch.num_sampled_nodes_dict,
                    batch.num_sampled_edges_dict,
                    **kwargs,
                )

            return self.head(x_dict[entity_table][: batch[entity_table].seed_time.size(0)])

    model = Model()
    if to_device:
        model.to(torch.device(cfg.device))
    return model
