from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from relbench.external.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder, HeteroGraphUnified


class Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        with_marking: bool = False, 
        conv_type: str = "sage",
        num_breakings: int = 10,
    ):
        super().__init__()

        self.encoder = HeteroEncoder( 
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=2*channels if with_marking else channels,
        )

        if conv_type == "sage":
            self.gnn = HeteroGraphSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=2*channels if with_marking else channels,
                aggr=aggr,
                num_layers=num_layers,
            )
        elif conv_type == "unified":
            self.gnn = HeteroGraphUnified(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=2*channels if with_marking else channels,
                aggr=aggr,
                num_layers=num_layers,
                num_breakings=num_breakings,
            )
        
        self.head = MLP(
            2*channels if with_marking else channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], 2*channels if with_marking else channels)
                for node in shallow_list
            }
        )

        self.with_marking = with_marking
        self.conv_type = conv_type

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)


    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        marking: Tensor = None,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        if self.with_marking:
            for node_type, embedding in x_dict.items():
                if node_type == entity_table:
                    x_dict[node_type] = torch.stack([torch.hstack([x_dict[node_type], m]) for m in marking])
                else:
                    zeros = torch.zeros_like(x_dict[node_type]).to(x_dict[node_type].device)
                    x_dict[node_type] = torch.stack([torch.hstack([x_dict[node_type], zeros])  for m in marking])


        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time.unsqueeze(0)

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id).unsqueeze(0)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
            entity_table if self.conv_type == "unified" else None,
        )

        if x_dict[entity_table].dim() == 3:
            return self.head(x_dict[entity_table][:,: seed_time.size(0)]).squeeze()
        else:
            return self.head(x_dict[entity_table][: seed_time.size(0)])

