import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict, Union, Tuple

import numpy as np
import pandas as pd
import torch
from model import Model
from text_embedder import GloveTextEmbedding
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, Table, TaskType, MultiEntityTask
from relbench.datasets import get_dataset
from relbench.modeling.graph import (
    get_node_train_table_input, make_pkey_fkey_graph,
    get_multi_entity_table_input, SAMPLE_NODE_TABLE_NAME
)
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task, get_task_names

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-event")
parser.add_argument("--task", type=str, default="user-attendance")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--temporal_strategy", type=str, default="uniform")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--include_label_tables",
    type=str,
    default="none",
    help="One of 'all', \
                    'task_only', and 'none'.",
)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("~/.cache/relbench_examples"),
)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)


stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

if args.include_label_tables == "all":
    tasks_to_add = get_task_names(args.dataset)
elif args.include_label_tables == "task_only":
    tasks_to_add = [args.task]
else:
    tasks_to_add = []


def _make_label_table(task: Union[EntityTask, MultiEntityTask]) -> Tuple[Table, Dict[str, stype]]:
    """ Makes a table with labels for the given task. To be used as features for training. """
    label_df = pd.concat([
        t.get_table("train").df,
        t.get_table("val").df,
        t.get_table("test", mask_input_cols=False).df
    ])
    # censor labels according to eval time
    label_df[t.time_col] = label_df[t.time_col] + t.timedelta
    col_to_table = {t.entity_col: t.entity_table} if isinstance(t, EntityTask) \
        else t.entities
    table = Table(
        df=label_df,
        fkey_col_to_pkey_table=col_to_table,
        pkey_col=None,
        time_col=task.time_col,
    )
    target_stype = stype.numerical if task.task_type == TaskType.REGRESSION \
        else stype.categorical
    col_to_stype = {
        task.entity_col: stype.numerical,
        task.time_col: stype.temporal,
        task.target_col: target_stype
    }
    return table, col_to_stype


db = dataset.get_db()
# add (time-censored) labels tables to the db
for task_name in tasks_to_add:
    t = get_task(args.dataset, task_name)
    if not isinstance(t, (EntityTask, MultiEntityTask)):
        continue
    labels_table_name = f"{task_name}_labels"
    ltable, col_to_stype = _make_label_table(t)
    db.table_dict[labels_table_name] = ltable
    col_to_stype_dict[labels_table_name] = col_to_stype

cache_name = (
    args.include_label_tables if args.include_label_tables != "task_only" else args.task
)

if isinstance(task, MultiEntityTask):
    # add samples nodes for prediction
    def _make_sample_table(task: MultiEntityTask) -> Tuple[Table, Dict[str, stype]]:
        """ Utility function to create sample nodes for multi-entity tasks. """
        sample_df = pd.concat([
            t.get_table("train", mask_input_cols=True).df.assign(split="train"),
            t.get_table("val", mask_input_cols=True).df.assign(split="val"),
            t.get_table("test", mask_input_cols=True).df.assign(split="test")
        ])
        sample_df["sample_id"] = np.arange(len(sample_df))
        table = Table(
            df=sample_df,
            fkey_col_to_pkey_table=task.entities,
            pkey_col="sample_id",
            time_col=task.time_col,
        )
        target_stype = stype.numerical if task.task_type == TaskType.REGRESSION \
            else stype.categorical
        col_to_stype = {
            table.pkey_col: stype.numerical,
            **{col: stype.numerical for col in task.entity_cols},
            task.time_col: stype.temporal,
            task.target_col: target_stype
        }
        return table, col_to_stype

    sample_table, col_to_stype = _make_sample_table(task)
    db.table_dict[SAMPLE_NODE_TABLE_NAME] = sample_table
    col_to_stype_dict[SAMPLE_NODE_TABLE_NAME] = col_to_stype


data, col_stats_dict = make_pkey_fkey_graph(
    db,
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}_{cache_name}/materialized",
)

clamp_min, clamp_max = None, None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    tune_metric = "mae"
    higher_is_better = False
    # Get the clamp value at inference time
    train_table = task.get_table("train")
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "multilabel_auprc_macro"
    higher_is_better = True
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

loader_dict: Dict[str, NeighborLoader] = {}
for split in ["train", "val", "test"]:
    if isinstance(task, MultiEntityTask):
        table_input = get_multi_entity_table_input(sample_table, task, split)
    else:
        table = task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=task)
    entity_table = table_input.nodes[0]
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=[int(args.num_neighbors / 2**i) for i in range(args.num_layers)],
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=args.batch_size,
        temporal_strategy=args.temporal_strategy,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )


def train() -> float:
    model.train()

    loss_accum = count_accum = 0
    steps = 0
    total_steps = min(len(loader_dict["train"]), args.max_steps_per_epoch)
    for batch in tqdm(loader_dict["train"], total=total_steps):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        loss = loss_fn(pred.float(), batch[entity_table].y.float())
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

        steps += 1
        if steps > args.max_steps_per_epoch:
            break

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(
            batch,
            task.entity_table,
        )
        if task.task_type == TaskType.REGRESSION:
            assert clamp_min is not None
            assert clamp_max is not None
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if task.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ]:
            pred = torch.sigmoid(pred)

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


model = Model(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=args.num_layers,
    channels=args.channels,
    out_channels=out_channels,
    aggr=args.aggr,
    norm="batch_norm",
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
state_dict = None
best_val_metric = -math.inf if higher_is_better else math.inf
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    val_pred = test(loader_dict["val"])
    val_metrics = task.evaluate(val_pred, task.get_table("val"))
    print(f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}")

    if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
        not higher_is_better and val_metrics[tune_metric] < best_val_metric
    ):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())


model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, task.get_table("val"))
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")
