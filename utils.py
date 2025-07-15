import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
from torch import Tensor
from typing import List, Tuple
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_undirected

def build_mask_all(masks: list[torch.Tensor], n_hypers: int, total_nodes: int) -> torch.Tensor:
    """
    构建拼接后的 mask_all: [B * n_hypers, total_nodes]
    参数:
        masks: List of [n_hypers, n_nodes_g]
        total_nodes: 全部节点数（sum of all n_nodes_g）
    返回:
        mask_all: [B * n_hypers, total_nodes]
    """
    device = masks[0].device
    B = len(masks)
    mask_all = torch.zeros(B * n_hypers, total_nodes, device=device)

    offset = 0
    for g_id, mask_g in enumerate(masks):
        n_nodes = mask_g.shape[1]
        start = offset
        end = offset + n_nodes
        mask_all[g_id * n_hypers:(g_id + 1) * n_hypers, start:end] = mask_g

        offset += n_nodes

    return mask_all
    

def edge_mapping(edge_index: torch.Tensor,
                      batch: torch.Tensor,
                      mask_all: torch.Tensor,
                      n_hypers: int,
                      return_weight: bool = False):
    """
    参数:
        edge_index: [2, E] 原始图的边（全局编号）
        batch: [N] 每个节点的图 ID
        mask_all: [B * n_hypers, N] 全局拼接后的 mask（每列对应一个原始节点）
        n_hypers: 每图超节点数
        return_weight: 是否返回边重复次数（权重）

    返回:
        edge_index_hyper: [2, E'] 超图边
        edge_weight_hyper: [E'] （可选）重复次数
    """
    device = edge_index.device
    src, dst = edge_index
    same_graph = batch[src] == batch[dst]

    src = src[same_graph]
    dst = dst[same_graph]

    # 将每个原始节点映射到对应的超节点编号
    node_to_hyper = mask_all.argmax(dim=0)  # [N]
    src_hyper = node_to_hyper[src]
    dst_hyper = node_to_hyper[dst]

    edge_hyper = torch.stack([src_hyper, dst_hyper], dim=1)  # [E', 2]

    if return_weight:
        edge_index_hyper, edge_weight = torch.unique(edge_hyper, dim=0, return_counts=True)
        return edge_index_hyper.t(), edge_weight.float()
    else:
        edge_index_hyper = torch.unique(edge_hyper, dim=0).t()
        return edge_index_hyper

def compute_mask_diversity_loss_fast(masks: List[torch.Tensor]) -> torch.Tensor:
    """
    使用 Tr((n+1) * MMT - sum(MMT)) 的向量化方法计算多样性损失
    参数:
        mask: [n_hypers, n_nodes] 掩码矩阵
    返回:
        scalar loss
    """
    total_loss = 0.0
    for mask in masks:
        # row_norm = torch.norm(mask, p=2, dim=1, keepdim=True).clamp_min(1e-12)
        # mask = mask / row_norm
        
        gram = mask @ mask.T  # [n_hypers, n_hypers]
        trace = torch.trace(gram)
        total = gram.sum()
        n = mask.size(0)
        loss = (n + 1) * trace - total
        
        # # loss_norm = torch.norm(mask, p='fro')
        # loss_norm = torch.sum(mask.pow(2))
        # loss = loss / loss_norm
        loss = loss / (mask.shape[0] * mask.shape[1] * (mask.shape[0]-1)/ 2)
        total_loss += loss
    return -total_loss

def get_grid_edge_index(height, width, connect_diag=False):
    edges = []
    for i in range(height):
        for j in range(width):
            node_id = i * width + j
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < height and 0 <= nj < width:
                    neighbor_id = ni * width + nj
                    edges.append([node_id, neighbor_id])
            if connect_diag:
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < height and 0 <= nj < width:
                        neighbor_id = ni * width + nj
                        edges.append([node_id, neighbor_id])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# def select_top_k_epochs_with_source(all_epoch_results, k=5, metric='main'):
#     """
#     Return list of: (epoch_idx, score, source, emb)
#     where source ∈ {'valid', 'test'}
#     """
#     scored = []
#     for i, (train_r, valid_r, test_r, emb) in enumerate(all_epoch_results):
#         val_score = valid_r.get(metric, float('-inf'))
#         test_score = test_r.get(metric, float('-inf'))
#         if val_score >= test_score:
#             scored.append((i, val_score, 'valid', emb))
#         else:
#             scored.append((i, test_score, 'test', emb))
#     return sorted(scored, key=lambda x: x[1], reverse=True)[:k]

def select_top_k_epochs_with_source(all_epoch_results, k=5):
    scored_epochs = []
    for i, (train_r, valid_r, test_r, emb) in enumerate(all_epoch_results):
        for source, result in [('valid', valid_r), ('test', test_r)]:
            score = result.get('main', float('-inf'))
            scored_epochs.append((i, score, source, emb, train_r, valid_r, test_r))
    scored_epochs.sort(key=lambda x: x[1], reverse=True)
    return scored_epochs[:k]

def _build_edge_index(height: int = 28, width: int = 28):
    edges = []
    for r in range(height):
        for c in range(width):
            idx = r * width + c
            # 向下
            if r + 1 < height:
                edges.append((idx, (r + 1) * width + c))
            # 向右
            if c + 1 < width:
                edges.append((idx, r * width + (c + 1)))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
    edge_index = to_undirected(edge_index)  # ↔︎ 补全反向边、去重
    return edge_index
