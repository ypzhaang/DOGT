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
from utils import build_mask_all, edge_mapping, compute_mask_diversity_loss_fast

class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super(GraphConvLayer, self).__init__()

        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index, x0):
        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()
        value = torch.ones_like(row) * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        x = matmul(adj, x)  # [N, D]

        if self.use_init:
            x = torch.cat([x, x0], 1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)

        return x


class GraphConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_init=False, use_act=True):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_channels, hidden_channels, use_weight, use_init))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])
            if self.use_bn:
                x = self.bns[i+1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
        return x

class TransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, output_attn=False):
        # feature transformation
        qs = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        ks = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            vs = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            vs = source_input.reshape(-1, 1, self.out_channels)

        # normalize input
        qs = qs / torch.norm(qs, p=2)  # [N, H, M]
        ks = ks / torch.norm(ks, p=2)  # [L, H, M]
        N = qs.shape[0]

        # numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        attention_num += N * vs

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer  # [N, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
            attention = attention / normalizer

        final_output = attn_output.mean(dim=1)

        if output_attn:
            return final_output, attention
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=True):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x):
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]

class Masker(nn.Module):
    def __init__(self, n_hypers: int, max_nodes: int):
        super().__init__()
        self.n_hypers = n_hypers
        self.max_nodes = max_nodes
        self.mask_logits = nn.Parameter(torch.Tensor(n_hypers, max_nodes, 2))
        nn.init.xavier_uniform_(self.mask_logits)

    def forward(self, node_counts: list[int]) -> tuple[list[Tensor], list[Tensor]]:
        masks, logits_out = [], []
        for n_nodes in node_counts:
            logits = self.mask_logits[:, :n_nodes, :] 
            probs = F.gumbel_softmax(torch.log_softmax(logits, dim=-1), tau=1, hard=False)[..., 1]
            masks.append(probs)            # [n_hypers, n_nodes]
            logits_out.append(logits[..., 1])
        return masks, logits_out
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mask_logits)

class HyBRiDConstructor(nn.Module):
    def __init__(self, n_hypers: int, max_nodes: int, dropout: float = 0.1):
        super().__init__()
        self.masker = Masker(n_hypers, max_nodes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, batch: Tensor) -> Tuple[Tensor, Tuple[List[Tensor], List[Tensor]], Tensor]:
        device = x.device
        num_graphs = int(batch.max().item()) + 1
        x_batch = [x[batch == i] for i in range(num_graphs)]
        node_counts = [xb.size(0) for xb in x_batch]

        D = x.size(1)
        masks, mask_logits = self.masker(node_counts)

        # vectorized aggregation
        total_nodes = sum(node_counts)
        mask_all = build_mask_all(masks, self.masker.n_hypers, total_nodes)  # [B*n_hypers, total_nodes]
        h_out = self.aggregate_vectorized(x, mask_all)                        # [B*n_hypers, D]
        h_out = self.dropout(h_out)

        batch_hyper = torch.arange(num_graphs, device=x.device).repeat_interleave(self.masker.n_hypers)
        return h_out, (masks, mask_logits), batch_hyper

    @staticmethod
    def aggregate_vectorized(x: Tensor, mask_all: Tensor) -> Tensor:
        x_exp = x.unsqueeze(0).expand(mask_all.size(0), -1, -1)
        mask_exp = mask_all.unsqueeze(-1)
        x_masked = x_exp * mask_exp
        return x_masked.sum(1) / (mask_exp.sum(1) + 1e-6)
    
    def reset_parameters(self):
        self.masker.reset_parameters()

class AGGT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5, trans_use_bn=True, trans_use_residual=True, trans_use_weight=True, trans_use_act=True,
                 gnn_num_layers=1, gnn_dropout=0.5, gnn_use_weight=True, gnn_use_init=False, gnn_use_bn=True, gnn_use_residual=True, gnn_use_act=True, gnn_for_origin_num_layers = 2, 
                 n_hypers = 4, hyper_dropout = 0.5, hyper_weight = 0.5, 
                 use_graph=True, graph_weight=0.8, aggregate='add'):
        super().__init__()
        self.trans_conv = TransConv(in_channels, hidden_channels, trans_num_layers, trans_num_heads, trans_dropout, trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act)
        self.graph_conv = GraphConv(in_channels, hidden_channels, gnn_num_layers, gnn_dropout, gnn_use_bn, gnn_use_residual, gnn_use_weight, gnn_use_init, gnn_use_act)
        self.graph_conv_for_origin = GraphConv(in_channels, hidden_channels, gnn_for_origin_num_layers, gnn_dropout, gnn_use_bn, gnn_use_residual, gnn_use_weight, gnn_use_init, gnn_use_act)
        self.constructor = HyBRiDConstructor(n_hypers=n_hypers, max_nodes=3000, dropout=hyper_dropout)
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.hyper_weight = hyper_weight
        self.n_hypers = n_hypers
        self.aggregate = aggregate

        if aggregate == 'add':
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.graph_conv.parameters()) if self.graph_conv is not None else []
        self.params2.extend(list(self.fc.parameters()))
        self.params3 = list(self.graph_conv_for_origin.parameters())
        self.params4 = list(self.constructor.parameters())

    def forward(self, x, edge_index, batch):
        start1 = time.time()
        x_origin = x
        x, (masks, mask_logits), batch_hyper = self.constructor(x, batch)
        diversity_loss = compute_mask_diversity_loss_fast(masks)
        # print("constructor time:", time.time()-start1)
        start2 = time.time()
        total_nodes = x_origin.size(0)
        mask_all = build_mask_all(masks, n_hypers=self.n_hypers, total_nodes=total_nodes)
        edge_index_hyper, edge_weight = edge_mapping(edge_index, batch, mask_all, self.n_hypers, return_weight=True)
        # print("edge mapping time:", time.time()-start2)
        start3 = time.time()
        x1 = self.trans_conv(x)
        # print("trans conv time:", time.time()-start3)
        start4 = time.time()
        if self.use_graph:
            x2 = self.graph_conv(x, edge_index_hyper)
            if self.aggregate == 'add':
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
        else:
            x = x1
        # print("merge time:", time.time()-start4)
        start5 = time.time()
        x3 = self.graph_conv_for_origin(x_origin, edge_index)
        # print("gnn time:", time.time()-start5)
        x3 = global_mean_pool(x3, batch)
        x = global_mean_pool(x, batch_hyper)
        x_emb = self.hyper_weight * x + (1-self.hyper_weight) * x3
        x = self.fc(x_emb)
        return x, diversity_loss / len(masks) * self.n_hypers, masks, x_emb
    
    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x) # [layer num, N, N]
        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()
            self.graph_conv_for_origin.reset_parameters()
        self.constructor.reset_parameters()