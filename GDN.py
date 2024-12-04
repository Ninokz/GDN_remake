import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from dataprocess import DataProcess
from graph_layer import GraphLayer
from utils import get_device


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num=512):
        """
        :param in_num: 输入特征维度
        :param node_num: 节点数量[参数未使用]
        :param layer_num: MLP中的层数
        :param inter_num: 中间层维数
        """
        super(OutLayer, self).__init__()
        modules = []
        # 对指定的层数进行构建MLP
        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num - 1:
                # 用于确定某个线性层（nn.Linear）的输入和输出维度
                # 如果只有一层，线性层会直接从输入维度 in_num 映射到一个输出（输出维度为 1）
                # 如果多于一层，线性层会从中间层的维数 inter_num 映射到一个输出（输出维度同样为 1）
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                # 设置非最后一层的每层的输入输出特征维度
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())
        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x
        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                # 调整输入维度格式使之能正确应用批归一化
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                # 对于非nn.BatchNorm1d类型的模块，直接对out应用当前模块
                out = mod(out)
        return out


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

        self.att_weight_1 = None
        self.edge_index_1 = None

    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
        out = self.bn(out)
        return self.relu(out)


class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64,
                 out_layer_inter_dim=256, input_dim=10, out_layer_num=1,
                 top_k=20):
        super(GDN, self).__init__()
        device = get_device()

        self.edge_index_sets = edge_index_sets

        # node -> embed
        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_out_layer_in = nn.BatchNorm1d(embed_dim)

        # 为每个边索引集创建一个GNNLayer, 用于处理图的不同关系类型
        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim + embed_dim, heads=1) for i in range(edge_set_num)
        ])
        self.node_embedding = None
        self.top_k = top_k
        self.learned_graph = None

        # 用于将 GNNLayer 的输出转换为最终的节点表示
        self.out_layer = OutLayer(dim * edge_set_num, node_num, out_layer_num, inter_num=out_layer_inter_dim)
        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index):
        device = data.device

        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets
        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            # 这段代码原作者是未注释掉的，但是阅读后发现该代码段落未被后续使用
            # edge_num = edge_index.shape[1]
            # cache_edge_index = self.cache_edge_index_sets[i]
            # if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
            #     self.cache_edge_index_sets[i] = DataProcess.get_batch_edge_index(edge_index, batch_num, node_num).to(
            #         device)
            # batch_edge_index = self.cache_edge_index_sets[i]

            all_embeddings = self.embedding(torch.arange(node_num).to(device))
            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)
            weights = weights_arr.view(node_num, -1)
            # weights shape: rows*columns -> features_length * 64

            # v_j cdot v_i^T
            # 节点间的仅用余弦相似度计算
            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
            cos_ji_mat = cos_ji_mat / normed_mat
            

            # Top K选择：通过torch.topk选择与每个节点最相似的Top K个节点的索引top_k_indices_ji，用于构建学习到的局部图结构
            dim = weights.shape[-1]
            top_k_num = self.top_k
            top_k_indices_ji = torch.topk(cos_ji_mat, top_k_num, dim=-1)[1]

            self.learned_graph = top_k_indices_ji
            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, top_k_num).flatten().to(device).unsqueeze(0)
            gated_j = top_k_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
            batch_gated_edge_index = DataProcess.get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num * batch_num,
                                         embedding=all_embeddings)
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        # v_i:self.embedding(indexes) cdot z_i:x
        indexes = torch.arange(0, node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))

        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_out_layer_in(out))
        out = out.permute(0, 2, 1)
        out = self.dp(out)

        # MLP
        out = self.out_layer(out)
        out = out.view(-1, node_num)
        return out
