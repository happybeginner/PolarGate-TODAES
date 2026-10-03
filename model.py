# coding=utf-8
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layers import create_spectral_features, MLP, PolarGateConv,  restPolarGateConv_MetaLA,  restPolarGateConv


class PolarGate(nn.Module):
    def __init__(
            self,
            args,
            device: torch.device,
            in_dim: int = 64,
            out_dim: int = 64,
            layer_num: int = 2,
            norm_emb: bool = False,
            **kwargs
    ):

        super().__init__(**kwargs)

        self.args = args
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        self.pos_edge_index = None
        self.neg_edge_index = None

        self.x = None

        self.conv1 = PolarGateConv(in_dim, out_dim // 2, first_aggr=True)

        self.convs = torch.nn.ModuleList()
        for _ in range(layer_num - 1):
            if args.model == 'PolarGate':
                self.convs.append(
                    restPolarGateConv(out_dim // 2, out_dim // 2, first_aggr=False,
                                      norm_emb=norm_emb))
            # elif args.model == 'PolarGate_Mamba':
            #     self.convs.append(
            #         restPolarGateConv_Mamba(out_dim // 2, out_dim // 2, first_aggr=False,
            #                      norm_emb=norm_emb))
            elif args.model == 'PolarGate_MetaLA':
                self.convs.append(
                    restPolarGateConv_MetaLA(out_dim // 2, out_dim // 2, first_aggr=False,norm_emb=norm_emb))
            else:
                raise NotImplementedError(f'Model {args.model} not implemented.')

        self.weight = torch.nn.Linear(self.out_dim, self.out_dim)
        self.readout_prob = MLP(self.out_dim, self.out_dim, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm',
                                act_layer='relu')

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.weight.reset_parameters()

    def get_x_edge_index(self, init_emb, edge_index_s):
        self.pos_edge_index = edge_index_s[edge_index_s[:, 2] > 0][:, :2].t()
        self.neg_edge_index = edge_index_s[edge_index_s[:, 2] < 0][:, :2].t()
        self.x = init_emb

    def forward(self, init_emb, edge_index_s, G) -> Tuple[Tensor, Tensor]:
        self.get_x_edge_index(init_emb, edge_index_s)
        z = torch.tanh(self.conv1(
            self.x, self.pos_edge_index, self.neg_edge_index))
        for conv in self.convs:
            if self.args.model == 'PolarGate':
                z = torch.tanh(conv(z, self.pos_edge_index, self.neg_edge_index))
            # elif self.args.model == 'PolarGate_Mamba':
            #     z = torch.tanh(conv(z, self.pos_edge_index, self.neg_edge_index, G))
            elif self.args.model == 'PolarGate_MetaLA':
                z = torch.tanh(conv(z, self.pos_edge_index, self.neg_edge_index, G))

        z = torch.tanh(self.weight(z))

        prob = self.readout_prob(z)
        prob = F.sigmoid(prob)

        return z, prob

class GCNSumConv(nn.Module):
    """Convolutional Sum layer from NeuroSAT (Selsam et al., ICLR 2019)"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.msg_linear = nn.Linear(in_dim, out_dim)
        self.update_linear = nn.Linear(in_dim + out_dim, out_dim)

    def reset_parameters(self):
        self.msg_linear.reset_parameters()
        self.update_linear.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
        """
        row, col = edge_index
        # Message passing: sum aggregation
        msg = self.msg_linear(x[row])
        aggr_msg = torch.zeros(x.size(0), self.out_dim, device=x.device)
        aggr_msg.index_add_(0, col, msg)

        # Update
        out = torch.cat([x, aggr_msg], dim=-1)
        out = self.update_linear(out)
        return out


class GCN_Sum(nn.Module):
    """GCN with Sum aggregation from NeuroSAT paper"""
    def __init__(
        self,
        args,
        device: torch.device,
        in_dim: int = 64,
        out_dim: int = 64,
        layer_num: int = 2,
        norm_emb: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.args = args
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.layer_num = layer_num

        # First layer
        self.conv1 = GCNSumConv(in_dim, out_dim)

        # Subsequent layers
        self.convs = nn.ModuleList()
        for _ in range(layer_num - 1):
            self.convs.append(GCNSumConv(out_dim, out_dim))

        self.weight = nn.Linear(self.out_dim, self.out_dim)
        self.readout_prob = MLP(self.out_dim, self.out_dim, 1, num_layer=3, p_drop=0.2,
                                norm_layer='batchnorm', act_layer='relu')

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.weight.reset_parameters()

    def forward(self, init_emb, edge_index_s, G) -> Tuple[Tensor, Tensor]:
        # Combine positive and negative edges into single edge_index
        pos_edges = edge_index_s[edge_index_s[:, 2] > 0][:, :2].t()
        neg_edges = edge_index_s[edge_index_s[:, 2] < 0][:, :2].t()
        edge_index = torch.cat([pos_edges, neg_edges], dim=1)

        z = torch.tanh(self.conv1(init_emb, edge_index))
        for conv in self.convs:
            z = torch.tanh(conv(z, edge_index))

        z = torch.tanh(self.weight(z))
        prob = self.readout_prob(z)
        prob = F.sigmoid(prob)

        return z, prob


class GCNAttnConv(nn.Module):
    """Graph Attention layer (Veličković et al., ICLR 2017; Thost & Chen, ICLR 2021)"""
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.linear_q = nn.Linear(in_dim, out_dim)
        self.linear_k = nn.Linear(in_dim, out_dim)
        self.linear_v = nn.Linear(in_dim, out_dim)
        self.attn_linear = nn.Linear(2 * self.head_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.linear_q.reset_parameters()
        self.linear_k.reset_parameters()
        self.linear_v.reset_parameters()
        self.attn_linear.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
        """
        num_nodes = x.size(0)
        row, col = edge_index

        # Linear transformations
        q = self.linear_q(x).view(num_nodes, self.num_heads, self.head_dim)
        k = self.linear_k(x).view(num_nodes, self.num_heads, self.head_dim)
        v = self.linear_v(x).view(num_nodes, self.num_heads, self.head_dim)

        # Compute attention scores
        q_i = q[col]  # [num_edges, num_heads, head_dim]
        k_j = k[row]  # [num_edges, num_heads, head_dim]
        v_j = v[row]  # [num_edges, num_heads, head_dim]

        # Concatenate and compute attention
        attn_input = torch.cat([q_i, k_j], dim=-1)  # [num_edges, num_heads, 2*head_dim]
        attn_scores = self.attn_linear(attn_input).squeeze(-1)  # [num_edges, num_heads]

        # Softmax per target node
        max_scores = torch.full((num_nodes, self.num_heads), -1e9, device=x.device)
        max_scores.scatter_reduce_(0, col.unsqueeze(-1).expand(-1, self.num_heads),
                                   attn_scores, reduce='amax', include_self=False)

        exp_scores = torch.exp(attn_scores - max_scores[col])
        exp_sum = torch.zeros(num_nodes, self.num_heads, device=x.device)
        exp_sum.index_add_(0, col, exp_scores)

        alpha = exp_scores / (exp_sum[col] + 1e-16)  # [num_edges, num_heads]
        alpha = self.dropout(alpha)

        # Aggregate messages
        msg = v_j * alpha.unsqueeze(-1)  # [num_edges, num_heads, head_dim]
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        out.index_add_(0, col, msg)

        out = out.view(num_nodes, -1)  # [num_nodes, out_dim]
        return out


class GCN_Attn(nn.Module):
    """GCN with Attention mechanism (GAT)"""
    def __init__(
        self,
        args,
        device: torch.device,
        in_dim: int = 64,
        out_dim: int = 64,
        layer_num: int = 2,
        norm_emb: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.args = args
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.layer_num = layer_num

        # First layer
        self.conv1 = GCNAttnConv(in_dim, out_dim)

        # Subsequent layers
        self.convs = nn.ModuleList()
        for _ in range(layer_num - 1):
            self.convs.append(GCNAttnConv(out_dim, out_dim))

        self.weight = nn.Linear(self.out_dim, self.out_dim)
        self.readout_prob = MLP(self.out_dim, self.out_dim, 1, num_layer=3, p_drop=0.2,
                                norm_layer='batchnorm', act_layer='relu')

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.weight.reset_parameters()

    def forward(self, init_emb, edge_index_s, G) -> Tuple[Tensor, Tensor]:
        # Combine positive and negative edges
        pos_edges = edge_index_s[edge_index_s[:, 2] > 0][:, :2].t()
        neg_edges = edge_index_s[edge_index_s[:, 2] < 0][:, :2].t()
        edge_index = torch.cat([pos_edges, neg_edges], dim=1)

        z = torch.tanh(self.conv1(init_emb, edge_index))
        for conv in self.convs:
            z = torch.tanh(conv(z, edge_index))

        z = torch.tanh(self.weight(z))
        prob = self.readout_prob(z)
        prob = F.sigmoid(prob)

        return z, prob


class GCNDeepSetConv(nn.Module):
    """DeepSet aggregation layer (Amizadeh et al., ICLR 2019)"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # DeepSet uses two MLPs: one for encoding, one for decoding
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def reset_parameters(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
        """
        row, col = edge_index

        # Encode messages
        msg = self.encoder(x[row])

        # Aggregate with sum (permutation invariant)
        aggr_msg = torch.zeros(x.size(0), self.out_dim, device=x.device)
        aggr_msg.index_add_(0, col, msg)

        # Decode aggregated messages
        out = self.decoder(aggr_msg)
        return out


class GCN_Deepset(nn.Module):
    """GCN with DeepSet aggregation"""
    def __init__(
        self,
        args,
        device: torch.device,
        in_dim: int = 64,
        out_dim: int = 64,
        layer_num: int = 2,
        norm_emb: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.args = args
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.layer_num = layer_num

        # First layer
        self.conv1 = GCNDeepSetConv(in_dim, out_dim)

        # Subsequent layers
        self.convs = nn.ModuleList()
        for _ in range(layer_num - 1):
            self.convs.append(GCNDeepSetConv(out_dim, out_dim))

        self.weight = nn.Linear(self.out_dim, self.out_dim)
        self.readout_prob = MLP(self.out_dim, self.out_dim, 1, num_layer=3, p_drop=0.2,
                                norm_layer='batchnorm', act_layer='relu')

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.weight.reset_parameters()

    def forward(self, init_emb, edge_index_s, G) -> Tuple[Tensor, Tensor]:
        # Combine positive and negative edges
        pos_edges = edge_index_s[edge_index_s[:, 2] > 0][:, :2].t()
        neg_edges = edge_index_s[edge_index_s[:, 2] < 0][:, :2].t()
        edge_index = torch.cat([pos_edges, neg_edges], dim=1)

        z = torch.tanh(self.conv1(init_emb, edge_index))
        for conv in self.convs:
            z = torch.tanh(conv(z, edge_index))

        z = torch.tanh(self.weight(z))
        prob = self.readout_prob(z)
        prob = F.sigmoid(prob)

        return z, prob


class GCNGatedSumConv(nn.Module):
    """Gated Sum layer (Zhang et al., NeurIPS 2019 - D-VAE)"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Message computation
        self.msg_linear = nn.Linear(in_dim, out_dim)

        # Gate computation
        self.gate_linear = nn.Linear(2 * in_dim, out_dim)

        # Update computation
        self.update_linear = nn.Linear(in_dim + out_dim, out_dim)

    def reset_parameters(self):
        self.msg_linear.reset_parameters()
        self.gate_linear.reset_parameters()
        self.update_linear.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
        """
        row, col = edge_index
        num_nodes = x.size(0)

        # Compute messages
        msg = self.msg_linear(x[row])

        # Compute gates
        gate_input = torch.cat([x[col], x[row]], dim=-1)
        gate = torch.sigmoid(self.gate_linear(gate_input))

        # Apply gates to messages
        gated_msg = gate * msg

        # Aggregate gated messages
        aggr_msg = torch.zeros(num_nodes, self.out_dim, device=x.device)
        aggr_msg.index_add_(0, col, gated_msg)

        # Update
        out = torch.cat([x, aggr_msg], dim=-1)
        out = self.update_linear(out)
        return out


class GCN_GatedSum(nn.Module):
    """GCN with Gated Sum aggregation (D-VAE)"""
    def __init__(
        self,
        args,
        device: torch.device,
        in_dim: int = 64,
        out_dim: int = 64,
        layer_num: int = 2,
        norm_emb: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.args = args
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.layer_num = layer_num

        # First layer
        self.conv1 = GCNGatedSumConv(in_dim, out_dim)

        # Subsequent layers
        self.convs = nn.ModuleList()
        for _ in range(layer_num - 1):
            self.convs.append(GCNGatedSumConv(out_dim, out_dim))

        self.weight = nn.Linear(self.out_dim, self.out_dim)
        self.readout_prob = MLP(self.out_dim, self.out_dim, 1, num_layer=3, p_drop=0.2,
                                norm_layer='batchnorm', act_layer='relu')

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.weight.reset_parameters()

    def forward(self, init_emb, edge_index_s, G) -> Tuple[Tensor, Tensor]:
        # Combine positive and negative edges
        pos_edges = edge_index_s[edge_index_s[:, 2] > 0][:, :2].t()
        neg_edges = edge_index_s[edge_index_s[:, 2] < 0][:, :2].t()
        edge_index = torch.cat([pos_edges, neg_edges], dim=1)

        z = torch.tanh(self.conv1(init_emb, edge_index))
        for conv in self.convs:
            z = torch.tanh(conv(z, edge_index))

        z = torch.tanh(self.weight(z))
        prob = self.readout_prob(z)
        prob = F.sigmoid(prob)

        return z, prob