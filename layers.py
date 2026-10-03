# coding=utf-8
from typing import Union
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.typing import PairTensor, Adj
from torch_sparse import matmul
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from torch_sparse import coalesce
# from mamba_ssm import Mamba
from torch_geometric.utils import to_dense_batch
from metala import MetaLAConfig, MetaLADecoderLayer


_norm_layer_factory = {
    'batchnorm': nn.BatchNorm1d,
}

_act_layer_factory = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'sigmoid': nn.Sigmoid,
}


def create_spectral_features(
        pos_edge_index: torch.LongTensor,
        neg_edge_index: torch.LongTensor,
        node_num: int,
        dim: int
) -> torch.FloatTensor:
    edge_index = torch.cat(
        [pos_edge_index, neg_edge_index], dim=1)
    N = node_num
    edge_index = edge_index.to(torch.device('cpu'))

    pos_val = torch.full(
        (pos_edge_index.size(1),), 2, dtype=torch.float)
    neg_val = torch.full(
        (neg_edge_index.size(1),), 0, dtype=torch.float)
    val = torch.cat([pos_val, neg_val], dim=0)

    row, col = edge_index
    edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
    val = torch.cat([val, val], dim=0)

    edge_index, val = coalesce(edge_index, val, N, N)
    val = val - 1

    edge_index = edge_index.detach().numpy()
    val = val.detach().numpy()
    A = sp.coo_matrix((val, edge_index), shape=(N, N))
    svd = TruncatedSVD(n_components=dim, n_iter=128)
    svd.fit(A)
    x = svd.components_.T
    return torch.from_numpy(x).to(torch.float)


class MLP(nn.Module):
    def __init__(self, dim_in=256, dim_hidden=32, dim_pred=1, num_layer=3, norm_layer=None, act_layer=None, p_drop=0.5,
                 sigmoid=False, tanh=False):
        super(MLP, self).__init__()
        '''
        The basic structure is refered from 
        '''
        assert num_layer >= 2, 'The number of layers shoud be larger or equal to 2.'
        if norm_layer in _norm_layer_factory.keys():
            self.norm_layer = _norm_layer_factory[norm_layer]
        if act_layer in _act_layer_factory.keys():
            self.act_layer = _act_layer_factory[act_layer]
        if p_drop > 0:
            self.dropout = nn.Dropout

        fc = []
        # 1st layer
        fc.append(nn.Linear(dim_in, dim_hidden))
        if norm_layer:
            fc.append(self.norm_layer(dim_hidden))
        if act_layer:
            fc.append(self.act_layer(inplace=True))
        if p_drop > 0:
            fc.append(self.dropout(p_drop))
        for _ in range(num_layer - 2):
            fc.append(nn.Linear(dim_hidden, dim_hidden))
            if norm_layer:
                fc.append(self.norm_layer(dim_hidden))
            if act_layer:
                fc.append(self.act_layer(inplace=True))
            if p_drop > 0:
                fc.append(self.dropout(p_drop))
        # last layer
        fc.append(nn.Linear(dim_hidden, dim_pred))
        # sigmoid
        if sigmoid:
            fc.append(nn.Sigmoid())
        if tanh:
            fc.append(nn.Tanh())
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        out = self.fc(x)
        return out


class PolarGateConv(MessagePassing):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        first_aggr: bool,
        bias: bool = True,
        norm_emb: bool = False,
        **kwargs
    ):

        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.first_aggr = first_aggr
        self.norm_emb = norm_emb

        if first_aggr:
            self.lin_b = Linear(2 * in_dim, out_dim, bias)
            self.lin_u = Linear(2 * in_dim, out_dim, bias)
        else:
            self.lin_b = Linear(3 * in_dim, out_dim, bias)
            self.lin_u = Linear(3 * in_dim, out_dim, bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_b.reset_parameters()
        self.lin_u.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: Adj,
                neg_edge_index: Adj) -> Tensor:

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if self.first_aggr:
            out_b = self.propagate(pos_edge_index, x=x)
            out_b = self.lin_b(torch.cat([out_b, x[1]], dim=-1))

            out_u = self.propagate(neg_edge_index, x=x)
            out_u = self.lin_u(torch.cat([out_u, x[1]], dim=-1))
            out = torch.cat([out_b, out_u], dim=-1)
        else:
            F_in = self.in_dim
            out_b1 = self.propagate(pos_edge_index, x=(
                x[0][..., :F_in], x[1][..., :F_in]))
            out_b2 = self.propagate(neg_edge_index, x=(
                x[0][..., F_in:], x[1][..., F_in:]))
            out_b = torch.cat([out_b1, out_b2, x[1][..., :F_in]], dim=-1)
            out_b = self.lin_b(out_b)

            out_u1 = self.propagate(pos_edge_index, x=(
                x[0][..., F_in:], x[1][..., F_in:]))
            out_u2 = self.propagate(neg_edge_index, x=(
                x[0][..., :F_in], x[1][..., :F_in]))
            out_u = torch.cat([out_u1, out_u2, x[1][..., F_in:]], dim=-1)
            out_u = self.lin_u(out_u)

            out = torch.cat([out_b, out_u], dim=-1)
        if self.norm_emb:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: PairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.out_dim}, first_aggr={self.first_aggr})')


class restPolarGateConv(MessagePassing):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        first_aggr: bool = False,
        bias: bool = True,
        norm_emb: bool = False,
        **kwargs
    ):

        kwargs.setdefault('aggr', 'min')
        # kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.first_aggr = first_aggr
        self.norm_emb = norm_emb

        self.lin_b = Linear(3 * in_dim, out_dim, bias)
        self.lin_u = Linear(3 * in_dim, out_dim, bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_b.reset_parameters()
        self.lin_u.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: Adj,
                neg_edge_index: Adj) -> Tensor:

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        F_in = self.in_dim

        # update positive embeddings
        out_b1 = self.propagate(pos_edge_index, x=(
            x[0][..., :F_in], x[1][..., :F_in]))
        out_b2 = self.propagate(neg_edge_index, x=(
            x[0][..., F_in:], x[1][..., F_in:]))
        out_b = torch.cat([out_b1, out_b2 * -1, x[1][..., :F_in]], dim=-1)
        out_b = self.lin_b(out_b)

        # update negative embeddings
        out_u1 = self.propagate(pos_edge_index, x=(
            x[0][..., F_in:], x[1][..., F_in:]))
        out_u2 = self.propagate(neg_edge_index, x=(
            x[0][..., :F_in], x[1][..., :F_in]))
        out_u = torch.cat([out_u1, out_u2 * -1, x[1][..., F_in:]], dim=-1)
        out_u = self.lin_u(out_u)

        out = torch.cat([out_b, out_u], dim=-1)
        if self.norm_emb:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: PairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.out_dim}, first_aggr={self.first_aggr})')


# class restPolarGateConv_Mamba(MessagePassing):
#     def __init__(
#         self,
#         in_dim: int,
#         out_dim: int,
#         first_aggr: bool = False,
#         bias: bool = True,
#         norm_emb: bool = False,
#         dropout: float = 0.5,
#         **kwargs
#     ):

#         kwargs.setdefault('aggr', 'min')
#         # kwargs.setdefault('aggr', 'mean')
#         super().__init__(**kwargs)

#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.first_aggr = first_aggr
#         self.norm_emb = norm_emb

#         self.lin_b = Linear(3 * in_dim, out_dim, bias)
#         self.lin_u = Linear(3 * in_dim, out_dim, bias)

#         self.norm1_local = nn.BatchNorm1d(out_dim * 2)
#         self.dropout_local = nn.Dropout(dropout)

#         # Mamba
#         self.self_attn0 = Mamba(d_model=out_dim*2, # Model dimension d_model
#             d_state=8,  # SSM state expansion factor
#             d_conv=4,    # Local convolution width
#             expand=1,    # Block expansion factor
#             )

#         self.linman = Linear(6*in_dim,2*out_dim , bias)

#         self.dropout_attn0 = nn.Dropout(dropout)
#         self.norm1_attn = nn.BatchNorm1d(out_dim * 2)
        
#         self.norm2 = nn.BatchNorm1d(out_dim * 2)

#         # Feed Forward block.
#         self.ff_dropout1 = nn.Dropout(dropout)
#         self.ff_dropout2 = nn.Dropout(dropout)
#         self.activation = F.relu
#         self.ff_linear1 = nn.Linear(out_dim * 2, out_dim * 2)
#         self.ff_linear2 = nn.Linear(out_dim * 2, out_dim * 2)

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin_b.reset_parameters()
#         self.lin_u.reset_parameters()

#     def _ff_block(self, x):
#         """Feed Forward block"""
#         x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
#         return self.ff_dropout2(self.ff_linear2(x))

#     def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: Adj,
#                 neg_edge_index: Adj, G) -> Tensor:
        
#         xff = x.clone()

#         if isinstance(x, Tensor):
#             x: PairTensor = (x, x)

#         F_in = self.in_dim

#         # update positive embeddings
#         out_b1 = self.propagate(pos_edge_index, x=(
#             x[0][..., :F_in], x[1][..., :F_in]))
#         out_b2 = self.propagate(neg_edge_index, x=(
#             x[0][..., F_in:], x[1][..., F_in:]))
#         out_b = torch.cat([out_b1, out_b2 * -1, x[1][..., :F_in]], dim=-1)
#         out_b = self.lin_b(out_b)

#         # update negative embeddings
#         out_u1 = self.propagate(pos_edge_index, x=(
#             x[0][..., F_in:], x[1][..., F_in:]))
#         out_u2 = self.propagate(neg_edge_index, x=(
#             x[0][..., :F_in], x[1][..., :F_in]))
#         out_u = torch.cat([out_u1, out_u2 * -1, x[1][..., F_in:]], dim=-1)
#         out_u = self.lin_u(out_u)

#         out = torch.cat([out_b, out_u], dim=-1)
#         if self.norm_emb:
#             out = F.normalize(out, p=2, dim=-1)

#         out = self.dropout_local(out)
#         out = xff + out # Residual connection.
#         out = self.norm1_local(out)

#         # Preprocess_Mamba
#         h_in1 = out
#         h_in2 = h_in1[G.ID_seq[0]]
#         h_in2[~G.ID_seq_mask[0]] = 0
#         h_in3 = h_in1[G.ID_seq[1]]
#         h_in3[~G.ID_seq_mask[1]] = 0

#         h_dense1, mask = to_dense_batch(h_in1, G.batch)
#         h_dense2, mask = to_dense_batch(h_in2, G.batch)
#         h_dense3, mask = to_dense_batch(h_in3, G.batch)
#         h_dense_list = [h_dense1, h_dense2, h_dense3]
        
#         # Mamba
#         h_attn_list = []
#         for i in range(3):
#             h_dense = h_dense_list[i]
#             h_attn = self.self_attn0(h_dense)[mask]
#             h_attn = self.dropout_attn0(h_attn)
#             h_attn_list.append(h_attn)

#         # h_attn = sum(h_attn_list) / len(h_attn_list)

#         h_attn = self.linman(torch.cat(h_attn_list, dim = -1))

#         h_attn = xff + h_attn  # Residual connection.
#         h_attn = self.norm1_attn(h_attn)

#         # Combine
#         out = out + h_attn
        
#         # Feed Forward block.
#         out = out + self._ff_block(out)
#         out = self.norm2(out)

#         return out

#     def message(self, x_j: Tensor) -> Tensor:
#         return x_j

#     def message_and_aggregate(self, adj_t: SparseTensor, x: PairTensor) -> Tensor:
#         adj_t = adj_t.set_value(None, layout=None)
#         return matmul(adj_t, x[0], reduce=self.aggr)

#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.in_dim}, '
#                 f'{self.out_dim}, first_aggr={self.first_aggr})')

class restPolarGateConv_MetaLA(MessagePassing):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        first_aggr: bool = False,
        bias: bool = True,
        norm_emb: bool = False,
        dropout: float = 0.0,
        **kwargs
    ):

        kwargs.setdefault('aggr', 'min')
        # kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.first_aggr = first_aggr
        self.norm_emb = norm_emb

        self.lin_b = Linear(3 * in_dim, out_dim, bias)
        self.lin_u = Linear(3 * in_dim, out_dim, bias)

        self.norm1_local = nn.BatchNorm1d(out_dim * 2)
        self.dropout_local = nn.Dropout(dropout)

        # MetaLA
        self.mate_config = MetaLAConfig()
        self.mate_config.decoder_embed_dim = out_dim*2
        
        self.self_attn0 = MetaLADecoderLayer(self.mate_config, 0)

        self.linman = Linear(6*out_dim, 2*out_dim , bias)
        # self.linman = Linear(2*out_dim, 2*out_dim , bias)

        self.dropout_attn0 = nn.Dropout(dropout)
        self.norm1_attn = nn.BatchNorm1d(out_dim * 2)
        
        self.norm2 = nn.BatchNorm1d(out_dim * 2)

        # Feed Forward block.
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(out_dim * 2, out_dim * 2)
        self.ff_linear2 = nn.Linear(out_dim * 2, out_dim * 2)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_b.reset_parameters()
        self.lin_u.reset_parameters()

    def _ff_block(self, x):
        """Feed Forward block"""
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: Adj,
                neg_edge_index: Adj, G) -> Tensor:

        xff = x.clone()

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        F_in = self.in_dim

        # update positive embeddings
        out_b1 = self.propagate(pos_edge_index, x=(
            x[0][..., :F_in], x[1][..., :F_in]))
        out_b2 = self.propagate(neg_edge_index, x=(
            x[0][..., F_in:], x[1][..., F_in:]))
        out_b = torch.cat([out_b1, out_b2 * -1, x[1][..., :F_in]], dim=-1)
        out_b = self.lin_b(out_b)

        # update negative embeddings
        out_u1 = self.propagate(pos_edge_index, x=(
            x[0][..., F_in:], x[1][..., F_in:]))
        out_u2 = self.propagate(neg_edge_index, x=(
            x[0][..., :F_in], x[1][..., :F_in]))
        out_u = torch.cat([out_u1, out_u2 * -1, x[1][..., F_in:]], dim=-1)
        out_u = self.lin_u(out_u)

        out = torch.cat([out_b, out_u], dim=-1)
        if self.norm_emb:
            out = F.normalize(out, p=2, dim=-1)

        out = self.dropout_local(out)
        out = xff + out # Residual connection.
        out = self.norm1_local(out)

        # Preprocess_MetaLA
        h_in1 = out
        h_in2 = h_in1[G.ID_seq[0]]
        h_in2[~G.ID_seq_mask[0]] = 0
        h_in3 = h_in1[G.ID_seq[1]]
        h_in3[~G.ID_seq_mask[1]] = 0

        h_dense1, mask = to_dense_batch(h_in1, G.batch)
        h_dense2, mask = to_dense_batch(h_in2, G.batch)
        h_dense3, mask = to_dense_batch(h_in3, G.batch)
        h_dense_list = [h_dense1, h_dense2, h_dense3]

        # h_dense_list = [h_dense1]

        # MetaLA
        h_attn_list = []
        for i in range(len(h_dense_list)):
            h_dense = h_dense_list[i]
            h_dense = h_dense.transpose(1, 0)
            h_attn = self.self_attn0(h_dense)[0]
            h_attn = h_attn.transpose(1, 0)[mask]
            h_attn = self.dropout_attn0(h_attn)
            h_attn_list.append(h_attn)

        # h_attn = sum(h_attn_list) / len(h_attn_list)

        h_attn = self.linman(torch.cat(h_attn_list, dim = -1))

        h_attn = xff + h_attn  # Residual connection.
        h_attn = self.norm1_attn(h_attn)

        # Combine
        out = out + h_attn
        
        # Feed Forward block.
        out = out + self._ff_block(out)
        out = self.norm2(out)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: PairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_dim}, '
                f'{self.out_dim}, first_aggr={self.first_aggr})')