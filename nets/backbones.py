import dgl
import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
import torch.nn.functional as F
from dgl.utils import check_eq_shape, expand_as_pair
from dgl.base import DGLError

from dgl.dataloading import (
	DataLoader,
	MultiLayerFullNeighborSampler,
	NeighborSampler,
	BlockSampler
)


class NodeApplyModule(nn.Module):
    # Update node feature h_v with (Wh_v+b)
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, node):
        h = self.linear(node.data['h'])  
        return {'h': h}



class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout, bn, residual, out_mlp, dgl_builtin = True, **kwargs):
        super().__init__()
        self.in_dim      = in_dim
        self.out_dim     = out_dim
        self.bn          = bn
        self.residual    = residual
        self.dgl_builtin = dgl_builtin
        if in_dim != out_dim:
            self.residual = False
        self.batchnorm   = nn.BatchNorm1d(out_dim)
        self.activation  = activation
        self.dropout     = nn.Dropout(dropout)
        if not self.dgl_builtin:
            self.apply_mod = NodeApplyModule(in_dim, out_dim)
        self.conv = GraphConv(in_dim, out_dim, allow_zero_in_degree=True)
        self.out_mlp = out_mlp
        if self.out_mlp:
            self.out_lin = NodeApplyModule(in_dim=out_dim, out_dim=out_dim)


    def forward(self, g, feature, last_layer = False):
        h_in = feature
        if not self.dgl_builtin:
            g.ndata['h'] = feature
            msg_fn = fn.copy_u('h', 'm')  # send feature of source nodes into mailbox
            reduce = fn.mean('m', 'h')    # compute the mean of source nodes in the mailbox and 赋值给节点的'h'属性
            g.update_all(msg_fn, reduce)
            g.apply_nodes(func = self.apply_mod) 
            h = g.ndata['h']
        else:
            h = self.conv(g, feature)
            h = self.out_lin(h) if self.out_mlp else h
        
        if self.bn:
            h = self.batchnorm(h)
        
        if self.activation is not None:
            h = self.activation(h)
        
        if self.residual:
            h = h_in + h
        
        h = self.dropout(h)
        return h

            
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout, bn, residual=False, activation=F.elu, **kwargs):
        super().__init__()
        self.residual   = residual
        self.activation = activation
        self.bn         = bn
        if in_dim != (out_dim*num_heads):
            self.residual = False
        self.gatconv = GATConv(in_dim, out_dim, num_heads, dropout, dropout, allow_zero_in_degree=True)
        if self.bn:
            self.batchnorm = nn.BatchNorm1d(out_dim * num_heads)
    
    def forward(self, g, h, last_layer = False):
        h_in = h
        h = self.gatconv(g, h)
        h = h.flatten(1) if not last_layer else h.mean(1)

        if self.bn:
            h = self.batchnorm(h)
        
        if self.activation is not None:
            h = self.activation(h)
        
        if self.residual:
            h = h_in  + h
        
        return h


class SAGELayer(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
    ):
        super(SAGELayer, self).__init__()
        valid_aggre_types = {"mean", "gcn", "pool", "lstm"}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                "Invalid aggregator_type. Must be one of {}. "
                "But got {!r} instead.".format(
                    valid_aggre_types, aggregator_type
                )
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                self._in_src_feats, self._in_src_feats, batch_first=True
            )

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)

        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The LSTM module is using xavier initialization method for its weights.
        """
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == "lstm":
            self.lstm.reset_parameters()
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox["m"]  # (B, L, D)
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats)),
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(0)}

    def forward(self, graph, feat, edge_weight=None, last_layer = False):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == "mean":
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == "gcn":
                check_eq_shape(feat)
                graph.srcdata["h"] = (
                    self.fc_neigh(feat_src) if lin_before_mp else feat_src
                )
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata["h"] = (
                        self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                    )
                else:
                    if graph.is_block:
                        graph.dstdata["h"] = graph.srcdata["h"][
                            : graph.num_dst_nodes()
                        ]
                    else:
                        graph.dstdata["h"] = graph.srcdata["h"]
                graph.update_all(msg_fn, fn.sum("m", "neigh"))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (
                    degs.unsqueeze(-1) + 1
                )
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == "pool":
                graph.srcdata["h"] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max("m", "neigh"))
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            elif self._aggre_type == "lstm":
                graph.srcdata["h"] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata["neigh"])
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == "gcn":
                rst = h_neigh
                # add bias manually for GCN
                if self.bias is not None:
                    rst = rst + self.bias
            else:
                rst = self.fc_self(h_self) + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst
