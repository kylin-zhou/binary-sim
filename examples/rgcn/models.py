from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing, RGCNConv
from torch_geometric.nn.norm import GraphNorm, MessageNorm
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter
from torch_sparse import SparseTensor, masked_select_nnz, matmul

from inits import glorot, zeros

try:
    from pyg_lib.ops import segment_matmul  # noqa

    _WITH_PYG_LIB = True
except ImportError:
    _WITH_PYG_LIB = False

    def segment_matmul(inputs: Tensor, ptr: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (Tensor, Tensor) -> Tensor
    pass


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (SparseTensor, Tensor) -> SparseTensor
    pass


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    else:
        return masked_select_nnz(edge_index, edge_mask, layout="coo")


class RGINConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    .. note::
        This implementation is as memory-efficient as possible by iterating
        over each individual relation type.
        Therefore, it may result in low GPU utilization in case the graph has a
        large number of relations.
        As an alternative approach, :class:`FastRGCNConv` does not iterate over
        each individual type, but may consume a large amount of memory to
        compensate.
        We advise to check out both implementations to see which one fits your
        needs.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
            In case no input features are given, this argument should
            correspond to the number of nodes in your graph.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int, optional): If set, this layer will use the
            basis-decomposition regularization scheme where :obj:`num_bases`
            denotes the number of bases to use. (default: :obj:`None`)
        num_blocks (int, optional): If set, this layer will use the
            block-diagonal-decomposition regularization scheme where
            :obj:`num_blocks` denotes the number of blocks to use.
            (default: :obj:`None`)
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`edge_index` is sorted by :obj:`edge_type`. This avoids
            internal re-sorting of the data and can improve runtime and memory
            efficiency. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        aggr: str = "mean",
        root_weight: bool = True,
        is_sorted: bool = False,
        bias: bool = True,
        eps: float = 0.0,
        train_eps: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("aggr", aggr)
        super().__init__(node_dim=0, **kwargs)
        self._WITH_PYG_LIB = torch.cuda.is_available() and _WITH_PYG_LIB

        if num_bases is not None and num_blocks is not None:
            raise ValueError(
                "Can not apply both basis-decomposition and "
                "block-diagonal-decomposition at the same time."
            )
        self.nn = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
        )
        # self.nn = torch.nn.Linear(out_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.is_sorted = is_sorted

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        if num_bases is not None:
            self.weight = Parameter(
                torch.Tensor(num_bases, in_channels[0], out_channels)
            )
            self.comp = Parameter(torch.Tensor(num_relations, num_bases))

        elif num_blocks is not None:
            assert in_channels[0] % num_blocks == 0 and out_channels % num_blocks == 0
            self.weight = Parameter(
                torch.Tensor(
                    num_relations,
                    num_blocks,
                    in_channels[0] // num_blocks,
                    out_channels // num_blocks,
                )
            )
            self.register_parameter("comp", None)

        else:
            self.weight = Parameter(
                torch.Tensor(num_relations, in_channels[0], out_channels)
            )
            self.register_parameter("comp", None)

        if root_weight:
            self.root = Param(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter("root", None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)
        self.eps.data.fill_(self.initial_eps)

    def forward(
        self,
        x: Union[OptTensor, Tuple[OptTensor, Tensor]],
        edge_index: Adj,
        edge_type: OptTensor = None,
    ):
        r"""
        Args:
            x: The input node features. Can be either a :obj:`[num_nodes,
                in_channels]` node feature matrix, or an optional
                one-dimensional node index tensor (in which case input features
                are treated as trainable node embeddings).
                Furthermore, :obj:`x` can be of type :obj:`tuple` denoting
                source and destination node features.
            edge_index (LongTensor or SparseTensor): The edge indices.
            edge_type: The one-dimensional relation type/index for each edge in
                :obj:`edge_index`.
                Should be only :obj:`None` in case :obj:`edge_index` is of type
                :class:`torch_sparse.tensor.SparseTensor`.
                (default: :obj:`None`)
        """

        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # propagate_type: (x: Tensor, edge_type_ptr: OptTensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels
            )

        if self.num_blocks is not None:  # Block-diagonal-decomposition =====

            if x_l.dtype == torch.long and self.num_blocks is not None:
                raise ValueError(
                    "Block-diagonal decomposition not supported "
                    "for non-continuous input features."
                )

            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                h = self.propagate(tmp, x=x_l, edge_type_ptr=None, size=size)
                h = h.view(-1, weight.size(1), weight.size(2))
                h = torch.einsum("abc,bcd->abd", h, weight[i])
                out += h.contiguous().view(-1, self.out_channels)

        else:  # No regularization/Basis-decomposition ========================
            if self._WITH_PYG_LIB and isinstance(edge_index, Tensor):
                if not self.is_sorted:
                    if (edge_type[1:] < edge_type[:-1]).any():
                        edge_type, perm = edge_type.sort()
                        edge_index = edge_index[:, perm]
                edge_type_ptr = torch.ops.torch_sparse.ind2ptr(
                    edge_type, self.num_relations
                )
                out = self.propagate(
                    edge_index, x=x_l, edge_type_ptr=edge_type_ptr, size=size
                )
            else:
                for i in range(self.num_relations):
                    tmp = masked_edge_index(edge_index, edge_type == i)

                    if x_l.dtype == torch.long:
                        out += self.propagate(
                            tmp, x=weight[i, x_l], edge_type_ptr=None, size=size
                        )
                    else:
                        h = self.propagate(tmp, x=x_l, edge_type_ptr=None, size=size)
                        out = out + (h @ weight[i])

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        if x_r is not None and x_r.shape[1] == out.shape[1]:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_type_ptr: OptTensor) -> Tensor:
        if edge_type_ptr is not None:
            return segment_matmul(x_j, edge_type_ptr, self.weight)

        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, num_relations={self.num_relations})"
        )


class RGIN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_relations,
        num_bases=None,
        num_blocks=None,
        n_layers=2,
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        # self.norm = GraphNorm(hidden_channels)
        # self.norm = nn.BatchNorm1d(hidden_channels)
        # self.norm = nn.InstanceNorm1d(hidden_channels)
        # self.norm2 = nn.InstanceNorm1d(hidden_channels)
        self.convs.append(
            RGINConv(
                in_channels,
                hidden_channels,
                num_relations,
                num_blocks=num_blocks,
                aggr="max",
            )
        )
        for i in range(n_layers - 2):
            self.convs.append(
                RGINConv(
                    hidden_channels,
                    hidden_channels,
                    num_relations,
                    num_blocks=num_blocks,
                    aggr="max",
                )
            )
        self.convs.append(
            RGINConv(
                hidden_channels,
                out_channels,
                num_relations,
                num_bases=num_bases,
                aggr="max",
            )
        )

    def forward(self, x, edge_index, edge_type):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                # if i == 0:
                #     x = self.norm(x)
                # if i == 1:
                #     x = self.norm2(x)
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x
