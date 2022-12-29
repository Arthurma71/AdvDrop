from module.inv_loss import *
from utils import sparse_dense_mul
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import is_sparse, is_torch_sparse_tensor


class Mask_Model(nn.Module):
    def __init__(self, args, u_i_matrix, graph):
        super().__init__()
        # TODO: Modify M with Attention
        self.inv_loss = Inv_Loss(args)
        self.embed_size = args.embed_size
        self.embed_h = args.att_dim
        self.device = torch.device(args.cuda)
        self.graph = graph
        if args.mask == 1:
            self.gumble_tau = args.gumble_tau
            self.Q = nn.Linear(self.embed_size, self.embed_h)
            self.K = nn.Linear(self.embed_size, self.embed_h)
        if args.mask == 0:
            self.rand_var = torch.nn.Parameter(torch.ones_like(graph._values())).to(self.device)
        self.u_i_matrix = u_i_matrix
        # self.V = nn.Linear(embed_h, embed_h)

    def get_M_attention(self, u_i_matrix, user_embed, item_embed):
        Q = self.Q(user_embed)  # No.user x embedding
        K = self.K(item_embed)  # No.item x embedding
        weights = torch.matmul(Q, K.T)  # No.user x No.item
        gumble_G = torch.log(-torch.log(torch.rand(u_i_matrix._values().shape[0]).to(self.device)))
        gumble_G = torch.sparse_coo_tensor(u_i_matrix._indices(), gumble_G, u_i_matrix.size())
        # apply mask
        mask_weights = sparse_dense_mul(u_i_matrix, weights)
        g_mask_weights = (mask_weights - gumble_G) / self.gumble_tau
        weights_softmax = torch.sparse.softmax(g_mask_weights, dim=1)
        # weights_softmax = weights_exp / (torch.sparse.sum(weights_exp, dim=1) + 1e-5) #No. user * No.items
        return weights_softmax

    def mask_attention(self, user_embed, item_embed):
        user_num = user_embed.shape[0]
        item_num = item_embed.shape[0]
        user_pad = torch.sparse.FloatTensor(torch.Size([user_num, user_num])).to(self.device)
        item_pad = torch.sparse.FloatTensor(torch.Size([item_num, item_num])).to(self.device)
        # user as the query 
        M_ui = self.get_M_attention(self.u_i_matrix, user_embed, item_embed)
        M_ui = torch.cat([user_pad, M_ui], dim=1).coalesce()
        # item as the query 
        M_iu = self.get_M_attention(torch.transpose(self.u_i_matrix, 0, 1), item_embed, user_embed)
        M_iu = torch.cat([M_iu, item_pad], dim=1).coalesce()

        mask = torch.cat([M_ui, M_iu], dim=0).coalesce()
        mask = torch.sparse_coo_tensor(mask._indices(), torch.cat([M_ui.values(), M_iu.values()]), mask.size())
        return mask

    def mask_simple(self, user_embed, item_embed):
        # user_num = user_embed.shape[0]
        # item_num = item_embed.shape[0]
        # user_pad = torch.sparse.FloatTensor(torch.Size([user_num, user_num])).to(self.device)
        # item_pad = torch.sparse.FloatTensor(torch.Size([item_num, item_num])).to(self.device)

        # M_ui = torch.cat([user_pad, self.rand_var_sparse], dim=1)
        # M_iu = torch.cat([torch.transpose(self.rand_var_sparse, 0, 1), item_pad], dim=1)

        # mask = torch.cat([M_ui, M_iu], dim=0)
        mask = torch.sparse_coo_tensor(self.graph._indices(), self.rand_var, self.graph.size())
        # print("sparse var grad:",self.rand_var_sparse.requires_grad)
        # print("var grad:",self.rand_var.requires_grad)

        return mask


class Mask_Model_Geometric(MessagePassing):
    def __init__(self, args):
        super().__init__(aggr='add')
        self.embed_size = args.embed_size
        self.embed_h = args.att_dim
        self.Q = Linear(self.embed_size, self.embed_h)
        self.K = Linear(self.embed_size, self.embed_h)
        self.gumble_tau = args.gumble_tau
        self.device = torch.device(args.cuda)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x_norm = F.normalize(x, p=2., dim=-1)
        # propagate_type: (x: Tensor, x_norm: Tensor)
        return self.propagate(edge_index, x=x, x_norm = x_norm, norm = norm, size=None)

    def message(self, x_j: Tensor, x_norm_i: Tensor, x_norm_j: Tensor, norm,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # apply transformation layers
        Query = self.Q(x_norm_i)
        Keys = self.K(x_norm_j)
        # dot product of each query-key pair
        alpha = (Keys * Query).sum(dim=-1)
        # apply gumble
        gumble_G = torch.log(-torch.log(torch.rand(alpha.shape[0]).to(self.device)))
        alpha = (alpha - gumble_G) / self.gumble_tau
        # softmax
        alpha = softmax(alpha, index, ptr, size_i)

        return x_j * alpha.view(-1, 1) * norm.view(-1, 1)

class Mask_Model_Attention(MessagePassing):
    def __init__(self, args):
        super().__init__(aggr='add')
        self.embed_size = args.embed_size
        self.embed_h = args.att_dim
        self.Q = Linear(self.embed_size, self.embed_h)
        self.K = Linear(self.embed_size, self.embed_h)
        #self.W = Linear(2*self.embed_size, 1)
        self.gumble_tau = args.gumble_tau
        self.device = torch.device(args.cuda)
        self.args = args
    
    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        decomposed_layers = 1 if self.explain else self.decomposed_layers

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        size = self.__check_input__(edge_index, size)

        if decomposed_layers > 1:
                user_args = self.__user_args__
                decomp_args = {a[:-2] for a in user_args if a[-2:] == '_j'}
                decomp_kwargs = {
                    a: kwargs[a].chunk(decomposed_layers, -1)
                    for a in decomp_args
                }
                decomp_out = []

        for i in range(decomposed_layers):
            if decomposed_layers > 1:
                for arg in decomp_args:
                    kwargs[arg] = decomp_kwargs[arg][i]

            coll_dict = self.__collect__(self.__user_args__, edge_index,
                                            size, kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            for hook in self._message_forward_pre_hooks.values():
                res = hook(self, (msg_kwargs, ))
                if res is not None:
                    msg_kwargs = res[0] if isinstance(res, tuple) else res
            out = self.message(**msg_kwargs)

            for hook in self._message_forward_hooks.values():
                res = hook(self, (msg_kwargs, ), out)
                if res is not None:
                    out = res

            if self.explain:
                explain_msg_kwargs = self.inspector.distribute(
                    'explain_message', coll_dict)
                out = self.explain_message(out, **explain_msg_kwargs)

            if decomposed_layers > 1:
                    decomp_out.append(out)
        if decomposed_layers > 1:
                out = torch.cat(decomp_out, dim=-1)
        return out 

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        norm = deg[col]
        x_norm = F.normalize(x, p=2., dim=-1)
        # propagate_type: (x: Tensor, x_norm: Tensor)
        return self.propagate(edge_index, x=x, x_norm = x_norm,norm=norm, size=None)

    def message(self, x_j: Tensor, x_norm_i: Tensor, x_norm_j: Tensor,norm,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # apply transformation layers
        Query = self.Q(x_norm_i)
        Keys = self.K(x_norm_j)

        # alpha = torch.squeeze(self.W(torch.cat([x_norm_i,x_norm_j],dim=1)))
        # dot product of each query-key pair
        alpha = (Keys * Query).sum(dim=-1)
        # # apply gumble
        # gumble_G = torch.log(-torch.log(torch.rand(alpha.shape[0]).to(self.device)))
        # alpha = (alpha - gumble_G) / self.gumble_tau
        alpha = (alpha - scatter(alpha,index, dim=0, reduce='mean')[index])
        

        #alpha = (alpha - torch.mean(alpha))/torch.sqrt(torch.var(alpha))
        # softmax
        #alpha = softmax(alpha, index, ptr, size_i)
        #alpha = alpha*norm*self.args.keep_prob
        #return torch.clamp(alpha, min=0, max=1) 
        return torch.sigmoid(alpha) 


class GCN(MessagePassing):
    def __init__(self, args):
        super().__init__()
        self.n_layers = args.n_layers

    def forward(self, emd: Tensor, edge_index: Adj) -> Tensor:
        all_emd = [emd]
        for i in range(self.n_layers):
            emd = self.propagate(edge_index, x=emd)
            all_emd.append(emd)

        all_emd = torch.stack(all_emd, dim=1)
        emb_final = torch.mean(all_emd, dim=1)
        
        return emb_final

    def message(self, x_j: Tensor, norm) -> Tensor:
        # Constructs messages from node :math:`j` to node :math:`i`
        return x_j
    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)
