import torch.nn as nn
import torch
from module.inv_loss import *
from utils import sparse_dense_mul


class Mask_Model(nn.Module):
    def __init__(self, args, u_i_matrix,graph):
        super().__init__()
        # TODO: Modify M with Attention
        self.inv_loss = Inv_Loss(args)
        self.embed_size = args.embed_size
        self.embed_h = args.att_dim
        self.device = torch.device(args.cuda)
        self.graph = graph
        if args.mask == 0:
            self.gumble_tau = args.gumble_tau
            self.Q = nn.Linear(self.embed_size, self.embed_h)
            self.K = nn.Linear(self.embed_size, self.embed_h)
        if args.mask == 1:
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

        M_ui = self.get_M_attention(self.u_i_matrix, user_embed, item_embed)
        M_ui = torch.cat([user_pad, M_ui], dim=1)
        M_iu = self.get_M_attention(torch.transpose(self.u_i_matrix, 0, 1), item_embed, user_embed)
        M_iu = torch.cat([M_iu, item_pad], dim=1)

        mask = torch.cat([M_ui, M_iu], dim=0)
        mask = torch.sparse_coo_tensor(mask._indices(),torch.cat[M_ui.values(),M_iu.values()],mask.size())
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
        #print("sparse var grad:",self.rand_var_sparse.requires_grad)
        #print("var grad:",self.rand_var.requires_grad)

        return mask
