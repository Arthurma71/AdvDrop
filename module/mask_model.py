import torch.nn as nn
import torch
from module.inv_loss import *
from utils import sparse_dense_mul


class Mask_Model(nn.Module):
    def __init__(self, args, u_i_matrix):
        super().__init__(args, u_i_matrix)
        # TODO: Modify M with Attention
        self.inv_loss = Inv_Loss()
        self.tau = args.mask_tau
        self.embed_size = args.embed_size
        self.embed_h = args.att_dim
        self.gumble_tau = args.gumble_tau
        self.Q = nn.Linear(self.embed_size, self.embed_h)
        self.K = nn.Linear(self.embed_size, self.embed_h)
        self.u_i_matrix = u_i_matrix
        # self.V = nn.Linear(embed_h, embed_h)

    def forward(self, all_users, all_items, all_users_m, all_items_m, users, pos_items, neg_items):
        inv_loss = self.inv_loss(all_users, all_items, all_users_m, all_items_m)

        users_emb = all_users_m[users]
        pos_emb = all_items_m[pos_items]
        neg_emb = all_items_m[neg_items]

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)
        mf_loss_m = torch.negative(torch.mean(maxi))

        tot_loss = -inv_loss + self.tau * mf_loss_m

        return tot_loss
    


    def get_M_attention(self, u_i_matrix, user_embed, item_embed):
        # assume graph is the user-positive item matrix
        Q = self.Q(user_embed)  # No.user x embedding
        K = self.K(item_embed)  # No.item x embedding

        weights = torch.matmul(Q, K.T)  # No.user x No.item
        weights_max = torch.max(weights, dim=1, keepdim=True)[0]
        #weights_exp = torch.exp(weights - weights_max)

        gumble_G = torch.log(-torch.log(torch.rand(u_i_matrix._values().shape[0]).to(self.device)))
        gumble_G = torch.sparse_coo_tensor(u_i_matrix._indices(),gumble_G,u_i_matrix.size())
        # apply mask
        mask_weights = sparse_dense_mul(u_i_matrix, weights) 
        g_mask_weights = (mask_weights - gumble_G)/self.gumble_tau
        weights_softmax = torch.sparse.softmax(g_mask_weights, axis=1)
        #weights_softmax = weights_exp / (torch.sparse.sum(weights_exp, dim=1) + 1e-5) #No. user * No.items
        return weights_softmax

    def mask(self, user_embed, item_embed):
        user_num=user_embed.shape[0]
        item_num=item_embed.shape[0]
        user_pad=torch.sparse.FloatTensor(torch.Size([user_num,user_num]))
        item_pad=torch.sparse.FloatTensor(torch.Size([item_num,item_num]))

        M_ui = self.get_M_attention(self.u_i_matrix, user_embed, item_embed)
        M_ui = torch.cat([user_pad,M_ui],dim=1)
        M_iu = self.get_M_attention(self.u_i_matrix.T,item_embed, user_embed)
        M_iu = torch.cat([M_iu,item_pad],dim=1)

        mask = torch.cat([M_ui,M_iu],dim=0)
        return mask


    #def mask_simple(self):

