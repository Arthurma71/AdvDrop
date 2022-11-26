import torch.nn as nn
import torch
from module.inv_loss import *


class Mask_Model(nn.Module):
    def __init__(self, args, u_i_matrix, all_user_embed_m, all_item_embed_m):
        super().__init__(args, u_i_matrix, all_user_embed_m, all_item_embed_m)
        # TODO: Modify M with Attention
        self.M = self.get_M_attention(u_i_matrix, all_user_embed_m, all_item_embed_m)
        self.inv_loss = Inv_Loss()
        self.tau = args.mask_tau
        embed_h = all_user_embed_m.shape[1]
        self.Q = nn.Linear(embed_h, embed_h)
        self.K = nn.Linear(embed_h, embed_h)
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
        weights_exp = torch.exp(weights - weights_max)
        weights_exp = weights_exp * (u_i_matrix != 0).float()  # apply mask
        weights_softmax = weights_exp / (torch.sum(weights_exp, dim=1, keepdim=True) + 1e-5)

        return weights_softmax

    def mask(self, A):
        return self.M * A
