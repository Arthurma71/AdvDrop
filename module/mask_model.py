
import torch.nn as nn
import torch
from module.inv_loss import *


class Mask_Model(nn.Module):
    def __init__(self, args):
        super().__init__(args)
        # TODO: Modify M with Attention
        self.M = nn.Linear()
        self.inv_loss = Inv_Loss()
        self.tau=args.mask_tau


    def forward(self, all_users, all_items, all_users_m, all_items_m, users, pos_items, neg_items):
        inv_loss=self.inv_loss(all_users, all_items, all_users_m, all_items_m)

        users_emb = all_users_m[users]
        pos_emb = all_items_m[pos_items]
        neg_emb = all_items_m[neg_items]

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)


        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)
        mf_loss_m = torch.negative(torch.mean(maxi))

        tot_loss = -inv_loss + self.tau*mf_loss_m

        return tot_loss

    def mask(self, A):
        return self.M*A