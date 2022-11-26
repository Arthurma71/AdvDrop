import torch.nn as nn
import torch


class Inv_Loss(nn.Module):
    def __init__(self, args):
        super().__init__(args)
    
    def pearson_corr(self, rank1, rank2):
        numer = torch.sum((rank1-torch.mean(rank1,dim=1,keepdims=True))*(rank2-torch.mean(rank1,dim=1,keepdims=True)),dim=1)
        denom = torch.var(rank1,dim=1)*torch.var(rank2,dim=1)
        pearson = numer/denom
        return pearson
    
    def forward(self, all_items, all_items_m, all_users, all_users_m, users):

        user_embed=all_users[users]   # batch_size * emb_dim
        user_embed_m=all_users_m[users]

        ranking = torch.matmul(user_embed, all_items.T)  # batch_size * num_items
        ranking_m = torch.matmul(user_embed_m, all_items_m.T)  # batch_size * num_items

        pearson = self.pearson_corr(ranking, ranking_m)

        inv_loss=torch.mean(pearson)

        return inv_loss
    
   
