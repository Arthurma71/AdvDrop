import torch.nn as nn
import torch


class Inv_Loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def pearson_corr(self, rank1, rank2):
        numer = torch.sum((rank1-torch.mean(rank1, dim=1, keepdims=True))*(rank2-torch.mean(rank2, dim=1, keepdims=True)), dim=1)
        denom = torch.std(rank1, dim=1,unbiased=False)*torch.std(rank2, dim=1,unbiased=False)
        pearson = numer/(denom)
        return pearson/rank1.shape[1]
    
    def forward(self, all_items, all_items_m, all_users, all_users_m, users):
        num_all_items = len(all_items)
        selected_index = torch.randint(0, num_all_items, (self.args.num_samples,))
        user_embed = all_users[users]   # batch_size * emb_dim
        user_embed_m = all_users_m[users]

        ranking = torch.matmul(user_embed, all_items[selected_index].T)  # batch_size * num_samples
        ranking_m = torch.matmul(user_embed_m, all_items_m[selected_index].T)  # batch_size * num_samples

        pearson = self.pearson_corr(ranking, ranking_m)

        inv_loss = torch.mean(pearson)

        return inv_loss
    
   
