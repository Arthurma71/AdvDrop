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
    
   
class Inv_Loss_Embed(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.align_W=nn.Linear(args.embed_size,args,embed_size)
    
    def forward(self, all_users, all_items):
        num_items = len(all_items)
        num_users = len(all_users)
        item_index = torch.randint(0, num_items, (self.args.num_samples,))
        selected_items=all_items[item_index]
        user_index = torch.randint(0, num_users, (self.args.num_samples,))
        selected_users=all_users[user_index]

        inv_loss=0
        losses=[]
        #ratings = torch.matmul(selected_users, torch.transpose(self.align_W(selected_users), 0, 1))
        for embed in [selected_users, selected_items]:
            ratings = torch.matmul(embed, torch.transpose(embed, 0, 1))
            ratings_diag = torch.diag(ratings)
            numerator = torch.exp(ratings_diag / self.tau)
            denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
            ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))
            inv_loss = inv_loss + ssm_loss
            losses.append(ssm_loss)
        
        
        return inv_loss, losses
    
   
