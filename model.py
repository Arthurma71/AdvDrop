from cmath import cos
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import torch_sparse
from module.mask_model import *
from module.inv_loss import *
from torch_geometric.nn.conv import LGConv
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter
from torch.nn import ModuleList
from module.fairness import *
import networkx as nx

def gini_index(p, device):
    n = p.shape[0]
    p, indices = torch.sort(p)
    k = (n+1) - torch.arange(1,n+1).to(device)
    numerator = torch.sum(k*p)*2
    denomitor = n * torch.sum(p)
    return (n+1)/n - (numerator/denomitor)

class MF(nn.Module):
    def __init__(self, args, data):
        super(MF, self).__init__()
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.decay = args.regs
        self.device = torch.device(args.cuda)
        self.saveID = args.saveID

        self.train_user_list = data.train_user_list
        self.valid_user_list = data.valid_user_list
        # = torch.tensor(data.population_list).cuda(self.device)
        self.user_pop = torch.tensor(data.user_pop_idx).type(torch.LongTensor).cuda(self.device)
        self.item_pop = torch.tensor(data.item_pop_idx).type(torch.LongTensor).cuda(self.device)
        self.user_pop_max = data.user_pop_max
        self.item_pop_max = data.item_pop_max

        self.embed_user = nn.Embedding(self.n_users, self.emb_dim)
        self.embed_item = nn.Embedding(self.n_items, self.emb_dim)

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)

    # Prediction function used when evaluation
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        users = self.embed_user(torch.tensor(users).cuda(self.device))
        items = torch.transpose(self.embed_item(torch.tensor(items).cuda(self.device)), 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()


class LGN(MF):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.Graph = data.getSparseGraph().cuda(self.device)
        self.n_layers = args.n_layers
        self.args = args
        self.user_tags=[]
        self.item_tags=[]
        
        if 'ml' in args.dataset:
            self.user_tags = data.get_user_tags()
        
        if 'coat' in args.dataset:
            self.user_tags = data.get_user_tags()
            self.item_tags = data.get_item_tags()

        if self.args.use_attribute:
            self.user_feature_embed = []
            self.item_feature_embed = []
            self.generate_embedings(self.user_tags, self.user_feature_embed)
            self.generate_embedings(self.item_tags, self.item_feature_embed)
            self.user_dense = nn.Linear(self.emb_dim* (len(self.user_tags)+1) ,self.emb_dim)
            self.item_dense = nn.Linear(self.emb_dim*(len(self.item_tags)+1),self.emb_dim)
        
    def generate_embedings(self, tags, feature_embed):
        featuren_len = len(tags) 
        if featuren_len > 0:
            for i in range(featuren_len):
                max_value = torch.max(tags[i])+1
                embed = nn.Embedding(max_value, self.emb_dim).to(self.device)
                nn.init.xavier_normal_(embed.weight)
                feature_embed.append(embed)
                # feature_embed[i] = embed

    def concat_features(self):
        user_features = []
        for i in range(len(self.user_feature_embed)):
            # print(len(self.user_feature_embed), len(self.user_tags))
            # print(self.user_feature_embed[i].weight.shape)
            user_features.append(self.user_feature_embed[i].weight[self.user_tags[i].to(torch.int64)])

        item_features = []
        for i in range(len(self.item_feature_embed)):
            item_features.append(self.item_feature_embed[i].weight[self.item_tags[i].to(torch.int64)])

        if len(user_features)>0:
            user_features = torch.cat(user_features,1)
        if len(item_features)>0:
            item_features = torch.cat(item_features,1)

        return user_features, item_features
        


    def compute(self):
        final_embed_user, final_embed_item = None, None
        if self.args.use_attribute:
            combined_user_feature, combined_item_feature = self.concat_features()
            if len(self.user_feature_embed) > 0:
                final_embed_user = torch.cat([combined_user_feature, self.embed_user.weight],1)
            if len(self.item_feature_embed) > 0:
                final_embed_item = torch.cat([combined_item_feature, self.embed_item.weight],1)


        users_emb = self.user_dense(final_embed_user) if final_embed_user is not None else  self.embed_user.weight
        items_emb = self.item_dense(final_embed_item) if final_embed_item is not None else self.embed_item.weight

        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items

    def forward(self, users, pos_items, neg_items):
        # input is a user, a positive, a negative.
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = torch.transpose(all_items[torch.tensor(items).cuda(self.device)], 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()

    def get_top_embeddings(self):
        all_users, all_items = self.compute()
        return all_users

    def get_bottom_embeddings(self):
        return self.embed_user.weight

    def get_predict_bias(self, bs=1024):
        all_users, all_items = self.compute()
        users = list(range(self.n_users))
        start=0
        users = items = torch.transpose(all_users[torch.tensor(users).cuda(self.device)], 0, 1)
        user_attributes=[]
        for index in range(len(self.user_tags)):
            user_attributes.append(self.user_tags[index].to(torch.int64).to(self.device))
        bias_scores=torch.zeros(len(self.user_tags)).to(self.device)
        while start < self.n_items:
            end = start + bs if start + bs < self.n_items else self.n_items
            item_idx=np.arange(start, end)
            start=end
            items = all_items[torch.tensor(item_idx).cuda(self.device)]
            rate_batch = torch.sigmoid(torch.matmul(items, users))
            
            bias_score=[]
            for attr in user_attributes:
                grp_avg=scatter(rate_batch, attr, dim=1, reduce="mean")
                grp_bias = torch.sum(torch.max(grp_avg, dim=1).values - torch.min(grp_avg, dim=1).values)
                # print(grp_bias)
                bias_score.append(grp_bias)
            bias_scores = bias_scores + torch.stack(bias_score)
        bias_scores = bias_scores / self.n_users
        return bias_scores.detach().cpu().numpy()
    
class IPS(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)

    def forward(self, users, pos_items, neg_items, pos_weights):
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.mul(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10), pos_weights)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss


class IPS_BCE(IPS):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.data = data 
        self.sigmoid = torch.nn.Sigmoid()
        self.bce = torch.nn.BCELoss(reduction='none')
    def forward(users, pos_items, neg_items, pos_weights):
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        pos_label = torch.ones((len(pos_emb),))
        neg_label = torch.zeros((len(neg_emb),))
        all_label = torch.cat((pos_label, neg_label),0)

        pred = torch.sum(torch.mul(torch.cat((users_emb, users_emb),0), all_emb), dim=1)
        pred = self.sigmoid(pred)
        bce_loss = self.bce(pred,all_label.cuda(self.device))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(itemEmb0) ** 2 
        regularizer = regularizer / self.batch_size
        mf_loss = torch.mean(torch.mul(xent_loss, weights))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss
    


class CausE(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.cf_pen = args.cf_pen
        self.embed_item_ctrl = nn.Embedding(self.n_items, self.emb_dim)
        nn.init.xavier_normal_(self.embed_item_ctrl.weight)

    def forward(self, users, pos_items, neg_items, all_reg, all_ctrl):
        all_users, all_items = self.compute()
        all_items = torch.cat([all_items, self.embed_item_ctrl.weight])

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]
        item_embed = all_items[all_reg]
        control_embed = all_items[all_ctrl]

        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        cf_loss = torch.sqrt(torch.sum(
            torch.square(torch.subtract(F.normalize(item_embed, p=2, dim=0), F.normalize(control_embed, p=2, dim=0)))))
        cf_loss = cf_loss * self.cf_pen  # / self.batch_size

        return mf_loss, reg_loss, cf_loss


class MACR(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.alpha = args.alpha
        self.beta = args.beta
        self.w = nn.Embedding(self.emb_dim, 1)
        self.w_user = nn.Embedding(self.emb_dim, 1)
        nn.init.xavier_normal_(self.w.weight)
        nn.init.xavier_normal_(self.w_user.weight)

        self.pos_item_scores = torch.empty((self.batch_size, 1))
        self.neg_item_scores = torch.empty((self.batch_size, 1))
        self.user_scores = torch.empty((self.batch_size, 1))

        self.rubi_c = args.c * torch.ones([1]).cuda(self.device)

    def forward(self, users, pos_items, neg_items):
        # Original scores
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        # Item module and User module
        self.pos_item_scores = torch.matmul(pos_emb, self.w.weight)
        self.neg_item_scores = torch.matmul(neg_emb, self.w.weight)
        self.user_scores = torch.matmul(users_emb, self.w_user.weight)

        # fusion
        # [batch_size,] [batch_size, 1] -> [batch_size, batch_size] * [batch_size, 1]
        # [batch_size * (bs-1)]
        pos_scores = pos_scores * torch.sigmoid(self.pos_item_scores) * torch.sigmoid(self.user_scores)
        neg_scores = neg_scores * torch.sigmoid(self.neg_item_scores) * torch.sigmoid(self.user_scores)
        # pos_scores = torch.mean(pos_scores) * torch.squeeze(torch.sigmoid(self.pos_item_scores)) * torch.squeeze(torch.sigmoid(self.user_scores))
        # neg_scores = torch.mean(neg_scores) * torch.squeeze(torch.sigmoid(self.neg_item_scores)) * torch.squeeze(torch.sigmoid(self.user_scores))

        # loss
        mf_loss_ori = torch.mean(torch.negative(torch.log(torch.sigmoid(pos_scores) + 1e-10)) + torch.negative(
            torch.log(1 - torch.sigmoid(neg_scores) + 1e-10)))

        mf_loss_item = torch.mean(
            torch.negative(torch.log(torch.sigmoid(self.pos_item_scores) + 1e-10)) + torch.negative(
                torch.log(1 - torch.sigmoid(self.neg_item_scores) + 1e-10)))

        mf_loss_user = torch.mean(torch.negative(torch.log(torch.sigmoid(self.user_scores) + 1e-10)) + torch.negative(
            torch.log(1 - torch.sigmoid(self.user_scores) + 1e-10)))

        mf_loss = mf_loss_ori + self.alpha * mf_loss_item + self.beta * mf_loss_user

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = torch.transpose(all_items[torch.tensor(items).cuda(self.device)], 0, 1)

        rate_batch = torch.matmul(users, items)

        item_scores = torch.matmul(torch.transpose(items, 0, 1), self.w.weight)
        user_scores = torch.matmul(users, self.w_user.weight)

        rubi_rating_both = (rate_batch - self.rubi_c) * (torch.sigmoid(user_scores)) * torch.transpose(
            torch.sigmoid(item_scores), 0, 1)

        return rubi_rating_both.cpu().detach().numpy()


class SAMREG(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.rweight = args.rweight

    def get_correlation_loss(self, y_true, y_pred):
        x = y_true
        y = y_pred
        mx = torch.mean(x)
        my = torch.mean(y)
        xm, ym = x - mx, y - my
        r_num = torch.sum(torch.mul(xm, ym))
        r_den = torch.sqrt(torch.mul(torch.sum(torch.square(xm)), torch.sum(torch.square(ym))))
        # print(r_den)
        r = r_num / (r_den + 1e-5)
        r = torch.square(torch.clamp(r, -1, 1))
        return r

    def forward(self, users, pos_items, neg_items, pop_weight):
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]

        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer / self.batch_size

        bpr = torch.sigmoid(pos_scores - neg_scores)

        maxi = torch.log(bpr)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        mf_loss = (1 - self.rweight) * (mf_loss + reg_loss)

        cor_loss = self.rweight * self.get_correlation_loss(pop_weight, bpr)

        return mf_loss, cor_loss


class INFONCE(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample = args.neg_sample if args.neg_sample != -1 else self.batch_size - 1

    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        neg_emb = F.normalize(neg_emb, dim=-1)

        pos_ratings = torch.sum(users_emb * pos_emb, dim=-1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1),
                                   neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim=1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator / denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss


class INFONCE_batch(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample = args.neg_sample if args.neg_sample != -1 else self.batch_size - 1

    def forward(self, users, pos_items):
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)

        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        numerator = torch.exp(ratings_diag / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim=1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator / denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2
        regularizer = regularizer
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss


class BC_LOSS(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau1 = args.tau1
        self.tau2 = args.tau2
        self.w_lambda = args.w_lambda
        self.neg_sample = args.neg_sample if args.neg_sample != -1 else self.batch_size - 1
        self.n_users_pop = data.n_user_pop
        self.n_items_pop = data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)

    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop):
        # popularity branch
        users_pop_emb = self.embed_user_pop(users_pop)
        pos_pop_emb = self.embed_item_pop(pos_items_pop)
        neg_pop_emb = self.embed_item_pop(neg_items_pop)

        pos_ratings_margin = torch.sum(users_pop_emb * pos_pop_emb, dim=-1)

        users_pop_emb = F.normalize(users_pop_emb, dim=-1)
        pos_pop_emb = F.normalize(pos_pop_emb, dim=-1)
        neg_pop_emb = F.normalize(neg_pop_emb, dim=-1)

        pos_ratings = torch.sum(users_pop_emb * pos_pop_emb, dim=-1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_pop_emb, 1),
                                   neg_pop_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau2)
        denominator = torch.sum(torch.exp(ratings / self.tau2), dim=1)
        loss2 = self.w_lambda * torch.mean(torch.negative(torch.log(numerator / denominator)))

        # main branch
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        neg_emb = F.normalize(neg_emb, dim=-1)

        pos_ratings = torch.sum(users_emb * pos_emb, dim=-1)
        pos_ratings = torch.cos(
            torch.arccos(torch.clamp(pos_ratings, -1 + 1e-7, 1 - 1e-7)) + (1 - torch.sigmoid(pos_ratings_margin)))
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1),
                                   neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)

        numerator = torch.exp(pos_ratings / self.tau1)
        denominator = torch.sum(torch.exp(ratings / self.tau1), dim=1)

        loss1 = (1 - self.w_lambda) * torch.mean(torch.negative(torch.log(numerator / denominator)))

        # reg loss
        regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + \
                       0.5 * torch.norm(negEmb0) ** 2
        regularizer1 = regularizer1 / self.batch_size

        regularizer2 = 0.5 * torch.norm(users_pop_emb) ** 2 + 0.5 * torch.norm(pos_pop_emb) ** 2 + \
                       0.5 * torch.norm(neg_pop_emb) ** 2
        regularizer2 = regularizer2 / self.batch_size
        reg_loss = self.decay * (regularizer1 + regularizer2)

        reg_loss_freeze = self.decay * (regularizer2)
        reg_loss_norm = self.decay * (regularizer1)

        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm

    def freeze_pop(self):
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)


class BC_LOSS_batch(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau1 = args.tau1
        self.tau2 = args.tau2
        self.w_lambda = args.w_lambda
        self.neg_sample = args.neg_sample if args.neg_sample != -1 else self.batch_size - 1
        self.n_users_pop = data.n_user_pop
        self.n_items_pop = data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)

    def forward(self, users, pos_items, users_pop, pos_items_pop):
        # popularity branch
        users_pop_emb = self.embed_user_pop(users_pop)
        pos_pop_emb = self.embed_item_pop(pos_items_pop)

        pos_ratings_margin = torch.sum(users_pop_emb * pos_pop_emb, dim=-1)

        users_pop_emb = F.normalize(users_pop_emb, dim=-1)
        pos_pop_emb = F.normalize(pos_pop_emb, dim=-1)

        ratings = torch.matmul(users_pop_emb, torch.transpose(pos_pop_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        numerator = torch.exp(ratings_diag / self.tau2)
        denominator = torch.sum(torch.exp(ratings / self.tau2), dim=1)
        loss2 = self.w_lambda * torch.mean(torch.negative(torch.log(numerator / denominator)))

        # main branch
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)

        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)

        ratings_diag = torch.cos(torch.arccos(torch.clamp(ratings_diag, -1 + 1e-7, 1 - 1e-7)) + \
                                 (1 - torch.sigmoid(pos_ratings_margin)))

        numerator = torch.exp(ratings_diag / self.tau1)
        denominator = torch.sum(torch.exp(ratings / self.tau1), dim=1)
        loss1 = (1 - self.w_lambda) * torch.mean(torch.negative(torch.log(numerator / denominator)))

        # reg loss
        regularizer1 = 0.5 * torch.norm(userEmb0) ** 2 + self.batch_size * 0.5 * torch.norm(posEmb0) ** 2
        regularizer1 = regularizer1 / self.batch_size

        regularizer2 = 0.5 * torch.norm(users_pop_emb) ** 2 + self.batch_size * 0.5 * torch.norm(pos_pop_embp) ** 2
        regularizer2 = regularizer2 / self.batch_size
        reg_loss = self.decay * (regularizer1 + regularizer2)

        reg_loss_freeze = self.decay * (regularizer2)
        reg_loss_norm = self.decay * (regularizer1)

        return loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm

    def freeze_pop(self):
        self.embed_user_pop.requires_grad_(False)
        self.embed_item_pop.requires_grad_(False)


class SimpleX(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.w_neg = args.w_neg
        self.margin = args.neg_margin
        self.neg_sample = args.neg_sample if args.neg_sample != -1 else self.batch_size - 1

    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb = F.normalize(users_emb, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        neg_emb = F.normalize(neg_emb, dim=-1)

        pos_ratings = torch.sum(users_emb * pos_emb, dim=-1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1),
                                   neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        pos_margin_loss = 1 - pos_ratings
        neg_margin_loss = torch.mean(torch.clamp(neg_ratings - self.margin, 0, 1), dim=-1)

        mf_loss = torch.mean(pos_margin_loss + self.w_neg * neg_margin_loss)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss


class SimpleX_batch(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.w_neg = args.w_neg
        self.margin = args.neg_margin
        self.neg_sample = self.batch_size - 1

    def forward(self, users, pos_items):
        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)

        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        diag_mask = torch.ones_like(ratings_diag).cuda(self.device) - torch.eye(self.batch_size).cuda(self.device)

        pos_margin_loss = 1 - ratings_diag
        neg_margin_loss = torch.sum(torch.clamp(ratings - self.margin, 0, 1) * diag_mask, dim=-1) / self.neg_sample

        mf_loss = torch.mean(pos_margin_loss + self.w_neg * neg_margin_loss)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + self.batch_size * 0.5 * torch.norm(posEmb0) ** 2
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

        

class INV_LGN(MF):
    def __init__(self, args, data):
        super().__init__(args, data)
        if args.is_geometric == 1:
            self.edge_index = data.getEdgeIndex().cuda(self.device)

    #         self.adj_mat = data.getSparseGraph().coalesce()
    #         self.index = self.adj_mat.indices()
    #         self.edge_index = SparseTensor(row=self.index[0], col=self.index[1], value=self.adj_mat.values(), sparse_sizes=(
    # self.n_users + self.n_items, self.n_users + self.n_items)).cuda(self.device)

            self.M = Mask_Model_Geometric(args)
            self.lgn = LGConv()
    #         alpha = 1. / (args.n_layers + 1)
    #         if isinstance(alpha, Tensor):
    #             assert alpha.size(0) == args.n_layers + 1
    #         else:
    #             alpha = torch.tensor([alpha] * (args.n_layers + 1))
    #         self.register_buffer('alpha', alpha)

        else:
            # (U+I) x (U+I) matrix
            self.Graph = data.getSparseGraph().cuda(self.device)
            # U-I matrix
            self.adj_mat = data.getSparseGraph(ui_only=True).cuda(self.device)
            self.M = Mask_Model(args, self.adj_mat, self.Graph)

        self.n_layers = args.n_layers
        self.inv_loss = Inv_Loss_Rank(args)
        self.args = args
        self.warmup = True

    def compute(self, mask=False):
        # add masked LGN propogation
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        if self.args.is_geometric == 1:
            for layer in range(self.n_layers):
                all_emb = self.M(all_emb, self.edge_index) if mask else self.lgn(all_emb, self.edge_index)
                embs.append(all_emb)
            # out = self.gcn(all_emb, self.edge_index)
            # users, items = torch.split(out, [self.n_users, self.n_items])

        else:
            g_droped = self.Graph
            if mask:
                # case 0: simple mask
                if self.args.mask == 0:
                    M = self.M.mask_simple(users_emb, items_emb)
                # case 1: attention mask
                if self.args.mask == 1:
                    M = self.M.mask_attention(users_emb, items_emb)
            else:
                M = None

            for layer in range(self.n_layers):
                all_emb = torch_sparse.spmm(g_droped.indices(), M.coalesce().values() * g_droped.values(),
                                                g_droped.shape[0], g_droped.shape[1],
                                                all_emb) if mask else torch_sparse.spmm(g_droped.indices(), g_droped.values(), g_droped.shape[0], g_droped.shape[1],
                                            all_emb)
                # torch.sparse.mm(g_droped, all_emb)
                # if mask and layer == self.n_layers - 1:
                #     all_emb = torch_sparse.spmm(g_droped.indices(),  M.coalesce().values()* g_droped.values(), g_droped.shape[0], g_droped.shape[1],
                #                             all_emb)
                embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items


    def regularize(self, users, pos_items, neg_items):
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)
        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        return regularizer

    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        regularizer = self.regularize(users, pos_items, neg_items) / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        if self.warmup:
            inv_loss = torch.tensor(0).to(self.device)
        else:
            all_users_m, all_items_m = self.compute(True)
            inv_loss = self.inv_loss(all_items, all_items_m, all_users, all_users_m, users)

        return mf_loss, reg_loss, inv_loss

    def forward_adaptive(self, users, pos_items, neg_items):

        all_users_m, all_items_m = self.compute(True)
        all_users, all_items = self.compute()
        inv_loss = self.inv_loss(all_items, all_items_m, all_users, all_users_m, users)

        users_emb = all_users_m[users]
        pos_emb = all_items_m[pos_items]
        neg_emb = all_items_m[neg_items]

        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)
        mf_loss_m = torch.negative(torch.mean(maxi))

        return mf_loss_m, inv_loss

    def freeze_args(self, flag):
        if flag:
            self.embed_user.requires_grad_(False)
            self.embed_item.requires_grad_(False)
            for param in self.M.parameters():
                param.requires_grad = True
        else:
            self.embed_user.requires_grad_(True)
            self.embed_item.requires_grad_(True)
            for param in self.M.parameters():
                param.requires_grad = False

    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = torch.transpose(all_items[torch.tensor(items).cuda(self.device)], 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()


class INV_LGN_DUAL(MF):
    def __init__(self, args, data, writer):
        super().__init__(args, data)
        self.Graph = data.getSparseGraph().cuda(self.device)
        self.args = args
        if self.args.dropout_type == 1:
            self.edge_index = data.getEdgeIndex().cuda(self.device)
        self.n_layers = args.n_layers
        self.inv_loss = Inv_Loss_Embed(args)
        self.M = Mask_Model_Attention(args)
        self.warmup = True
        self.sigmoid = nn.Sigmoid()

        self.embed_user_dual = nn.Embedding(self.n_users, self.emb_dim)
        self.embed_item_dual = nn.Embedding(self.n_items, self.emb_dim)

        nn.init.xavier_normal_(self.embed_user_dual.weight)
        nn.init.xavier_normal_(self.embed_item_dual.weight)
        self.is_train=True
        self.writer = writer
        self.global_step=0
        self.user_tags=[]
        self.item_tags=[]
        if 'ml' in args.dataset:
            self.user_tags = data.get_user_tags()
        
        if 'coat' in args.dataset:
            self.user_tags = data.get_user_tags()
            self.item_tags = data.get_item_tags()
        
        if self.args.use_attribute:
            self.user_feature_embed = []
            self.item_feature_embed = []
            self.generate_embedings(self.user_tags, self.user_feature_embed)
            self.generate_embedings(self.item_tags, self.item_feature_embed)
            self.user_dense = nn.Linear(self.emb_dim* (len(self.user_tags)+1) ,self.emb_dim)
            self.user_dense_dual = nn.Linear(self.emb_dim*(len(self.user_tags)+1),self.emb_dim)
            self.item_dense = nn.Linear(self.emb_dim*(len(self.item_tags)+1),self.emb_dim)
            self.item_dense_dual = nn.Linear(self.emb_dim*(len(self.item_tags)+1),self.emb_dim)

            
    def generate_embedings(self, tags, feature_embed):
        featuren_len = len(tags) 
        if featuren_len > 0:
            for i in range(featuren_len):
                max_value = torch.max(tags[i])+1
                embed = nn.Embedding(max_value, self.emb_dim).to(self.device)
                nn.init.xavier_normal_(embed.weight)
                feature_embed.append(embed)
                # feature_embed[i] = embed

    def concat_features(self):
        user_features = []
        for i in range(len(self.user_feature_embed)):
            # print(len(self.user_feature_embed), len(self.user_tags))
            # print(self.user_feature_embed[i].weight.shape)
            user_features.append(self.user_feature_embed[i].weight[self.user_tags[i].to(torch.int64)])

        item_features = []
        for i in range(len(self.item_feature_embed)):
            item_features.append(self.item_feature_embed[i].weight[self.item_tags[i].to(torch.int64)])

        if len(user_features)>0:
            user_features = torch.cat(user_features,1)
        if len(item_features)>0:
            item_features = torch.cat(item_features,1)

        return user_features, item_features
    

    def compute_mask_gini(self, mask, index, view='user'):
        # get edge_user_index 
        edge_user_index = torch.where(self.edge_index[0,:] < self.n_users, self.edge_index[0,:], self.edge_index[1,:])
        edge_item_index = torch.where(self.edge_index[0,:] < self.n_users, self.edge_index[1,:]-self.n_users, self.edge_index[0,:]-self.n_users)
        edge_attribute = self.user_tags[index][edge_user_index].to(torch.int64).to(self.device) if view=='user' else self.item_tags[index][edge_item_index].to(torch.int64).to(self.device)
        # print(edge_attribute_user.shape,  mask.shape)
        kk = scatter(mask, edge_attribute, dim=0, reduce="mean")
        return gini_index(kk, self.device), kk
    
    def compute_cluster_loss(self, mask, index, view = 'user'):
        # get edge_user_index 
        edge_user_index = torch.where(self.edge_index[0,:] < self.n_users, self.edge_index[0,:], self.edge_index[1,:])
        edge_item_index = torch.where(self.edge_index[0,:] < self.n_users, self.edge_index[1,:]-self.n_users, self.edge_index[0,:]-self.n_users)
        edge_attribute = self.user_tags[index][edge_user_index].to(torch.int64).to(self.device) if view=='user' else self.item_tags[index][edge_item_index].to(torch.int64).to(self.device)
        # print(edge_attribute_user.shape,  mask.shape)
        kk = scatter(mask, edge_attribute, dim=0, reduce="mean")
        kk = kk.reshape((1,-1))
        loss = torch.mean(torch.pow((kk - kk.T)**2 + 1e-10, 1/2))
        return loss, kk

        
    def draw_graph_init(self, mask, start='user'):
        G = nx.DiGraph()
        edges = self.edge_index.cpu().numpy().T
        new_mask=[]
        for i in range(len(edges)):
            e = edges[i]
            if start=='user':
                if e[0]<self.n_users:
                    G.add_edge(e[0], e[1], weight=mask[i])
                    new_mask.append(mask[i])
            else:
                if e[0]>=self.n_users:
                    G.add_edge(e[0], e[1], weight=mask[i])
                    new_mask.append(mask[i])
        edge_labels = nx.get_edge_attributes(G, "weight")
        return G, edge_labels,new_mask

    def add_node_tag(self, G, user_index, item_index):
        node_attribute_user = self.user_tags[user_index]
        node_attribute_item = self.item_tags[item_index]
        node_attribute_all = torch.cat((node_attribute_user, node_attribute_item),0).numpy()
        for i in range(self.n_users + self.n_items):
            if i< self.n_users:
                G.add_node(i, feature= node_attribute_all[i])
            else:
                G.add_node(i, feature= node_attribute_all[i])
    
        labels = nx.get_node_attributes(G, 'feature')
        return G, labels

    def step(self):
        self.global_step+=1

    def __dropout(self, graph, keep_prob, mask, is_arm=False):
        size = graph.size()
        index = graph.indices().t()
        values = graph.values()
        if not is_arm:
            if self.args.dropout_type == 0:
                random_index = torch.cuda.FloatTensor(len(values)).uniform_().cuda(self.device) + keep_prob
                # random_index = torch.rand(len(values)).cuda(self.device) + keep_prob
            else:
                random_index = torch.cuda.FloatTensor(len(values)).uniform_().cuda(self.device) + mask
                # random_index = torch.rand(len(values)).cuda(self.device) + mask
            random_index = random_index.int().bool()
        else:
            random_index = mask
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def compute(self, dual=False, dropout = False, mask = None):
        final_embed_user, final_embed_item = None, None
        if self.args.use_attribute:
            combined_user_feature, combined_item_feature = self.concat_features()
            if len(self.user_feature_embed) > 0:
                final_embed_user = torch.cat([combined_user_feature, self.embed_user_dual.weight],1) if dual else torch.cat([combined_user_feature, self.embed_user.weight],1)
            if len(self.item_feature_embed) > 0:
                final_embed_item = torch.cat([combined_item_feature, self.embed_item_dual.weight],1) if dual else torch.cat([combined_item_feature, self.embed_item.weight],1)

        is_arm = True if mask != None else False

        if not dual:
            users_emb = self.user_dense(final_embed_user) if final_embed_user is not None else  self.embed_user.weight
            items_emb = self.item_dense(final_embed_item) if final_embed_item is not None else self.embed_item.weight
        else:
            users_emb = self.user_dense_dual(final_embed_user) if final_embed_user is not None else  self.embed_user_dual.weight
            items_emb = self.item_dense_dual(final_embed_item) if final_embed_item is not None else self.embed_item_dual.weight
        
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        if dropout:
            if mask == None:
                if self.args.dropout_type == 0:
                    mask = None
                else:
                    mask = self.M(all_emb, self.edge_index) if dual else 1 - self.M(all_emb, self.edge_index)
            g_droped = self.__dropout(self.Graph, self.args.keep_prob, mask, is_arm).cuda(self.device)
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items
    
    def regularize(self, users, pos_items, neg_items, dual=False):
        if not dual:
            userEmb0 = self.embed_user(users)
            posEmb0 = self.embed_item(pos_items)
            negEmb0 = self.embed_item(neg_items)
        else:
            userEmb0 = self.embed_user_dual(users)
            posEmb0 = self.embed_item_dual(pos_items)
            negEmb0 = self.embed_item_dual(neg_items)
        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        return regularizer
    

    def forward(self, users, pos_items, neg_items, is_draw=False, is_cluster = False):
        mf_loss=0
        reg_loss=0
        user_embeds=[]
        item_embeds=[]


        for dual_ind in [True,False]:
            all_users, all_items =  self.compute(dual = dual_ind, dropout=True)
            if dual_ind and self.args.dropout_type == 1:
                if is_draw:
                    mask = self.get_mask(dual_ind)
                    self.writer.add_histogram('Dropout Mask', mask, self.global_step)
                    for a_index in range(len(self.user_tags)):
                        #gini_value, kk  = self.compute_mask_gini(mask, a_index,'user')
                        dist_value, kk  = self.compute_cluster_loss(mask, a_index,'user')
                        self.writer.add_scalar(f'Attribute_Dist/User Attribute {a_index}', dist_value, self.global_step)
                        
                        #self.writer.add_scalar(f'Attribute_Gini/User Attribute {a_index}', gini_value, self.global_step)
                        self.writer.add_histogram(f'Attribute_Distribution/User Attribute {a_index}',kk, self.global_step)
                        #self.writer.add_scalars(f'Attribute_Means/User Attribute Distribution {a_index}', {f"group {i}":kk[i] for i in range(len(kk))}, self.global_step)
                    for a_index in range(len(self.item_tags)):
                        #print(self.item_tags[a_index].shape)
                        #gini_value, kk  = self.compute_mask_gini(mask, a_index, 'item')
                        #self.writer.add_scalar(f'Attribute_Gini/Item Attribute {a_index}', gini_value, self.global_step)
                        dist_value, kk  = self.compute_cluster_loss(mask, a_index,'item')
                        self.writer.add_scalar(f'Attribute_Dist/Item Attribute {a_index}', dist_value, self.global_step)
                        self.writer.add_histogram(f'Attribute_Distribution/Item Attribute {a_index}',kk, self.global_step)
                        #self.writer.add_scalars(f'Attribute_Means/Item Attribute Distribution {a_index}', {f"group {i}":kk[i] for i in range(len(kk))}, self.global_step)

            user_embeds.append(all_users)
            item_embeds.append(all_items)

            users_emb = all_users[users]
            pos_emb = all_items[pos_items]
            neg_emb = all_items[neg_items]

            pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
            neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

            regularizer = self.regularize(users, pos_items, neg_items, dual_ind) / self.batch_size

            maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

            mf_loss = mf_loss + torch.negative(torch.mean(maxi))
            reg_loss = reg_loss+ self.decay * regularizer
        
        #print(type(self.args.inv_tau))
        
        inv_loss, losses= self.inv_loss(user_embeds, item_embeds)
        inv_loss = self.args.inv_tau*inv_loss
        #inv_loss = -self.inv_loss(item_embeds[0], item_embeds[1], user_embeds[0], user_embeds[1], users)
        if is_cluster:
            mask = self.get_mask(True)
            print("------")
            for i in range(len(self.user_tags)):
                print(self.compute_cluster_loss(mask, i)[0])
            print("------")
        return mf_loss, reg_loss, inv_loss


    def get_mask(self, dual_ind):
        if not dual_ind:
            users_emb = self.embed_user.weight
            items_emb = self.embed_item.weight
        else:
            users_emb = self.embed_user_dual.weight
            items_emb = self.embed_item_dual.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]

        mask = self.M(all_emb, self.edge_index) if dual_ind else 1 - self.M(all_emb, self.edge_index) 

        return mask

    def forward_ARM(self):

        u=torch.rand(len(self.Graph.values())).cuda(self.device)

        mf_loss=0
        reg_loss=0
        user_embeds=[[],[]]
        item_embeds=[[],[]]
        cluster_loss_inv1 = 0 
        cluster_loss_inv2 = 0 
        
        for dual_ind in [True,False]:
            mask = self.get_mask(dual_ind)

            if dual_ind:
                self.writer.add_histogram('Dropout Mask', mask, self.global_step)
            drop1 = u > 1 - mask
            drop2 = u < mask
            if self.args.use_mask_inv:
                for i in range(len(self.user_tags)):
                    if dual_ind:
                        cluster_loss_inv1 += self.compute_cluster_loss(drop1.to(torch.float), i)[0]
                    else: 
                        cluster_loss_inv2 += self.compute_cluster_loss(drop2.to(torch.float), i)[0]
            # print("drop1 shape: ", drop1.shape)
            # print("count", torch.sum(drop1))

            # print("drop2 shape: ", drop2.shape)
            # print("count", torch.sum(drop2))

            for idx, drop in enumerate([drop1, drop2]):
                all_users, all_items =  self.compute(dual = dual_ind, dropout=True, mask=drop)
                user_embeds[idx].append(all_users)
                item_embeds[idx].append(all_items)

        a = self.inv_loss(user_embeds[0], item_embeds[0])[0]
        b = self.inv_loss(user_embeds[1], item_embeds[1])[0]
        if self.args.use_mask_inv:
            inv_loss1 = a + cluster_loss_inv1*self.args.cluster_coe 
            inv_loss2 = b + cluster_loss_inv2*self.args.cluster_coe 
        else:
            inv_loss1 = a 
            inv_loss2 = b 
        # print("inv loss 1", inv_loss1)
        # print(user_embeds[0].shape, item_embeds[0].shape)
        # print("inv loss 2", inv_loss2)
        my_grad = self.args.grad_coeff * (-inv_loss1 + inv_loss2) * (u-0.5) 
        return my_grad
        
    
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = torch.transpose(all_items[torch.tensor(items).cuda(self.device)], 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()
    
    def freeze_args(self, adv=True):
        if adv:
            self.embed_user.requires_grad_(False)
            self.embed_item.requires_grad_(False)
            self.embed_user_dual.requires_grad_(False)
            self.embed_item_dual.requires_grad_(False)
            if self.args.use_attribute:
                for u in self.user_feature_embed:
                    u.requires_grad_(False)
                for i in self.item_feature_embed:
                    i.requires_grad_(False)

                self.user_dense.requires_grad_(False)
                self.user_dense_dual.requires_grad_(False)
                self.item_dense.requires_grad_(False)
                self.item_dense_dual.requires_grad_(False)
                
            
            for param in self.M.parameters():
                param.requires_grad = True
        else:
            self.embed_user.requires_grad_(True)
            self.embed_item.requires_grad_(True)
            self.embed_user_dual.requires_grad_(True)
            self.embed_item_dual.requires_grad_(True)
            if self.args.use_attribute:
                for u in self.user_feature_embed:
                    u.requires_grad_(True)
                for i in self.item_feature_embed:
                    i.requires_grad_(True)

                self.user_dense.requires_grad_(True)
                self.user_dense_dual.requires_grad_(True)
                self.item_dense.requires_grad_(True)
                self.item_dense_dual.requires_grad_(True)

            for param in self.M.parameters():
                param.requires_grad = False

    def new_predict(self, user_idx, item_idx):
        all_users, all_items = self.compute()
        users_emb = all_users[user_idx]
        all_emb = all_items[item_idx]

        pred = torch.sum(torch.mul(users_emb, all_emb), dim=1)
        pred = self.sigmoid(pred)

        return pred.detach().cpu().numpy()

    def get_top_embeddings(self):
        all_users, all_items = self.compute()
        return all_users
    def get_bottom_embeddings(self):
        return self.embed_user.weight

    def get_predict_bias(self, bs=1024):
        all_users, all_items = self.compute()
        users = list(range(self.n_users))
        start=0
        users = items = torch.transpose(all_users[torch.tensor(users).cuda(self.device)], 0, 1)
        user_attributes=[]
        for index in range(len(self.user_tags)):
            user_attributes.append(self.user_tags[index].to(torch.int64).to(self.device))
        bias_scores=torch.zeros(len(self.user_tags)).to(self.device)
        while start < self.n_items:
            end = start + bs if start + bs < self.n_items else self.n_items
            item_idx=np.arange(start, end)
            start=end
            items = all_items[torch.tensor(item_idx).cuda(self.device)]
            rate_batch = torch.sigmoid(torch.matmul(items, users))
            
            bias_score=[]
            for attr in user_attributes:
                grp_avg=scatter(rate_batch, attr, dim=1, reduce="mean")
                grp_bias = torch.sum(torch.max(grp_avg, dim=1).values - torch.min(grp_avg, dim=1).values)
                # print(grp_bias)
                bias_score.append(grp_bias)
            bias_scores = bias_scores + torch.stack(bias_score)
        bias_scores = bias_scores / self.n_users
        return bias_scores.detach().cpu().numpy()
        


class INV_LGN_DUAL_BCE(INV_LGN_DUAL):
    def __init__(self, args, data, writer):
        super().__init__(args, data, writer)
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss()
    def forward(self, users, pos_items, neg_items):
        mf_loss=0
        reg_loss=0
        user_embeds=[]
        item_embeds=[]

        for dual_ind in [True,False]:
            all_users, all_items =  self.compute(dual = dual_ind, dropout=True)
            if dual_ind and self.args.dropout_type == 1:
                mask = self.get_mask(dual_ind)
                self.writer.add_histogram('Dropout Mask', mask, self.global_step)
            user_embeds.append(all_users)
            item_embeds.append(all_items)

            users_emb = all_users[users]
            pos_emb = all_items[pos_items]
            neg_emb = all_items[neg_items]
            all_emb = torch.cat((pos_emb, neg_emb),0)

            userEmb0 = self.embed_user(users)
            posEmb0 = self.embed_item(pos_items)
            negEmb0 = self.embed_item(neg_items)

            pos_label = torch.ones((len(pos_emb),))
            neg_label = torch.zeros((len(neg_emb),))
            all_label = torch.cat((pos_label, neg_label),0)

            pred = torch.sum(torch.mul(torch.cat((users_emb, users_emb),0), all_emb), dim=1)
            pred = self.sigmoid(pred)
            bce_loss = self.bce(pred,all_label.cuda(self.device))

            regularizer = self.regularize(users, pos_items, neg_items, dual_ind) / self.batch_size

            mf_loss = mf_loss + bce_loss
            reg_loss = reg_loss+ self.decay * regularizer
        
        inv_loss, losses=self.args.inv_tau*self.inv_loss(user_embeds, item_embeds)
        #inv_loss = -self.inv_loss(item_embeds[0], item_embeds[1], user_embeds[0], user_embeds[1], users)

        return mf_loss, reg_loss, inv_loss


class CVIB(MF):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.Graph = data.getSparseGraph().cuda(self.device)
        self.n_layers = args.n_layers
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss()
        self.alpha = args.cvib_alpha
        self.gamma = args.cvib_gamma
        self.args = args

    def generate_samples(self):
        user = torch.randint(self.n_users, (self.args.batch_size*2, ))
        item = torch.randint(self.n_items, (self.args.batch_size*2, ))
        return user, item


    def compute(self):
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items
    


    def forward(self, users, pos_items, neg_items, sampled_user, sampled_items):
        # input is a user, a positive, a negative.
        all_users, all_items = self.compute()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        all_emb = torch.cat((pos_emb, neg_emb),0)


        pos_label = torch.ones((len(pos_emb),))
        neg_label = torch.zeros((len(neg_emb),))
        all_label = torch.cat((pos_label, neg_label),0)

        pred = torch.sum(torch.mul(torch.cat((users_emb, users_emb),0), all_emb), dim=1)
        pred = self.sigmoid(pred)
        # need label 
        bce_loss = self.bce(pred, all_label.cuda(self.device))

        sampled_user_emb = all_users[sampled_user]
        sampled_item_emb = all_items[sampled_items]
        pred_ul = torch.sum(torch.mul(sampled_user_emb, sampled_item_emb), dim=1)
        pred_ul = self.sigmoid(pred_ul)

        logp_hat = pred.log()

        pred_avg = pred.mean()
        pred_ul_avg = pred_ul.mean()
        
        info_loss = self.alpha * (- pred_avg * pred_ul_avg.log() - (1-pred_avg) * (1-pred_ul_avg).log()) + self.gamma* torch.mean(pred * logp_hat)

        return bce_loss, info_loss

    
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = torch.transpose(all_items[torch.tensor(items).cuda(self.device)], 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()



class CVIB_SEQ(CVIB):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.Graph = data.getSparseGraph().cuda(self.device)
        self.n_layers = args.n_layers
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss()
        self.alpha = args.cvib_alpha
        self.gamma = args.cvib_gamma
        self.args = args


    def compute(self):
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items
    


    def forward(self, users, items, labels, sampled_user, sampled_items):
        # input is a user, a positive, a negative.
        all_users, all_items = self.compute()
        users_emb = all_users[users]
        items_emb = all_items[items]

        pred = torch.sum(torch.mul(users_emb, items_emb), dim=1)
        pred = self.sigmoid(pred)
        # need label 
        bce_loss = self.bce(pred, labels)

        sampled_user_emb = all_users[sampled_user]
        sampled_item_emb = all_items[sampled_items]
        pred_ul = torch.sum(torch.mul(sampled_user_emb, sampled_item_emb), dim=1)
        pred_ul = self.sigmoid(pred_ul)

        logp_hat = pred.log()

        pred_avg = pred.mean()
        pred_ul_avg = pred_ul.mean()

        L2_reg: torch.Tensor = self.get_L2_reg(users, items)
        #L1_reg: torch.Tensor = self.get_L1_reg(users, items)
        
        info_loss = self.alpha * (- pred_avg * pred_ul_avg.log() - (1-pred_avg) * (1-pred_ul_avg).log()) + self.gamma* torch.mean(pred * logp_hat)

        return bce_loss, info_loss*0.1+ L2_reg 

    
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = torch.transpose(all_items[torch.tensor(items).cuda(self.device)], 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()

    def new_predict(self, user_idx, item_idx):
        all_users, all_items = self.compute()
        users_emb = all_users[user_idx]
        all_emb = all_items[item_idx]

        pred = torch.sum(torch.mul(users_emb, all_emb), dim=1)
        pred = self.sigmoid(pred)

        return pred.detach().cpu().numpy()

    def get_users_reg(self, users_id, norm: int):
        embed_gmf: torch.Tensor = self.embed_user(users_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(users_id)) * float(self.emb_dim))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(users_id)) * float(self.emb_dim))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss
        
    def get_items_reg(self, items_id, norm: int):
        embed_gmf: torch.Tensor = self.embed_item(items_id)
        if norm == 2:
            reg_loss: torch.Tensor = embed_gmf.norm(2).pow(2) / (float(len(items_id)) * float(self.emb_dim))
        elif norm == 1:
            reg_loss: torch.Tensor = embed_gmf.norm(1) / (float(len(items_id)) * float(self.emb_dim))
        else:
            raise KeyError('norm must be 1 or 2 !!! wdnmd !!')
        return reg_loss

    def get_L1_reg(self, users_id, items_id) -> torch.Tensor:
        return self.get_users_reg(users_id, 1) + self.get_items_reg(items_id, 1)

    def get_L2_reg(self, users_id, items_id) -> torch.Tensor:
        return self.get_users_reg(users_id, 2) + self.get_items_reg(items_id, 2)

class DR_SEQ(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.Graph = data.getSparseGraph().cuda(self.device)
        self.n_layers = args.n_layers
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss()
        self.args = args
        self.data = data
    
    def generate_samples(self):
        user = torch.randint(self.n_users, (self.args.batch_size*2, ))
        item = torch.randint(self.n_items, (self.args.batch_size*2, ))
        return user, item
     
    def _compute_IPS(self,user_index, item_index, y, y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(user_index) / (max(user_index) * max(item_index))
            py1o1 = sum(y) / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / (propensity + 1e-6)

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl
    
    def forward(self, users, items, labels, sampled_user, sampled_items,inv_prop, imputation_y):
        all_users, all_items = self.compute()
        users_emb = all_users[users]
        items_emb = all_items[items]
        
        pred = torch.sum(torch.mul(users_emb, items_emb), dim=1)
        pred = self.sigmoid(pred)

        sampled_user_emb = all_users[sampled_user]
        sampled_item_emb = all_items[sampled_items]
        pred_ul = torch.sum(torch.mul(sampled_user_emb, sampled_item_emb), dim=1)
        pred_ul = self.sigmoid(pred_ul)
        # print(max(labels), min(labels))
        xent_loss = F.binary_cross_entropy(pred, labels, weight=inv_prop, reduction="sum")
        # print(max(imputation_y), min(imputation_y))
        imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")

        ips_loss = (xent_loss - imputation_loss)/len(labels)

        # direct loss
        direct_loss = F.binary_cross_entropy(pred_ul, imputation_y,reduction="mean")
        return ips_loss, direct_loss


    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = torch.transpose(all_items[torch.tensor(items).cuda(self.device)], 0, 1)
        rate_batch = torch.matmul(users, items)

        return rate_batch.cpu().detach().numpy()
    
    def new_predict(self, user_idx, item_idx):
        all_users, all_items = self.compute()
        users_emb = all_users[user_idx]
        all_emb = all_items[item_idx]

        pred = torch.sum(torch.mul(users_emb, all_emb), dim=1)
        pred = self.sigmoid(pred)

        return pred.detach().cpu().numpy()


class DR(DR_SEQ):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss()
    
    def forward(self, users, pos_items, neg_items, sampled_user, sampled_items,inv_prop, imputation_y):
        all_users, all_items = self.compute()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        all_emb = torch.cat((pos_emb, neg_emb),0)


        pos_label = torch.ones((len(pos_emb),))
        neg_label = torch.zeros((len(neg_emb),))
        all_label = torch.cat((pos_label, neg_label),0)

        pred = torch.sum(torch.mul(torch.cat((users_emb, users_emb),0), all_emb), dim=1)
        pred = self.sigmoid(pred)
        # inv_prop = inv_prop.reshape((-1,))
        xent_loss = F.binary_cross_entropy(pred, all_label.cuda(self.device), weight=inv_prop, reduction="sum")
        imputation_loss = F.binary_cross_entropy(pred, imputation_y.repeat(2, ), weight=inv_prop, reduction="sum")
        ips_loss = (xent_loss - imputation_loss)/len(all_label)

        direct_loss_o = F.binary_cross_entropy(pred, imputation_y.repeat(2, ), reduction="mean")

        sampled_user_emb = all_users[sampled_user]
        sampled_item_emb = all_items[sampled_items]
        pred_ul = torch.sum(torch.mul(sampled_user_emb, sampled_item_emb), dim=1)
        pred_ul = self.sigmoid(pred_ul)
        # direct loss
        direct_loss = F.binary_cross_entropy(pred_ul, imputation_y.repeat(2, ), reduction="mean")
        return ips_loss, direct_loss+direct_loss_o


class LGN_BCE(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.sigmoid = torch.nn.Sigmoid()
        self.bce = torch.nn.BCELoss()
        
    def forward(self, users, pos_items, neg_items):
        # input is a user, a positive, a negative.
        all_users, all_items = self.compute()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        all_emb = torch.cat((pos_emb, neg_emb),0)

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        pos_label = torch.ones((len(pos_emb),))
        neg_label = torch.zeros((len(neg_emb),))
        all_label = torch.cat((pos_label, neg_label),0)

        pred = torch.sum(torch.mul(torch.cat((users_emb, users_emb),0), all_emb), dim=1)
        pred = self.sigmoid(pred)
        bce_loss = self.bce(pred,all_label.cuda(self.device))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size

        reg_loss = self.decay * regularizer

        return bce_loss, reg_loss



'''
class BPRMF(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)

    def forward(self, users, pos_items, neg_items):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]

        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores))

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss

class BCEMF(MF):
    def __init__(self, args, data):
        super().__init__(args, data)

    def forward(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 0.5 * torch.norm(users) ** 2 + 0.5 * torch.norm(pos_items) ** 2 + 0.5 * torch.norm(neg_items) ** 2
        regularizer = regularizer / self.batch_size

        mf_loss = torch.mean(torch.negative(torch.log(torch.sigmoid(pos_scores) + 1e-9))
                             + torch.negative(torch.log(1 - torch.sigmoid(neg_scores) + 1e-9)))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss
'''


class CFC(LGN):
    def __init__(self, args, data, writer):
        super().__init__(args, data)
        self.user_tags=[]
        self.item_tags=[]
        self.embed_dim = args.embed_size
        self.writer= writer
        if 'ml' in args.dataset:
            self.user_tags = data.get_user_tags()
        
        if 'coat' in args.dataset:
            self.user_tags = data.get_user_tags()
            self.item_tags = data.get_item_tags()
        
        self.user_feature_embed = []
        self.item_feature_embed = []
        self.generate_embedings(self.user_tags, self.user_feature_embed)
        self.generate_embedings(self.item_tags, self.item_feature_embed)


        self.filters=[]
        self.discriminators=[]
        for idx in range(len(self.user_tags)):
            self.filters.append(Filter(args))
            self.discriminators.append(Discriminator(args,self.user_tags[idx]))
    
    def mask_fairDiscriminators(self, discriminators, mask):
        # compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
        return [d for d, s in zip(discriminators, mask) if s]


    def apply_filter(self, user_embeds, masked_filter_set):
        filter_emb = None
        S = 0 
        for filter_ in masked_filter_set:
            if filter_ is not None:
                filter_emb = filter_(user_embeds) if filter_emb is None else filter_emb+filter_(user_embeds)
                S += 1
        return filter_emb/S
    

    def apply_discriminator(self, user_embeds, attr_labels, discriminator_set):
        loss = 0
        for i, disc_ in enumerate(discriminator_set):
            if disc_ is not None:
                loss += disc_(user_embeds,attr_labels[i])
        return loss



    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.compute()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        # TODO: Filter Embedding
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        if self.args.sample_mask:
            mask = np.random.choice([0, 1], size=(len(self.filters),))
            while np.all(mask==0):
                mask = np.random.choice([0, 1], size=(len(self.filters),))
            masked_filter_set = self.mask_fairDiscriminators(self.filters, mask)
            masked_discriminator_set = self.mask_fairDiscriminators(self.discriminators, mask)
        else:
            masked_filter_set = self.filters

        users_emb = self.apply_filter(users_emb, masked_filter_set)
        #pos_emb = self.apply_filter(pos_emb, masked_filter_set)
        #neg_emb = self.apply_filter(neg_emb, masked_filter_set)


        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        # TODO: Discriminator Loss
        users_attr =[self.user_tags[index][users].to(torch.int64).to(self.device) for index in range(len(self.user_tags))]
        masked_users_attr = self.mask_fairDiscriminators(users_attr, mask)
        dis_loss = self.apply_discriminator(users_emb,masked_users_attr, masked_discriminator_set)



        return mf_loss, reg_loss, dis_loss

    
    def get_top_embeddings(self):
        all_users, all_items = self.compute()
        masked_filter_set = self.filters
        users_emb = self.apply_filter(all_users, masked_filter_set)
        return users_emb
    

    def freeze_args(self, adv=True):
        if adv:
            self.embed_user.requires_grad_(False)
            self.embed_item.requires_grad_(False)

            for filt_ in self.filters:
                for param in filt_.parameters():
                    param.requires_grad = False
            

            for disc_ in self.discriminators:
                for param in disc_.parameters():
                    param.requires_grad = True
            
        else:
            self.embed_user.requires_grad_(True)
            self.embed_item.requires_grad_(True)

            for filt_ in self.filters:
                for param in filt_.parameters():
                    param.requires_grad = True
            

            for disc_ in self.discriminators:
                for param in disc_.parameters():
                    param.requires_grad = False
    def to_device(self, device):
        for idx in range(len(self.user_tags)):
            self.filters[idx].to(device)
            self.discriminators[idx].to(device)


class sDRO(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1
        self.n_layers = args.n_layers
        self.n_items = data.n_items
        self.n_users = data.n_users

        # 0 niche, 1 diverse, 2 blockbuster
        self._group_labels = [0, 1, 2]
        self._group_labels_matrix = torch.reshape(torch.tensor([0, 1, 2]), (-1, 1)).cuda(self.device)
        self._num_groups = 3
        self._group_reweight_strategy = 'loss-dro'
        self._dro_temperature = args.dro_temperature
        self._streaming_group_loss_lr = args.str_lr

        self._group_weights = torch.tensor([1/3, 1/3, 1/3], requires_grad = False).cuda(self.device)
        self._group_loss = torch.tensor([0.0, 0.0, 0.0], requires_grad = False).cuda(self.device)


    def _compute_group_loss(self, sample_loss, group_identity):

        group_mask = torch.eq(group_identity, self._group_labels_matrix).type(torch.LongTensor).cuda(self.device)
        group_cnts = torch.sum(group_mask, axis=1)
        group_cnts += (group_cnts == 0).long()
        group_loss = torch.divide(torch.sum(group_mask * sample_loss, axis=1), group_cnts)

        return group_loss, group_mask


    def forward(self, users, pos_items, neg_items, users_group):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        # print(neg_emb.shape)

        users_emb = F.normalize(users_emb, dim = -1)
        pos_emb = F.normalize(pos_emb, dim = -1)
        neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        if len(neg_emb.shape)>2:
            neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                           neg_emb.permute(0, 2, 1)).squeeze(dim=1)
            ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)
        else:
            neg_ratings = torch.sum(torch.mul(users_emb, neg_emb), dim=1) 
            ratings = torch.cat([pos_ratings, neg_ratings], dim=0)

        #分子
        numerator = torch.exp(pos_ratings / self.tau)
        #分母
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1) if len(neg_emb.shape)>2 else torch.sum(torch.exp(ratings / self.tau), dim = 0)
        ssm_loss = torch.negative(torch.log(numerator/denominator))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        # sDRO part
        cur_group_loss, group_mask = self._compute_group_loss(ssm_loss, users_group)
        # Note: only update loss/metric estimations when subgroup exists in a batch.
        # group_exist_in_batch: [num_groups], bool
        group_exist_in_batch = (torch.sum(group_mask, axis=1) > 1e-16).type(torch.LongTensor).cuda(self.device)

        # Perform streaming estimation of group loss.
        stream_group_loss = (1 - group_exist_in_batch * self._streaming_group_loss_lr) \
                            * self._group_loss.data + self._streaming_group_loss_lr * cur_group_loss
        self._group_loss.data = stream_group_loss

        # update group weight
        m = nn.Softmax(dim=-1)
        new_group_weights = m(torch.log(self._group_weights.data) + self._dro_temperature * self._group_loss.data)
        self._group_weights.data = new_group_weights

        sDRO_loss = torch.sum(self._group_weights.data * cur_group_loss) * self.batch_size

        return sDRO_loss, reg_loss

class sDRO_batch(sDRO):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.neg_sample =  self.batch_size-1

    def forward(self, users, pos_items, users_group):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        
        #分子
        numerator = torch.exp(ratings_diag / self.tau)
        #分母
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        ssm_loss = torch.negative(torch.log(numerator/denominator))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2
        regularizer = regularizer
        reg_loss = self.decay * regularizer

        # sDRO part
        cur_group_loss, group_mask = self._compute_group_loss(ssm_loss, users_group)
        # Note: only update loss/metric estimations when subgroup exists in a batch.
        # group_exist_in_batch: [num_groups], bool
        group_exist_in_batch = (torch.sum(group_mask, axis=1) > 1e-16).type(torch.LongTensor).cuda(self.device)

        # Perform streaming estimation of group loss.
        stream_group_loss = (1 - group_exist_in_batch * self._streaming_group_loss_lr) \
                            * self._group_loss.data + self._streaming_group_loss_lr * cur_group_loss
        self._group_loss.data = stream_group_loss

        # update group weight
        m = nn.Softmax(dim=-1)
        new_group_weights = m(torch.log(self._group_weights.data) + self._dro_temperature * self._group_loss.data)
        self._group_weights.data = new_group_weights

        sDRO_loss = torch.sum(self._group_weights.data * cur_group_loss) * self.batch_size

        return sDRO_loss, reg_loss

class MLP(nn.Module):

    def __init__(self, input_size = 128, output_size = 64):
        super(MLP , self).__init__()
        self.layer = nn.Linear(  input_size  , output_size  )
        
    def forward(self, x):  
        scores   = self.layer(x)
        return scores

class CDAN(LGN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.tau = args.tau
        self.neg_sample =  self.batch_size-1
        self.hidden_size = args.hidden_size
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.bias = args.bias

        self.item_prop = MLP(self.emb_dim, self.emb_dim)
        self.item_pop = MLP(self.emb_dim, self.emb_dim)
        self.item_final = MLP(2*self.emb_dim, self.emb_dim)
        
        self.item_prop.cuda(self.device)
        self.item_pop.cuda(self.device)
        self.item_final.cuda(self.device)

        self.user_pop_idx = data.user_pop_idx
        self.item_pop_idx = data.item_pop_idx
        self.n_users_pop = data.n_user_pop
        self.n_items_pop = data.n_item_pop
        self.embed_user_pop = nn.Embedding(self.n_users_pop, self.emb_dim)
        self.embed_item_pop = nn.Embedding(self.n_items_pop, self.emb_dim)
        nn.init.xavier_normal_(self.embed_user_pop.weight)
        nn.init.xavier_normal_(self.embed_item_pop.weight)
    
    def compute_p(self):

        users_emb = self.embed_user_pop.weight[self.user_pop_idx]
        items_emb = self.embed_item_pop.weight[self.item_pop_idx]
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])

        return users, items

    def infonce_loss(self, users, pos_items, userEmb0, posEmb0, pos_weights, flag = False, batch_size = 1024):

        users_emb = F.normalize(users, dim=1)
        pos_emb = F.normalize(pos_items, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        
        numerator = torch.exp(ratings_diag / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)

        if flag :
            maxi = torch.mul(torch.log(torch.log(numerator/denominator)), pos_weights)
            ssm_loss = torch.mean(torch.negative(maxi))

        else:
            ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2  +  0.5 * torch.norm(posEmb0) ** 2 
        regularizer = regularizer
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss 

    def bpr_loss(self, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0):
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss
    
     
    def disentangle(self, pos_items, pos_items_pop):
        # get property and popularity branch of items
        item_prop = self.item_prop(pos_items)
        item_pop = self.item_pop(pos_items)

        item_prop = F.normalize(item_prop, dim=1)
        item_pop = F.normalize(item_pop, dim=1)
        pos_items_pop = F.normalize(pos_items_pop, dim=1)

        pos_sim = self.batch_size - torch.sum(torch.mul(item_pop, pos_items_pop ), dim=1)  # users, pos_items, neg_items have the same shape
        orthogonal = torch.sum(torch.square(torch.mul(item_pop, item_prop)), dim=1)

        return torch.sum(pos_sim + orthogonal)

    def forward(self, users, pos_items, users_pop, pos_items_pop, next_pos_item, pos_weights, neg_items, neg_items_pop):

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        userEmb0_p = self.embed_user_pop(users_pop)
        posEmb0_p = self.embed_item_pop(pos_items_pop)
        negEmb0_p = self.embed_item_pop(neg_items_pop)

        all_users, all_items = self.compute()
        all_users_p, all_items_p = self.compute_p()

        users_pop = all_users_p[users]
        pos_items_pop = all_items_p[pos_items]
        neg_items_pop = all_items_p[neg_items]
        #users_pop = userEmb0_p
        #pos_items_pop = posEmb0_p
        
        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]

        # get property and popularity branch of items

        # item_prop = self.item_prop(pos_items)
        # item_pop = self.item_pop(pos_items)

        item_prop_pos = self.item_prop(pos_items)
        item_prop_neg = self.item_prop(neg_items)

        item_pop_pos = self.item_pop(pos_items)
        item_pop_neg = self.item_pop(neg_items)

        
        # unbias
        # unbias_loss, unbias_reg = self.infonce_loss(users, item_prop, userEmb0, posEmb0, pos_weights)
        unbias_loss, unbias_reg = self.bpr_loss(users, item_prop_pos, item_prop_neg, userEmb0, posEmb0, negEmb0)

        # bias ()
        # item_final = self.item_final(torch.cat((item_prop, pos_items_pop), -1))
        # bias_loss, bias_reg = self.infonce_loss(users, item_final, userEmb0, posEmb0, pos_weights)

        # bpr loss 
        item_final_pos = self.item_final(torch.cat((item_prop_pos, pos_items_pop), -1))
        item_final_neg = self.item_final(torch.cat((item_prop_neg, neg_items_pop), -1))
        bias_loss, bias_reg = self.bpr_loss(users, item_final_pos,item_final_neg, userEmb0, posEmb0, negEmb0)
            
        # disentangled
        # dis_loss = self.lambda1 * self.disentangle(pos_items, pos_items_pop)
        dis_loss = 0.5 * self.lambda1 * self.disentangle(pos_items, pos_items_pop) + 0.5 * self.lambda1 * self.disentangle(neg_items, neg_items_pop)
        # long tail
        #next_item_prop = self.item_prop(all_items[next_pos_item])

        #long_tail_loss, long_tail_reg = self.lambda2 * self.infonce_loss(next_item_prop, item_prop, posEmb0, posEmb0, pos_weights, flag = True)
        
        #long_tail_loss  = self.lambda2 * (self.long_tail_loss + long_tail_reg)
        return unbias_loss, bias_loss, dis_loss, unbias_reg , bias_reg

    def predict(self, users, items =  None):

        if self.bias == 0:
            return self.predict_unbias(users, items)
        else:
            return self.predict_bias(users, items)
    
    def predict_unbias(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]

        items = all_items[torch.tensor(items).cuda(self.device)]
        items_prop = self.item_prop(items)
        items_prop = torch.transpose(items_prop, 0, 1)
        rate_batch = torch.matmul(users, items_prop)

        return rate_batch.cpu().detach().numpy()

    def predict_bias(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        all_users, all_items = self.compute()
        all_users_pop, all_items_pop = self.compute_p()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items_pop = all_items_pop[torch.tensor(items).cuda(self.device)]

        items = all_items[torch.tensor(items).cuda(self.device)]
        items_prop = self.item_prop(items)
        
        items_final = self.item_final(torch.cat((items_prop, items_pop), -1))
        items_final = torch.transpose(items_final, 0, 1)
        
        rate_batch = torch.matmul(users, items_final)

        return rate_batch.cpu().detach().numpy()
    
    
class CDAN_MF(CDAN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.neg_sample = args.neg_sample 
    
    def infonce_loss(self, users, pos_items, neg_items, pos_weights, flag = False, batch_size = 1024):

        users_emb = F.normalize(users, dim = -1)
        pos_emb = F.normalize(pos_items, dim = -1)
        neg_emb = F.normalize(neg_items, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)
 
        numerator = torch.exp(pos_ratings / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)

        if flag:
            maxi = torch.mul(torch.log(torch.log(numerator/denominator)), pos_weights)
            ssm_loss = torch.mean(torch.negative(maxi))

        else:
            ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))


        regularizer = 0.5 * torch.norm(users_emb) ** 2 + 0.5 * torch.norm(pos_emb) ** 2 + 0.5 ** torch.norm(neg_emb)
        regularizer = regularizer / batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss 
 
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop, next_pos_item, pos_weights):

        
        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        userEmb0_p = self.embed_user_pop(users_pop)
        posEmb0_p = self.embed_item_pop(pos_items_pop)
        negEmb0_p = self.embed_item_pop(neg_items_pop)

        all_users, all_items = self.compute()
        all_users_p, all_items_p = self.compute_p()

        users_pop = all_users_p[users]
        pos_items_pop = all_users_p[pos_items]
        neg_items_pop = all_users_p[neg_items]

        users_pop = userEmb0_p
        pos_items_pop = posEmb0_p 
        neg_items_pop = negEmb0_p

        users = all_users[users]
        pos_items = all_items[pos_items]
        neg_items = all_items[neg_items]

        # get property and popularity branch of items
        item_prop = self.item_prop(pos_items)
        neg_item_prop = self.item_prop(neg_items)
        item_pop = self.item_pop(pos_items)
        neg_item_pop_mlp = self.item_pop(neg_items)
        
        # unbias
        unbias_loss, unbias_reg = self.infonce_loss(users, item_prop, neg_item_prop, pos_weights)

        # bias 
        item_final = self.item_final(torch.cat((item_prop, pos_items_pop), -1))
        neg_item_final = self.item_final(torch.cat((neg_item_prop, neg_items_pop), -1))
        bias_loss, bias_reg = self.infonce_loss(users, item_final, neg_item_final, pos_weights)
        
        # disentangled
        dis_loss = self.lambda1 * self.disentangle(pos_items, pos_items_pop)
        # long tail
        #next_item_prop = self.item_prop(self.embed_item(next_pos_item))

        #long_tail_loss, long_tail_reg =  self.infonce_loss(next_item_prop, item_prop, neg_item_prop, pos_weights, flag = True)
        #long_tail_loss = self.lambda2 * (long_tail_loss + long_tail_reg)

        return unbias_loss, bias_loss, dis_loss, unbias_reg, bias_reg
   
class CDAN_test(CDAN):
    def __init__(self, args, data):
        super().__init__(args, data)
        self.neg_sample = args.neg_sample 

        self.item_prop = MLP(self.emb_dim, self.emb_dim)
        
        self.item_prop.cuda(self.device)
    
    
    def infonce_loss(self, users, pos_items, neg_items, pos_weights, flag = False, batch_size = 1024):

        users_emb = F.normalize(users, dim = -1)
        pos_emb = F.normalize(pos_items, dim = -1)
        neg_emb = F.normalize(neg_items, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), neg_emb.permute(0, 2, 1)).squeeze(dim=1)
        ratings = torch.cat([pos_ratings[:, None], neg_ratings], dim=1)
 
        numerator = torch.exp(pos_ratings / self.tau)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)

        if flag:
            maxi = torch.mul(torch.log(torch.log(numerator/denominator)), pos_weights)
            ssm_loss = torch.mean(torch.negative(maxi))

        else:
            ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))


        regularizer = 0.5 * torch.norm(users_emb) ** 2 + 0.5 * torch.norm(pos_emb) ** 2 + 0.5 ** torch.norm(neg_emb)
        regularizer = regularizer / batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss 
 
    def forward(self, users, pos_items, neg_items, users_pop, pos_items_pop, neg_items_pop, pos_weights):

        users = self.embed_user(users)
        pos_items = self.embed_item(pos_items)
        neg_items = self.embed_item(neg_items)

        #users_pop = self.embed_user_pop(users_pop)
        #pos_items_pop = self.embed_item_pop(pos_items_pop)
        neg_items_pop = self.embed_item_pop(neg_items_pop)

        # get property and popularity branch of items
        item_prop = self.item_prop(pos_items)
        neg_item_prop = self.item_prop(neg_items)
        #item_pop = self.item_pop(pos_items)
        #neg_item_pop_mlp = self.item_pop(neg_items)
        
        # unbias
        unbias_loss, unbias_reg = self.infonce_loss(users, item_prop, neg_item_prop, pos_weights)

        return unbias_loss, unbias_reg

    # UNBIAS
    
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.n_items))

        users = self.embed_user(torch.tensor(users).cuda(self.device))
        items = self.embed_item(torch.tensor(items).cuda(self.device))

        items_prop = self.item_prop(items)
        items_prop = torch.transpose(items_prop, 0, 1)
        rate_batch = torch.matmul(users, items_prop)

        return rate_batch.cpu().detach().numpy()