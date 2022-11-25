import random
import re
from sys import get_coroutine_origin_tracking_depth
from sys import exit
random.seed(101)
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
#from scipy.linalg import svd
import itertools
import torch
import time
import numpy as np
from tqdm import tqdm
from evaluator import ProxyEvaluator
import collections
import os
from data import Data
from parse import parse_args
from model import CausE, IPS, LGN, MACR, INFONCE_batch, INFONCE, SAMREG, BC_LOSS, BC_LOSS_batch, SimpleX, SimpleX_batch
from torch.utils.data import Dataset, DataLoader
from utils import *


if __name__ == '__main__':

    start = time.time()

    args = parse_args()
    data = Data(args)
    data.load_data()
    device="cuda:"+str(args.cuda)
    device = torch.device(args.cuda)
    saveID = args.saveID
    if args.modeltype == "INFONCE" or args.modeltype == 'INFONCE_batch':
        saveID += "n_layers=" + str(args.n_layers) + "tau=" + str(args.tau)
    if args.modeltype == "BC_LOSS" or args.modeltype == 'BC_LOSS_batch':
        saveID += "n_layers=" + str(args.n_layers) + "tau1=" + str(args.tau1) + "tau2=" + str(args.tau2) + "w=" + str(args.w_lambda)


    if args.n_layers == 2 and args.modeltype != "LGN":
        base_path = './weights/{}/{}-LGN/{}'.format(args.dataset, args.modeltype, saveID)
    else:
        base_path = './weights/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)

    if args.modeltype == 'LGN':
        saveID += "n_layers=" + str(args.n_layers)
        base_path = './weights/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)

    checkpoint_buffer=[]
    freeze_epoch=args.freeze_epoch if (args.modeltype=="BC_LOSS" or args.modeltype=="BC_LOSS_batch") else 0
    ensureDir(base_path)

    p_item = np.array([len(data.train_item_list[u]) if u in data.train_item_list else 0 for u in range(data.n_items)])
    p_user = np.array([len(data.train_user_list[u]) if u in data.train_user_list else 0 for u in range(data.n_users)])
    m_user=np.argmax(p_user)
    
    np.save("pop_user",p_user)
    np.save("pop_item",p_item)
    
    pop_sorted=np.sort(p_item)
    n_groups=3
    grp_view=[]
    for grp in range(n_groups):
        split=int((data.n_items-1)*(grp+1)/n_groups)
        grp_view.append(pop_sorted[split])
    print("group_view:",grp_view)
    idx=np.searchsorted(grp_view,p_item)

    eval_test_ood_split=split_grp_view(grp_view,data.test_ood_user_list,idx)
    eval_test_id_split=split_grp_view(grp_view,data.test_id_user_list,idx)

    grp_view=[0]+grp_view

    pop_dict={}
    for user,items in data.train_user_list.items():
        for item in items:
            if item not in pop_dict:
                pop_dict[item]=0
            pop_dict[item]+=1
    
    sort_pop=sorted(pop_dict.items(), key=lambda item: item[1],reverse=True)
    pop_mask=[item[0] for item in sort_pop[:20]]
    print(pop_mask)

    if not args.pop_test:
        eval_test_ood = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_id_user_list]))
        eval_test_id = ProxyEvaluator(data,data.train_user_list,data.test_id_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list]))
        eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[20])
    else:
        eval_test_ood = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_id_user_list]),pop_mask=pop_mask)
        eval_test_id = ProxyEvaluator(data,data.train_user_list,data.test_id_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list]),pop_mask=pop_mask)
        eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[20],pop_mask=pop_mask)

    evaluators=[ eval_valid,eval_test_id, eval_test_ood]
    eval_names=["valid","test_id", "test_ood" ]

    if args.modeltype == 'INV-LGN':
        model = LGN(args, data)
#    b=args.sample_beta

    model.cuda(device)

    model, start_epoch = restore_checkpoint(model, base_path, device)

    if args.test_only:

        for i,evaluator in enumerate(evaluators):
            is_best, temp_flag = evaluation(args, data, model, start_epoch, base_path, evaluator,eval_names[i])

        exit()
                

    flag = False
    
    optimizer = torch.optim.Adam([ param for param in model.parameters() if param.requires_grad == True], lr=model.lr)

    #item_pop_idx = torch.tensor(data.item_pop_idx).cuda(device)

    
    for epoch in range(start_epoch, args.epoch):

        # If the early stopping has been reached, restore to the best performance model
        if flag:
            break

        # All models
        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0
        # CausE
        running_cf_loss = 0
        # BC_LOSS
        running_loss1, running_loss2 = 0, 0

        t1=time.time()

        pbar = tqdm(enumerate(data.train_loader), total = len(data.train_loader))

        for batch_i, batch in pbar:            

            batch = [x.cuda(device) for x in batch]

            users = batch[0]
            pos_items = batch[1]

            if args.modeltype != 'CausE':
                users_pop = batch[2]
                pos_items_pop = batch[3]
                pos_weights = batch[4]
                if args.infonce == 0 or args.neg_sample != -1:
                    neg_items = batch[5]
                    neg_items_pop = batch[6]

            model.train()
         
            if args.modeltype == 'INFONCE_batch':

                mf_loss, reg_loss = model(users, pos_items)
                loss = mf_loss + reg_loss

            elif args.modeltype == 'INFONCE':

                mf_loss, reg_loss = model(users, pos_items, neg_items)
                loss = mf_loss + reg_loss
            
            elif args.modeltype == 'BC_LOSS_batch':
                loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm = model(users, pos_items, users_pop, pos_items_pop)
                
                if epoch < args.freeze_epoch:
                    loss =  loss2 + reg_loss_freeze
                else:
                    model.freeze_pop()
                    loss = loss1 + loss2 + reg_loss

            elif args.modeltype == 'BC_LOSS':
                loss1, loss2, reg_loss, reg_loss_freeze, reg_loss_norm  = model(users, pos_items, neg_items, \
                                                                                users_pop, pos_items_pop, neg_items_pop)
                
                if epoch < args.freeze_epoch:
                    loss =  loss2 + reg_loss_freeze
                else:
                    model.freeze_pop()
                    loss = loss1 + loss2 + reg_loss

            elif args.modeltype == 'IPS' or args.modeltype =='SAMREG':

                mf_loss, reg_loss = model(users, pos_items, neg_items, pos_weights)
                loss = mf_loss + reg_loss

            elif args.modeltype == 'CausE':
                neg_items = batch[2]
                all_reg = torch.squeeze(batch[3].T.reshape([1, -1]))
                all_ctrl = torch.squeeze(batch[4].T.reshape([1, -1]))
                mf_loss, reg_loss, cf_loss = model(users, pos_items, neg_items, all_reg, all_ctrl)
                loss = mf_loss + reg_loss + cf_loss 
            
            elif args.modeltype == "SimpleX":
                mf_loss, reg_loss = model(users, pos_items, neg_items)
                loss = mf_loss + reg_loss

            
            elif args.modeltype == "SimpleX_batch":
                mf_loss, reg_loss = model(users, pos_items)
                loss = mf_loss + reg_loss


            else:
                mf_loss, reg_loss = model(users, pos_items, neg_items)
                loss = mf_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            running_reg_loss += reg_loss.detach().item()

            if args.modeltype != 'BC_LOSS' and args.modeltype != 'BC_LOSS_batch':
                running_mf_loss += mf_loss.detach().item()
            
            if args.modeltype == 'CausE':
                running_cf_loss += cf_loss.detach().item()

            if args.modeltype == 'BC_LOSS' or args.modeltype == 'BC_LOSS_batch':
                running_loss1 += loss1.detach().item()
                running_loss2 += loss2.detach().item()

            num_batches += 1

        t2=time.time()

        # Training data for one epoch
        if args.modeltype == "CausE":
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                epoch, t2 - t1, running_loss / num_batches,
                running_mf_loss / num_batches, running_reg_loss / num_batches, running_cf_loss / num_batches)
        
        elif args.modeltype=="BC_LOSS" or args.modeltype=="BC_LOSS_batch":
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                epoch, t2 - t1, running_loss / num_batches,
                running_loss1 / num_batches, running_loss2 / num_batches, running_reg_loss / num_batches)

        else:
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, t2 - t1, running_loss / num_batches,
                running_mf_loss / num_batches, running_reg_loss / num_batches)

        with open(base_path + 'stats_{}.txt'.format(args.saveID),'a') as f:
            f.write(perf_str+"\n")

        # Evaluate the trained model
        if (epoch + 1) % args.verbose == 0 and epoch >= freeze_epoch:
            model.eval() 

            for i,evaluator in enumerate(evaluators):
                is_best, temp_flag = evaluation(args, data, model, epoch, base_path, evaluator,eval_names[i])
                
                if is_best:
                    checkpoint_buffer=save_checkpoint(model, epoch, base_path, checkpoint_buffer, args.max2keep)
                
                if temp_flag:
                    flag = True

            model.train()
        
    # Get result
    model = restore_best_checkpoint(data.best_valid_epoch, model, base_path, device)
    print_str = "The best epoch is % d" % data.best_valid_epoch
    with open(base_path +'stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(print_str + "\n")

    for i,evaluator in enumerate(evaluators[:]):
        evaluation(args, data, model, epoch, base_path, evaluator, eval_names[i])
    with open(base_path +'stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(print_str + "\n")
