import random
import re
from sys import get_coroutine_origin_tracking_depth
from sys import exit
import matplotlib.pyplot as plt
random.seed(101)
import math
# from scipy.linalg import svd
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
from model import INV_LGN_DUAL, LGN, INV_LGN_DUAL_BCE
from torch.utils.data import Dataset, DataLoader
from utils import *
from torch.utils.tensorboard import SummaryWriter
import networkx as nx
from t_sne_visualization import * 
from copy import deepcopy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def train_step():
    return


def adaptive_step():
    return


if __name__ == '__main__':

    start = time.time()

    args = parse_args()
    data = Data(args)
    data.load_data()
    device = torch.device(args.cuda)
    saveID = args.saveID

    saveID += "n_layers=" + str(args.n_layers)
    base_path = './weights/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)
    run_path = './runs/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)
    image_path = './image/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)

    checkpoint_buffer = []
    ensureDir(base_path)
    ensureDir(run_path)
    ensureDir(image_path)

    with open(base_path +'stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(str(args) + "\n")

    with open(base_path +'stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(str(args) + "\n")

    writer = SummaryWriter(log_dir=run_path)


    p_item = np.array([len(data.train_item_list[u]) if u in data.train_item_list else 0 for u in range(data.n_items)])
    p_user = np.array([len(data.train_user_list[u]) if u in data.train_user_list else 0 for u in range(data.n_users)])
    m_user = np.argmax(p_user)

    np.save("pop_user", p_user)
    np.save("pop_item", p_item)

    pop_sorted = np.sort(p_item)
    n_groups = 3
    grp_view = []
    for grp in range(n_groups):
        split = int((data.n_items - 1) * (grp + 1) / n_groups)
        grp_view.append(pop_sorted[split])
    print("group_view:", grp_view)
    item_pop_grp_idx = np.searchsorted(grp_view, p_item)

    

    pop_sorted = np.sort(p_user)
    grp_view = []
    for grp in range(n_groups):
        split = int((data.n_users - 1) * (grp + 1) / n_groups)
        grp_view.append(pop_sorted[split])
    print("group_view:", grp_view)
    user_pop_grp_idx = np.searchsorted(grp_view, p_user)
    

    # eval_test_ood_split = split_grp_view(grp_view, data.test_ood_user_list, idx)
    # eval_test_id_split = split_grp_view(grp_view, data.test_id_user_list, idx)

    grp_view = [0] + grp_view

    pop_dict = {}
    for user, items in data.train_user_list.items():
        for item in items:
            if item not in pop_dict:
                pop_dict[item] = 0
            pop_dict[item] += 1

    sort_pop = sorted(pop_dict.items(), key=lambda item: item[1], reverse=True)
    pop_mask = [item[0] for item in sort_pop[:20]]
    #print(pop_mask)
    if "douban" in args.dataset:
        top_ks=[30,20,20]
    elif "yelp" in args.dataset:
        top_ks = [20,20,20]
    else:
        top_ks=[5,3,3]
    print("top Ks : ", top_ks)

    if not args.pop_test:
        eval_test_ood = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list,top_k=[top_ks[0]],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_id_user_list]),user_neg_test=data.test_neg_user_list)
        eval_test_id = ProxyEvaluator(data,data.train_user_list,data.test_id_user_list,top_k=[top_ks[1]],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list]),user_neg_test=data.test_neg_user_list)
        eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[top_ks[2]],user_neg_test=data.test_neg_user_list)
        if 'coat' in args.dataset or 'yahoo' in args.dataset:
            eval_valid=ProxyEvaluator(data,data.train_user_list,data.test_id_user_list,top_k=[3],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list]),user_neg_test=data.test_neg_user_list)
    else:
        eval_test_ood = ProxyEvaluator(data, data.train_user_list, data.test_ood_user_list, top_k=[20],
                                       dump_dict=merge_user_list(
                                           [data.train_user_list, data.valid_user_list, data.test_id_user_list]),
                                       pop_mask=pop_mask)
        eval_test_id = ProxyEvaluator(data, data.train_user_list, data.test_id_user_list, top_k=[20],
                                      dump_dict=merge_user_list(
                                          [data.train_user_list, data.valid_user_list, data.test_ood_user_list]),
                                      pop_mask=pop_mask)
        eval_valid = ProxyEvaluator(data, data.train_user_list, data.valid_user_list, top_k=[20], pop_mask=pop_mask)

    evaluators = [eval_valid, eval_test_id, eval_test_ood]
    eval_names = ["valid", "test_id", "test_ood"]

    if args.modeltype == 'INV_LGN_DUAL':
        model = INV_LGN_DUAL(args, data,writer)
    if args.modeltype == 'LGN':
        model = LGN(args, data)
    if args.modeltype == 'INV_LGN_DUAL_BCE':
        model = INV_LGN_DUAL_BCE(args, data,writer)
    #    b=args.sample_beta
    model.cuda(device)

    model, start_epoch = restore_checkpoint(model, base_path, device)

    model.item_tags.append(torch.from_numpy(item_pop_grp_idx))
    model.user_tags.append(torch.from_numpy(user_pop_grp_idx))

    if args.test_only:

        for i, evaluator in enumerate(evaluators):
            is_best, temp_flag = evaluation(args, data, model, start_epoch, base_path, evaluator, eval_names[i])

        exit()

    flag = False

    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=model.lr)


    adv_optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=args.adv_lr)
    mask_optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=args.adv_lr)

    for epoch in range(start_epoch, args.epoch):
        # If the early stopping has been reached, restore to the best performance model
        # if flag:
        #     break
        running_loss, running_mf_loss, running_reg_loss, running_inv_loss, num_batches = 0, 0, 0, 0, 0
        if (epoch + 1)  % args.interval == 0:

            print("start adversarial training...")
            model.warmup = False
            
            avg_inv_loss_adp, num_batches_adp = 0, 0

            best_avg_inv = -np.inf

            cur_adv_patience=0

            epoch_adv = 0 
            #model.M.reset_parameters()
            #while cur_adv_patience < args.adv_patience:
            if args.draw_t_sne:
                    visualiza_embed(model, image_path, epoch, 0)

            for epoch_adv in range(args.adv_epochs):

                t1 = time.time()
                pbar = tqdm(enumerate(data.train_loader), total=len(data.train_loader))
                #print("embed_user grad before", model.embed_user.weight.requires_grad)
                model.freeze_args(True)
                #print("embed_user grad after", model.embed_user.weight.requires_grad)

                # adaptive mask step

                
                for batch_i, batch in pbar:
                    batch = [x.cuda(device) for x in batch]
                    if 'SEQ' in args.modeltype:
                        users = batch[0]
                        items = batch[1]
                        labels = batch[2].float()
                    else:
                        users = batch[0]
                        pos_items = batch[1]
                        users_pop = batch[2]
                        pos_items_pop = batch[3]
                        pos_weights = batch[4]
                        neg_items = batch[5]
                        neg_items_pop = batch[6]

                    model.train()

                    my_grad = model.forward_ARM()
                    mask = model.get_mask(True)
                    
                    if 'SEQ' in args.modeltype:
                        _, _, inv_loss = model(users, items, labels)
                    else:
                        _, _, inv_loss = model(users, pos_items, neg_items, is_draw=True, is_cluster = False)
                    inv_loss = 0


                    # loss = -inv_loss
                    adv_optimizer.zero_grad()
                    mask.backward(my_grad,retain_graph=True)
                    # print("grad: ",my_grad)
                    # print("inv loss: ",inv_loss)
                    # print(model.M.Q.weight.grad)
                    # loss.backward()
                    adv_optimizer.step()
                    model.step()
                    if args.use_new_mask_inv:
                        adv_optimizer.zero_grad()
                        loss = None 
                        for u_index in range(len(model.user_tags)):
                            if loss is None:
                                loss = model.compute_cluster_loss(mask,u_index)[0]
                            else:
                                loss +=  model.compute_cluster_loss(mask,u_index)[0]
                        loss = 0 - loss 
                        mask_optimizer.zero_grad()
                        loss.backward()
                        mask_optimizer.step()



                    
                    # avg_inv_loss_adp += inv_loss.detach().item()
                    avg_inv_loss_adp += 0
                    num_batches_adp += 1

                t2 = time.time()
                perf_str = 'Adv Epoch %d [%.1fs]: adjust avg inv == %.5f' % (
                    epoch_adv, t2 - t1,  avg_inv_loss_adp / num_batches_adp)

                epoch_adv += 1 
                cur_adv_patience+=1
                
                # # if (avg_inv_loss_adp / num_batches_adp) > best_avg_inv:
                # #     cur_adv_patience=0
                # #     best_avg_inv = avg_inv_loss_adp / num_batches_adp
                # #     save_checkpoint_adv(model, epoch, base_path)
            
                with open(base_path + 'stats_{}.txt'.format(args.saveID), 'a') as f:
                    f.write(perf_str + "\n")
            if args.draw_graph:
                mask = model.get_mask(True).detach().cpu().numpy()
                for u_idx in range(5):
                    for i_idx in range(5):
                        for start in ['user','item']:
                            plt.rcParams['figure.figsize']=(12.8, 7.2)
                            G, edge_labels,new_mask = model.draw_graph_init(mask,start)
                            G, labels = model.add_node_tag(G, user_index=u_idx, item_index=i_idx)
                            #pos = nx.nx_agraph.graphviz_layout(G, prog="neato")


                            user_val=list(model.user_tags[u_idx])#[0,1,1,0] ==> [0,2,3,1]
                            item_val=list(model.item_tags[i_idx])

                            pos_user=np.argsort(np.argsort(np.array(user_val)))#[0,3,1,2] [0,1,2,3] 
                            pos_item=np.argsort(np.argsort(np.array(item_val)))

                            pos = {}
                            pos.update((i, (1, 3*pos_user[i])) for i in range(model.n_users))
                            pos.update((i+model.n_users, (150, 3*pos_item[i])) for i in range(model.n_items))
                            #print(pos_user)
                            #print(['r' if val==0 else 'b' for val in user_val])#[]

                            nx.draw_networkx_nodes(G, pos, node_size=3, node_shape = 'd', nodelist = list(np.arange(model.n_users)),  node_color= user_val,cmap=plt.cm.bwr)
                            nx.draw_networkx_nodes(G, pos, node_size=3, node_shape = 'o', nodelist = list(np.arange(model.n_users,model.n_users+ model.n_items)), node_color=item_val, cmap=plt.cm.bwr)
                            nx.draw_networkx_edges(G,pos,edge_color=new_mask,width=0.5,
                                            edge_cmap=plt.cm.bwr)

                    
                            # nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", labels = labels)
                            # nx.draw_networkx_edge_labels(G, pos, edge_labels,font_size=5)
                            ax = plt.gca()
                            ax.margins(0.08)
                            plt.axis("off")
                            # plt.tight_layout()
                            plt.savefig(image_path+f'/u_index_{u_idx}_i_index{i_idx}_epoch_{epoch}_{start}.png')
                            plt.close() 



            ##model = restore_checkpoint_adv(model, base_path, device)


        t1 = time.time()
        pbar = tqdm(enumerate(data.train_loader), total=len(data.train_loader))
        model.freeze_args(False)
        # training step
        for batch_i, batch in pbar:
            batch = [x.cuda(device) for x in batch]
            if 'SEQ' in args.modeltype:
                users = batch[0]
                items = batch[1]
                labels = batch[2].float()
            else:
                users = batch[0]
                pos_items = batch[1]
                users_pop = batch[2]
                pos_items_pop = batch[3]
                pos_weights = batch[4]
                neg_items = batch[5]
                neg_items_pop = batch[6]

            model.train()
            #print(mf_loss.requires_grad)
            #print(reg_loss.requires_grad)
            if 'SEQ' in args.modeltype:
                mf_loss, reg_loss, inv_loss = model(users, items, labels)
            else:
                mf_loss, reg_loss, inv_loss = model(users, pos_items, neg_items)
            loss = mf_loss + reg_loss +  inv_loss

            # print(torch.cuda.memory_allocated(model.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            running_reg_loss += reg_loss.detach().item()
            running_mf_loss += mf_loss.detach().item()
            running_inv_loss += inv_loss.detach().item() 

            num_batches += 1

        t2 = time.time()

        # Training data for one epoch
        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
            epoch, t2 - t1, running_loss / num_batches,
            running_mf_loss / num_batches, running_reg_loss / num_batches,
            running_inv_loss / num_batches)

        with open(base_path + 'stats_{}.txt'.format(args.saveID), 'a') as f:
            f.write(perf_str + "\n")

        # Evaluate the trained model
        if (epoch + 1) % args.verbose == 0:
            model.eval()
            for i, evaluator in enumerate(evaluators):
                is_best, temp_flag = evaluation(args, data, model, epoch, base_path, evaluator, eval_names[i])

                if is_best:
                    checkpoint_buffer = save_checkpoint(model, epoch, base_path, checkpoint_buffer, args.max2keep)

                if temp_flag:
                    flag = True
            
            if args.modeltype == "INV_LGN_DUAL":
                predict_bias=model.get_predict_bias()
                perf_str = f"current predict bias:{predict_bias} \n"
                print(perf_str)
                with open(base_path + 'stats_{}.txt'.format(args.saveID), 'a') as f:
                    f.write(perf_str + "\n")

            model.train()
        

    # Get result
    model = restore_best_checkpoint(data.best_valid_epoch, model, base_path, device)
    print_str = "The best epoch is % d" % data.best_valid_epoch
    with open(base_path + 'stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(print_str + "\n")

    for i, evaluator in enumerate(evaluators[:]):
        evaluation(args, data, model, epoch, base_path, evaluator, eval_names[i])
    with open(base_path + 'stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(print_str + "\n")
