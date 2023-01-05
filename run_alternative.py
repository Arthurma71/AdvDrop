import random
import re
from sys import get_coroutine_origin_tracking_depth
from sys import exit

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

    checkpoint_buffer = []
    ensureDir(base_path)
    ensureDir(run_path)

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
    idx = np.searchsorted(grp_view, p_item)

    eval_test_ood_split = split_grp_view(grp_view, data.test_ood_user_list, idx)
    eval_test_id_split = split_grp_view(grp_view, data.test_id_user_list, idx)

    grp_view = [0] + grp_view

    pop_dict = {}
    for user, items in data.train_user_list.items():
        for item in items:
            if item not in pop_dict:
                pop_dict[item] = 0
            pop_dict[item] += 1

    sort_pop = sorted(pop_dict.items(), key=lambda item: item[1], reverse=True)
    pop_mask = [item[0] for item in sort_pop[:20]]
    print(pop_mask)

    if not args.pop_test:
        eval_test_ood = ProxyEvaluator(data, data.train_user_list, data.test_ood_user_list, top_k=[3],
                                       dump_dict=merge_user_list(
                                           [data.train_user_list, data.valid_user_list, data.test_id_user_list]),user_neg_test=data.test_neg_user_list)
        eval_test_id = ProxyEvaluator(data, data.train_user_list, data.test_id_user_list, top_k=[5],
                                      dump_dict=merge_user_list(
                                          [data.train_user_list, data.valid_user_list, data.test_ood_user_list]),user_neg_test=data.test_neg_user_list)
        eval_valid = ProxyEvaluator(data, data.train_user_list, data.valid_user_list, top_k=[20])
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

    if args.test_only:

        for i, evaluator in enumerate(evaluators):
            is_best, temp_flag = evaluation(args, data, model, start_epoch, base_path, evaluator, eval_names[i])

        exit()

    flag = False

    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=model.lr)


    adv_optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=args.adv_lr)
    #optimizer = torch.optim.SparseAdam([param for param in model.parameters() if param.requires_grad == True], lr=model.lr)
    

    for epoch in range(start_epoch, args.epoch):
        # If the early stopping has been reached, restore to the best performance model
        if flag:
            break
        running_loss, running_mf_loss, running_reg_loss, running_inv_loss, num_batches = 0, 0, 0, 0, 0
        if (epoch + 1)  % args.interval == 0:

            print("start adversarial training...")
            model.warmup = False
            
            avg_inv_loss_adp, num_batches_adp = 0, 0

            best_avg_inv = -np.inf

            cur_adv_patience=0

            epoch_adv = 0 
            model.M.reset_parameters()
            #while cur_adv_patience < args.adv_patience:
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
                        _, _, inv_loss = model(users, pos_items, neg_items, is_draw=True)


                    # loss = -inv_loss
                    adv_optimizer.zero_grad()
                    mask.backward(my_grad)
                    # print("grad: ",my_grad)
                    # print("inv loss: ",inv_loss)
                    # print(model.M.Q.weight.grad)
                    # loss.backward()
                    adv_optimizer.step()
                    model.step()

                    
                    avg_inv_loss_adp += inv_loss.detach().item()
                    num_batches_adp += 1

                t2 = time.time()
                perf_str = 'Adv Epoch %d [%.1fs]: adjust avg inv == %.5f' % (
                    epoch_adv, t2 - t1,  avg_inv_loss_adp / num_batches_adp)

                epoch_adv += 1 
                cur_adv_patience+=1
                
                # if (avg_inv_loss_adp / num_batches_adp) > best_avg_inv:
                #     cur_adv_patience=0
                #     best_avg_inv = avg_inv_loss_adp / num_batches_adp
                #     save_checkpoint_adv(model, epoch, base_path)
            
                with open(base_path + 'stats_{}.txt'.format(args.saveID), 'a') as f:
                    f.write(perf_str + "\n")

            #model = restore_checkpoint_adv(model, base_path, device)


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
