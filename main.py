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
from utils import *
from data import Data
from parse import parse_args
from model import CausE, IPS, LGN, MACR, INFONCE_batch, INFONCE, SAMREG, BC_LOSS, BC_LOSS_batch, SimpleX, SimpleX_batch, INV_LGN_DUAL, CVIB, CVIB_SEQ, DR, LGN_BCE, DR_SEQ
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

def compute_IPS(user_index, pos_item_index, neg_item_index, device,y_ips=None):
    item_index = torch.cat((pos_item_index, neg_item_index),0)
    pos_label = torch.ones((len(pos_item_index),))
    neg_label = torch.zeros((len(neg_item_index),))
    y = torch.cat((pos_label, neg_label),0) 

    if y_ips is None:
        one_over_zl = torch.ones(len(y))
    else:
        py1 = y_ips.sum() / len(y_ips)
        #print("py1:",py1.device)
        py0 = 1 - py1
        #print("py0:",py0.device)
        po1 = len(user_index) / (max(user_index) * max(item_index))
        py1o1 = sum(y) / len(y)
        #print("py1o1:",py1o1.device)
        py0o1 = 1 - py1o1
        #print("py0o1:",py0o1.device)

        propensity = torch.zeros(len(y)).cuda(device)

        propensity[y == 0] = (py0o1 * po1) / py0
        propensity[y == 1] = (py1o1 * po1) / py1
        one_over_zl = 1 / (propensity + 1e-6)

    return one_over_zl



def get_rotation_matrix(axis, theta):
    """
    Find the rotation matrix associated with counterclockwise rotation
    about the given axis by theta radians.
    Credit: http://stackoverflow.com/users/190597/unutbu

    Args:
        axis (list): rotation axis of the form [x, y, z]
        theta (float): rotational angle in radians

    Returns:
        array. Rotation matrix.
    """

    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

grads = {}
def save_grad(name):
    def hook(grad):
        torch.clamp(grad, -1, 1)
        grads[name] = grad
    return hook

def visulization(items,users,data,p_item,p_user,name):
    test_ood_user_list=data.test_ood_user_list
    test_id_user_list=data.test_id_user_list
    train_user_list=data.train_user_list

    def split_grp_view(data,grp_idx):
        n=len(grp_view)
        split_data=[collections.defaultdict(list) for _ in range(n)]

        for key,item in data.items():
            for it in item:
                if key not in split_data[grp_idx[it]].keys():
                    split_data[grp_idx[it]][key]=[]
                split_data[grp_idx[it]][key].append(it)
        return split_data

    pop_sorted=np.sort(p_item)
    n_items=p_item.shape[0]

    n_groups=3
    grp_view=[]
    for grp in range(n_groups):
        split=int((n_items-1)*(grp+1)/n_groups)
        grp_view.append(pop_sorted[split])
    #print("group_view:",grp_view)
    idx=np.searchsorted(grp_view,p_item)

    pop_group=[[] for _ in range(n_groups)]
    for i in range(n_items):
        pop_group[idx[i]].append(i)

    eval_test_ood_split=split_grp_view(test_ood_user_list,idx)
    eval_test_id_split=split_grp_view(test_id_user_list,idx)
    eval_train_split=split_grp_view(train_user_list,idx)

    pop_users=p_user.tolist()

    u_pop_sorted=np.sort(p_user)
    print(u_pop_sorted[-10:])



    fig = plt.figure(constrained_layout=True,figsize=(12,6))


    def plot_embed(ax1,ax2,idx):
        u, v = np.mgrid[0:2*np.pi:20j, 0:2*np.pi:20j]
        x1 = np.cos(u)*np.sin(v)
        y1 = np.sin(u)*np.sin(v)
        z1 = np.cos(v)
        ax1.plot_wireframe(x1, y1, z1, color="0.5",linewidth=0.1)
        user_idx=pop_users.index(idx)
        m_user=users[user_idx]
        target=np.array([1,-1,1])
        r_theta=np.arccos(np.dot(m_user,target)/(np.linalg.norm(m_user)*np.linalg.norm(target)))
        axis=np.cross(m_user,target)
        R=get_rotation_matrix(axis,r_theta)
        grp_theta=[]
        grp_r=[]
        sizes=[10,10,10]

        cmap_b = 'b'
        cmap_r = 'r'
        cmaps=[cmap_b,cmap_r]

        norm = plt.Normalize(vmin=-3, vmax=3)

        all_sampled=set([])
        all_pos=set([])
        for i,grp in enumerate(pop_group):
            sampled_group=set(np.random.choice(np.array(grp),50,replace=False).tolist())
            if user_idx in eval_test_id_split[i].keys():
                for item in eval_test_id_split[i][user_idx]:
                    sampled_group.add(item)
                    all_pos.add(item)
            for item in eval_train_split[i][user_idx]:
                sampled_group.add(item)
                all_pos.add(item)
            if user_idx in eval_test_ood_split[i].keys():
                for item in eval_test_ood_split[i][user_idx]:
                    sampled_group.add(item)
                    all_pos.add(item)
            
            all_sampled=all_sampled.union(sampled_group)
            

        all_neg=all_sampled.difference(all_pos)
        #print(all_neg)
        all_pos=np.array(list(all_pos),dtype=int)
        all_neg=np.array(list(all_neg),dtype=int)
        nor = plt.Normalize(vmin=-3, vmax=3)
        r=np.linalg.norm(target)

        lab=["neg","pos"]
        for i,idx in enumerate([all_neg,all_pos]):
            g_item=items[idx]
            g_item=np.matmul(g_item,R.T)
            norm=np.linalg.norm(g_item,axis=1)
            x=g_item[:,0]/norm#*r
            y=g_item[:,1]/norm#*r
            z=g_item[:,2]/norm#*r

            for j in range(len(idx)):
                ax1.plot([0,g_item[j][0]/norm[j]],[0,g_item[j][1]/norm[j]],[0,g_item[j][2]/norm[j]],color = cmaps[i],alpha=0.1)

            ax1.scatter(x, y, z, c = cmaps[i], marker =".",s=10,label=lab[i])

        
        #print("V^{T}",V_transpose)

        ax1.scatter(target[0]/r, target[1]/r, target[2]/r, c = 'g', marker ="*",s=120,label="user")
        ax1.plot([0,target[0]/r],[0,target[1]/r],[0,target[2]/r],color = 'g',alpha=0.1)
        ax1.legend()

        

        all_items=set([i for i in range(n_items)])
        all_neg=all_items.difference(all_pos)
        all_neg=np.array(list(all_neg),dtype=int)

        grp=["(neg):","(pos):"]
        txt=""

        for i,idx in enumerate([all_neg,all_pos]):
            g_item=items[idx]
            g_item=np.matmul(g_item,R.T)
            norm=np.linalg.norm(g_item,axis=1)
            cos=np.arccos(np.matmul(target,g_item.T)/norm/r)
            me=float(np.mean(cos))
            me=round(me,3)
            if i==1:
                txt="mean angle"+grp[i]+str(me)+"\n"+txt
            else:
                txt="mean angle"+grp[i]+str(me)+txt
            
            ax2.hist(cos,50,range=[0,np.pi],color=cmaps[i],weights=np.zeros_like(cos) + 1. / cos.size,edgecolor='black',alpha=0.6)
            mi_x,ma_x=ax2.get_xlim()
            mi_y,ma_y=ax2.get_ylim()
        ax2.text(mi_x+(ma_x-mi_x)*0.45, mi_y+(ma_y-mi_y)*0.75,txt , style ='italic') 
        red_patch = mpatches.Patch(color='red', alpha=0.6, label='pos')
        blue_patch = mpatches.Patch(color='blue', alpha=0.6,label='neg')
        ax2.legend(handles=[red_patch,blue_patch])
        
    
    pops=[205,30,10]


    fig = plt.figure(figsize=(6,8),constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0:2, 0:2],projection='3d')
    ax2 = fig.add_subplot(gs[2,0:2])
    ax1.set_xticks([-1,-0.5,0,0.5,1])
    ax1.set_yticks([-1,-0.5,0,0.5,1])
    ax1.set_zticks([-1,-0.5,0,0.5,1])
    ax1.grid(False)

    plot_embed(ax1,ax2,pops[0])
    #ax1.set_title("High Pop User(p=205)")
    #ax2.set_title("Angular Distribution(High Pop)")

    plt.savefig(name+"high_pop_"+str(pops[0])+".png",bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(6,8),constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax3 = fig.add_subplot(gs[0:2, 0:2],projection='3d')
    ax4 = fig.add_subplot(gs[2,0:2])
    ax3.set_xticks([-1,-0.5,0,0.5,1])
    ax3.set_yticks([-1,-0.5,0,0.5,1])
    ax3.set_zticks([-1,-0.5,0,0.5,1])
    ax3.grid(False)
    plot_embed(ax3,ax4,pops[1])
    #ax3.set_title("Mid Pop User(p=30)")
    #ax4.set_title("Angular Distribution(Mid Pop)")

    plt.savefig(name+"mid_pop_"+str(pops[1])+".png",bbox_inches='tight')
    plt.close()



    fig = plt.figure(figsize=(6,8),constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax5 = fig.add_subplot(gs[0:2, 0:2],projection='3d')
    ax6 = fig.add_subplot(gs[2,0:2])
    ax5.set_xticks([-1,-0.5,0,0.5,1])
    ax5.set_yticks([-1,-0.5,0,0.5,1])
    ax5.set_zticks([-1,-0.5,0,0.5,1])
    ax5.grid(False)
    plot_embed(ax5,ax6,pops[2])
    #ax5.set_title("Low Pop User(p=10)")
    #ax6.set_title("Angular Distribution(Low Pop)")

    plt.savefig(name+"low_pop_"+str(pops[2])+".png",bbox_inches='tight')
    plt.close()




if __name__ == '__main__':

    start = time.time()
    args = parse_args()
    print(str(args))

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
    
    run_path = './runs/{}/{}/{}'.format(args.dataset, args.modeltype, saveID)
    ensureDir(base_path)
    ensureDir(run_path)

    with open(base_path +'stats_{}.txt'.format(args.saveID), 'a') as f:
        f.write(str(args) + "\n")


    writer = SummaryWriter(log_dir=run_path)

    p_item = np.array([len(data.train_item_list[u]) if u in data.train_item_list else 0 for u in range(data.n_items)])
    p_user = np.array([len(data.train_user_list[u]) if u in data.train_user_list else 0 for u in range(data.n_users)])
    m_user=np.argmax(p_user)
    
    np.save("pop_user",p_user)
    np.save("pop_item",p_item)
    
    # pop_sorted=np.sort(p_item)
    # n_groups=3
    # grp_view=[]
    # for grp in range(n_groups):
    #     split=int((data.n_items-1)*(grp+1)/n_groups)
    #     grp_view.append(pop_sorted[split])
    # print("group_view:",grp_view)
    # idx=np.searchsorted(grp_view,p_item)

    # eval_test_ood_split=split_grp_view(grp_view,data.test_ood_user_list,idx)
    # eval_test_id_split=split_grp_view(grp_view,data.test_id_user_list,idx)

    # grp_view=[0]+grp_view

    # pop_dict={}
    # for user,items in data.train_user_list.items():
    #     for item in items:
    #         if item not in pop_dict:
    #             pop_dict[item]=0
    #         pop_dict[item]+=1
    
    # sort_pop=sorted(pop_dict.items(), key=lambda item: item[1],reverse=True)
    # pop_mask=[item[0] for item in sort_pop[:20]]
    # print(pop_mask)

    if not args.pop_test:
        eval_test_ood = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_id_user_list]),user_neg_test=data.test_neg_user_list)
        eval_test_id = ProxyEvaluator(data,data.train_user_list,data.test_id_user_list,top_k=[30],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list]),user_neg_test=data.test_neg_user_list)
        eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[20],user_neg_test=data.test_neg_user_list)
        if 'coat' in args.dataset or 'yahoo' in args.dataset:
            eval_valid=ProxyEvaluator(data,data.train_user_list,data.test_id_user_list,top_k=[5],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list]),user_neg_test=data.test_neg_user_list)
    else:
        eval_test_ood = ProxyEvaluator(data,data.train_user_list,data.test_ood_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_id_user_list]),pop_mask=pop_mask)
        eval_test_id = ProxyEvaluator(data,data.train_user_list,data.test_id_user_list,top_k=[20],dump_dict=merge_user_list([data.train_user_list,data.valid_user_list,data.test_ood_user_list]),pop_mask=pop_mask)
        eval_valid = ProxyEvaluator(data,data.train_user_list,data.valid_user_list,top_k=[20],pop_mask=pop_mask)

    evaluators=[ eval_valid,eval_test_id, eval_test_ood]
    eval_names=["valid","test_id", "test_ood" ]

    if args.modeltype == 'LGN':
        model = LGN(args, data)
    if args.modeltype == 'INFONCE':
        model = INFONCE(args, data)
    if args.modeltype == 'INFONCE_batch':
        model = INFONCE_batch(args, data)
    if args.modeltype == 'IPS':
        model = IPS(args, data)
    if args.modeltype == 'CausE':
        model = CausE(args, data)
    if args.modeltype == 'BC_LOSS':
        model = BC_LOSS(args, data)
    if args.modeltype == 'BC_LOSS_batch':
        model = BC_LOSS_batch(args, data)
    if args.modeltype == 'MACR':
        model = MACR(args, data)
    if args.modeltype == 'SAMREG':
        model = SAMREG(args, data)
    if args.modeltype == "SimpleX":
        model = SimpleX(args,data)
    if args.modeltype == "SimpleX_batch":
        model = SimpleX_batch(args,data)
    if args.modeltype == 'INV_LGN_DUAL':
        model = INV_LGN_DUAL(args, data, writer)
    if args.modeltype == 'CVIB':
        model = CVIB(args, data)
    if args.modeltype == 'CVIB_SEQ':
        model = CVIB_SEQ(args, data)
    if args.modeltype == 'DR':
        model = DR(args, data)
    if args.modeltype == 'DR_SEQ':
        model = DR_SEQ(args, data)
    if args.modeltype == 'LGN_BCE':
        model = LGN_BCE(args, data)
    if args.modeltype == 'IPS_SEQ':
        model = IPS_SEQ(args, data)

    
#    b=args.sample_beta

    model.cuda(device)

    model, start_epoch = restore_checkpoint(model, base_path, device)

    if args.test_only:

        for i,evaluator in enumerate(evaluators):
            is_best, temp_flag = evaluation(args, data, model, start_epoch, base_path, evaluator,eval_names[i])

        exit()
                

    flag = False
    if args.modeltype == 'INV_LGN_DUAL':
        model.freeze_args(False)


    optimizer = torch.optim.Adam([ param for param in model.parameters() if param.requires_grad == True], lr=model.lr)

    #item_pop_idx = torch.tensor(data.item_pop_idx).cuda(device)
    for epoch in range(start_epoch, args.epoch):

        # If the early stopping has been reached, restore to the best performance model
        if flag:
            break

        # All models
        running_loss, running_mf_loss, running_reg_loss, num_batches, running_bce_loss, running_info_loss, running_ips_loss, running_direct_loss = 0, 0, 0, 0, 0, 0, 0, 0
        # CausE
        running_cf_loss = 0
        # BC_LOSS
        running_loss1, running_loss2 = 0, 0

        # INV_LGN

        running_inv_loss = 0

        t1=time.time()

        pbar = tqdm(enumerate(data.train_loader), total = len(data.train_loader))
        
        if args.modeltype == 'CVIB' or args.modeltype == 'DR' or args.modeltype == 'CVIB_SEQ':
            model.shuffle()

        for batch_i, batch in pbar:            
            batch = [x.cuda(device) for x in batch]

            if 'SEQ' in args.modeltype:
                users = batch[0]
                items = batch[1]
                labels = batch[2].float()
            else:
                users = batch[0]
                pos_items = batch[1]

                if args.modeltype != 'CausE':
                    users_pop = batch[2]
                    pos_items_pop = batch[3]
                    pos_weights = batch[4]
                    if args.infonce == 0 or args.neg_sample != -1:
                        neg_items = batch[5]
                        neg_items_pop = batch[6]
                if 'DR' in args.modeltype:
                    prior_y = batch[7]
                    one_over_zl = compute_IPS(users, pos_items, neg_items, device, torch.FloatTensor(model.data.train_data.y_ips).cuda(device))


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
            
            elif args.modeltype == "INV_LGN_DUAL":
                if args.remove_inv == 0:
                    mf_loss, reg_loss, inv_loss = model(users,pos_items,neg_items)
                    loss = mf_loss + reg_loss + inv_loss 
                else:
                    mf_loss, reg_loss, _ = model(users,pos_items,neg_items)
                    loss = mf_loss + reg_loss 
                
            elif args.modeltype == "CVIB":
                x_sampled = model.all_samples[model.ul_idxs[batch_i*args.batch_size:(batch_i+1)*args.batch_size]]
                sampled_user = torch.LongTensor(x_sampled[:,0])
                sampled_items = torch.LongTensor(x_sampled[:,1])
                bce_loss, info_loss = model(users,pos_items,neg_items,sampled_user,sampled_items)
                loss = bce_loss + info_loss
            elif args.modeltype == "CVIB_SEQ":
                x_sampled = model.all_samples[model.ul_idxs[batch_i*args.batch_size:(batch_i+1)*args.batch_size]]
                sampled_user = torch.LongTensor(x_sampled[:,0])
                sampled_items = torch.LongTensor(x_sampled[:,1])
                bce_loss, info_loss = model(users,items,labels,sampled_user,sampled_items)
                loss = bce_loss + info_loss
            elif "DR" in args.modeltype :
                x_sampled = model.all_samples[model.ul_idxs[batch_i*args.batch_size:(batch_i+1)*args.batch_size]]
                sampled_user = torch.LongTensor(x_sampled[:,0])
                sampled_items = torch.LongTensor(x_sampled[:,1])
                inv_prop = one_over_zl
                imputation_y = prior_y.float()
                if "SEQ" in args.modeltype:
                    ips_loss, direct_loss = model(users, items, labels, sampled_user, sampled_items,inv_prop.cuda(device), imputation_y.cuda(device))
                    loss = ips_loss + direct_loss
                else:
                    ips_loss, direct_loss = model(users,pos_items, neg_items, sampled_user, sampled_items,inv_prop.cuda(device), imputation_y.cuda(device))
                    loss = ips_loss + direct_loss
            elif args.modeltype == "IPS_SEQ": 
                inv_prop = one_over_zl[batch_i*args.batch_size:(batch_i+1)*args.batch_size]
                bce_loss, reg_loss = model(users, items, labels, inv_prop.cuda(device))
                loss = bce_loss + reg_loss
            else:
                mf_loss, reg_loss = model(users, pos_items, neg_items)
                loss = mf_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.modeltype == "INV_LGN_DUAL":
                model.step()

            running_loss += loss.detach().item()
            if args.modeltype != "CVIB" and args.modeltype !=  "CVIB_SEQ" and args.modeltype !=  "DR":
                running_reg_loss += reg_loss.detach().item()
                
            if args.modeltype != 'BC_LOSS' and args.modeltype != 'BC_LOSS_batch' and args.modeltype != "CVIB" and args.modeltype != "CVIB_SEQ" and args.modeltype != "DR" and 'SEQ' not in args.modeltype:
                running_mf_loss += mf_loss.detach().item()
            
            if args.modeltype == 'CausE':
                running_cf_loss += cf_loss.detach().item()

            if args.modeltype == 'BC_LOSS' or args.modeltype == 'BC_LOSS_batch':
                running_loss1 += loss1.detach().item()
                running_loss2 += loss2.detach().item()

            if args.remove_inv == 0 and args.modeltype == "INV_LGN_DUAL":
                running_inv_loss += inv_loss.detach().item()
                # if args.modeltype == "INV_LGN_DUAL":
                #     running_inv_loss += inv_loss.detach().item()
            if args.modeltype == "CVIB" or args.modeltype == "CVIB_SEQ":
                running_bce_loss += bce_loss.detach().item()
                running_info_loss += info_loss.detach().item()
            if args.modeltype == "LGN_SEQ":
                running_bce_loss += bce_loss.detach().item()
                running_reg_loss += reg_loss.detach().item()
            if args.modeltype == "DR":
                running_ips_loss += ips_loss.detach().item()
                running_direct_loss += direct_loss.detach().item()
            if args.modeltype == "IPS_SEQ":
                running_bce_loss += bce_loss.detach().item()
                running_reg_loss += reg_loss.detach().item()


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
            
        elif args.modeltype=="INV_LGN_DUAL":
            if args.remove_inv == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                epoch, t2 - t1, running_loss / num_batches,
                running_mf_loss / num_batches, running_reg_loss / num_batches, running_inv_loss / num_batches)
            else:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, t2 - t1, running_loss / num_batches,
                    running_mf_loss / num_batches, running_reg_loss / num_batches)
                # + %.5f
                #, running_inv_loss / num_batches

        elif args.modeltype=="CVIB" or args.modeltype =="CVIB_SEQ":
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, t2 - t1, running_loss / num_batches,
                running_bce_loss / num_batches, running_info_loss / num_batches)
        elif args.modeltype=="DR":
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, t2 - t1, running_loss / num_batches,
                running_ips_loss / num_batches, running_direct_loss / num_batches)
        elif args.modeltype=="LGN_SEQ":
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, t2 - t1, running_loss / num_batches,
                running_bce_loss / num_batches, running_reg_loss / num_batches)
        elif args.modeltype=="IPS_SEQ":
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, t2 - t1, running_loss / num_batches,
                running_bce_loss / num_batches, running_reg_loss / num_batches)
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
    # ndcg_res = ndcg_func(model, x_test, y_test, device)
    # print("ndcg_5: ", np.mean(ndcg_res["ndcg_5"]))
    # print("ndcg_10: ",  np.mean(ndcg_res["ndcg_10"]))

    # print_str = "The best epoch is % d" % data.best_valid_epoch
    # with open(base_path +'stats_{}.txt'.format(args.saveID), 'a') as f:
    #     f.write(print_str + "\n")

    for i,evaluator in enumerate(evaluators[:]):
        evaluation(args, data, model, epoch, base_path, evaluator, eval_names[i])
    # with open(base_path +'stats_{}.txt'.format(args.saveID), 'a') as f:
    #     f.write(print_str + "\n")



