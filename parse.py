import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', nargs='?', default=-1,
                        help='we only want test value.')
    parser.add_argument('--test_only', nargs='?', default=False,
                        help='we only want test value.')
    parser.add_argument('--data_path', nargs='?', default='./data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='yahoo.new',
                        help='Choose a dataset')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate.')
    parser.add_argument('--regs', type=float, default=1e-5,
                        help='Regularization.')
    parser.add_argument('--epoch', type=int, default=1600,
                        help='Number of epoch.')
    parser.add_argument('--Ks', nargs='?', default= [3],
                        help='Evaluate on Ks optimal items.')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='log\'s interval epoch while training')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--saveID', type=str, default="",
                        help='Specify model save path.')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping point.')
    parser.add_argument('--checkpoint', type=str, default='./',
                        help='Specify model save path.')
    parser.add_argument('--modeltype', type=str, default= 'BC_LOSS',
                        help='Specify model save path.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Specify which gpu to use.')
    parser.add_argument('--IPStype', type=str, default='cn',
                        help='Specify the mode of weighting')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of GCN layers')
    parser.add_argument('--codetype', type=str, default='train',
                        help='Calculate overlap with Item pop')
    parser.add_argument('--max2keep', type=int, default=10,
                        help='max checkpoints to keep')
    parser.add_argument('--infonce', type=int, default=1,
                        help='whether to use infonce loss or not')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers in data loader')
    parser.add_argument('--neg_sample', type=int, default=1,
                        help='negative sample ratio.')
    parser.add_argument('--neg_test', type=bool, default=True,
                        help='whether to use neg items for testing')

    # MACR
    parser.add_argument('--alpha', type=float, default=1e-3,
                        help='alpha')
    parser.add_argument('--beta', type=float, default=1e-3,
                        help='beta')
    parser.add_argument('--c', type=float, default=30.0,
                        help='Constant c.')
    #CausE
    parser.add_argument('--cf_pen', type=float, default=0.05,
                        help='Imbalance loss.')
    
    #SAM-REG

    parser.add_argument('--rweight', type=float, default=0.05)
    parser.add_argument('--sam',type=bool,default=True)
    parser.add_argument('--pop_test',type=bool,default=False)

    #SimpleX

    parser.add_argument('--w_neg', type=float, default=1)
    parser.add_argument('--neg_margin',type=float, default=0.4)
    
    #BC_LOSS
    parser.add_argument('--tau1', type=float, default=0.07,
                        help='temperature parameter for L1')
    parser.add_argument('--tau2', type=float, default=0.1,
                        help='temperature parameter for L2')
    parser.add_argument('--w_lambda', type=float, default=0.5,
                        help='weight for combining l1 and l2.')
    parser.add_argument('--freeze_epoch',type=int,default=5)

    #INV-LGN
    parser.add_argument('--inv_tau', type=float, default=1,
                        help='temperature parameter for inv loss in train step')
    parser.add_argument('--adaptive_tau', type=float, default=1,
                        help='temperature parameter for rec loss in adaptive step')
    parser.add_argument('--gumble_tau', type=float, default=10,
                        help='temperature parameter for rec loss in adaptive step')
    parser.add_argument('--mask', type=int, default=0,
                        help='indicator for mask type')
    parser.add_argument('--att_dim', type=int, default=10,
                        help='attention dim')
    parser.add_argument('--pre_epochs', type=int, default=1,
                        help='warmup epochs')
    parser.add_argument('--interval', type=int, default=7,
                        help='normal training epoch before entering adversarial training in each cycle')
    parser.add_argument('--adv_epochs', type=int, default=5,
                        help='adversarial epochs')
    parser.add_argument('--adv_patience', type=int, default=3,
                        help='adversarial patience')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='No. samples in inv loss')
    parser.add_argument('--is_geometric', type=int, default=0,
                        help='using geometric package or not')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='keep prob for dropout')
    parser.add_argument('--embed_tau', type=float, default=0.1,
                        help='tau for embed-level infonce')
    parser.add_argument('--dropout_type', type=int, default=1,
                        help='dropout type for embed-level infonce 0 for random 1 for attention')
    parser.add_argument('--grad_coeff', type=float, default=5,
                        help='coefficient for ARM gradient')
    parser.add_argument('--inv_coeff', type=float, default=1,
                        help='coefficient for inv loss')
    parser.add_argument('--remove_inv', type=float, default=0,
                        help='0: not remove. 1: remove')
    parser.add_argument('--adv_lr',type=float,default=1e-2, help='lr for adv')

    parser.add_argument('--draw_graph',type=bool,default=False, help='draw graph or not')
    parser.add_argument('--use_attribute',type=bool,default=False, help='use attribute or not')


    # CVIB 
    parser.add_argument('--cvib_alpha', type=float, default=0.1,
                        help='alpha for CVIB')
    parser.add_argument('--cvib_gamma', type=float, default=0.01,
                        help='gamma for CVIB')




    
    
    return parser.parse_args()


