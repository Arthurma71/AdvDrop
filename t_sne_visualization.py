from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from model import CausE, IPS, LGN, MACR, INFONCE_batch, INFONCE, SAMREG, BC_LOSS, BC_LOSS_batch, SimpleX, SimpleX_batch, INV_LGN_DUAL, CVIB, CVIB_SEQ, DR, LGN_BCE, DR_SEQ
from torch.utils.tensorboard import SummaryWriter
from parse import parse_args
from data import Data
import torch
from utils import *

color_list = ['r', 'b', 'y', 'g', 'c', 'k', 'm', 'teal', 'dodgerblue',
                      'indigo', 'deeppink', 'pink', 'peru', 'brown', 'lime', 'darkorange']

def reduce_embed_dim(model):
    o_embed = model.get_top_embeddings().detach().cpu().numpy()
    reduce_embed = TSNE(n_components=2, learning_rate='auto',
                   init='random', perplexity=30).fit_transform(o_embed)
    return reduce_embed

def visualiza_embed(model, image_path, epoch, epoch_adv):
    reduce_embed = reduce_embed_dim(model)
    for i in range(len(model.user_tags)):
        plt.figure()
        labels = model.user_tags[i]
        num_groups = max(labels)
        for j in range(num_groups+1):
            target = reduce_embed[labels == j]
            plt.plot(target[:, 0], target[:,1],
                         'o', color=color_list[j])

        plt.savefig(image_path+f'/epoch_{epoch}_adv_{epoch_adv}_attribute_{i}.png')

            


# if __name__ == '__main__':
#     args = parse_args()
#     data = Data(args)
#     data.load_data()
#     device = torch.device(args.cuda)
#     writer = SummaryWriter(log_dir='data') 

#     model = INV_LGN_DUAL(args, data, writer).to(device)
#     image_path = '/storage/pbwei/INV-LGN/image/{}/{}/{}'.format(args.dataset, args.modeltype, "saveID_no_attribute_Top")
#     ensureDir(image_path)
#     visualiza_embed(model, image_path,0,0)


