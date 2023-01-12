import torch.nn as nn
import torch.nn.functional as F
import torch




class Filter(nn.Module):
    def __init__(self, embed_dim, attribute='gender'):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.attribute = attribute
        self.W1 = nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True)
        self.W2 = nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True)
        self.batchnorm = nn.BatchNorm1d(self.embed_dim)

    def forward(self, ents_emb):
        h1 = F.leaky_relu(self.W1(ents_emb))
        h2 = F.leaky_relu(self.W2(h1))
        h2 = self.batchnorm(h2)
        return h2

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))



class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        # TODO: Modify M with Attention
        pass

    def forward(self):
        pass

