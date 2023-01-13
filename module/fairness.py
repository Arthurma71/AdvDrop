import torch.nn as nn
import torch.nn.functional as F
import torch




class Filter(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.embed_size
        #self.attribute = attribute
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
    def __init__(self, args, tag):
        super().__init__()
        self.embed_dim = args.embed_size
        self.criterion = nn.NLLLoss()

        self.out_dim = torch.max(tag) + 1
        self.net = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim *2 ), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim*4),bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim*2),bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim*2), int(self.embed_dim*2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim*2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim/2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim / 2), self.out_dim,bias=True)
            )

    def forward(self,filter_embed,attr_label):
        scores = self.net(filter_embed)
        output = F.log_softmax(scores, dim=1)
        #print(attr_label.shape)
        #print(output.squeeze().shape)
        loss = self.criterion(output.squeeze(), attr_label)
        return loss
    

    def predict(self,filter_embed):
        scores = self.net(filter_embed)
        output = F.log_softmax(scores, dim=1)
        preds = output.max(1, keepdim=True)[1] # get the index of the max
        return preds





