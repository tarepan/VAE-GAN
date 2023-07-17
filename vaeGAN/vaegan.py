import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from utils import idx2onehot


class VAE(nn.Module):
    """VAE Encoder, cat(sample,onehot)-FC1-ReLU-FC21/FC22-μ/logσ."""

    def __init__(self):
        super(VAE, self).__init__()

        self.fc1  = nn.Linear(794, 400)
        self.fc21 = nn.Linear(400,  20)
        self.fc22 = nn.Linear(400,  20)
        self.relu = nn.ReLU()

    def encode(self, x, y, label, alpha):
        """cat(sample,onehot)-FC1-ReLU, then -FC21 for μ and -FC22 for logσ.
        
        Args:
            x :: (B, Feat) - Sample
            y ::           -
        """

        y = idx2onehot(y, 10, label, alpha)
        con = torch.cat((x, y), dim=-1)
        # :: (..., feat=28*28+10=794) -> (..., feat=400) -> (..., feat=20)
        h1 = self.relu(self.fc1(con))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick, z = μ + n * exp(logσ), n ~ N(0,1)"""
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def forward(self, x, y, label=None, alpha = 1):
       """
       Args:
        x     :: - Sample
        y     :: - Class label 
        label :: -
        alpha
       Returns:
        mu     :: (B, Feat=20, 1, 1) - μ    distribution parameter
        logvar :: (B, Feat=20, 1, 1) - logσ distribution parameter
       """

       # Parameterization :: (...) -> (B, Feat=28*28=784) -> (B, Feat=20) x2
       mu, logvar = self.encode(x.view(-1,28*28), y, label, alpha)

       # Sampling :: (B, Feat=20) x2 -> (B, Feat=20)
       # NOTE: Not used
       z = self.reparameterize(mu, logvar)

       # Reshape :: (B, Feat=20) -> (B, Feat=20, 1, 1)
       mu.unsqueeze_(-1)
       mu.unsqueeze_(-1)
       logvar.unsqueeze_(-1)
       logvar.unsqueeze_(-1)

       return mu, logvar


class Aux(nn.Module):
    """VAE Decoder, cat(z,onehot)-FC-ReLU-FC-σ."""
    def __init__(self):
        super(Aux,self).__init__()

        self.fc3 = nn.Linear( 30, 400)
        self.fc4 = nn.Linear(400, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def decode(self,z, y, label, alpha):
        """
        Args:
            z :: (B, Feat=20, 1, 1) - Latent variables
            y :: ()                 - Digit label
        """

        # Reshape :: (B, Feat=20, 1, 1) -> (B, Feat=20)
        z = z.view(-1, 20)

        # Conditioning :: (B, Feat=20) & () -> (B, Feat=30)
        y_c = idx2onehot(y, 10, label, alpha)
        cat = torch.cat((z, y_c), dim=-1)

        # Transform :: (B, Feat=30) -> (B, Feat=784)
        ret = self.sigmoid(self.fc4(self.relu(self.fc3(cat))))

        ret = torch.narrow(ret, 1, 0, 784)
        return ret
    
    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps
        else:
          return mu

    def forward(self,z, y, label=None, alpha = 1):
        """
        Args:
            z :: (B, Feat=20, 1, 1) - Latent variables
            y :: ()                 - Digit label
        """
        #z = self.reparameterize(mu,logvar)
        return self.decode(z, y, label, alpha)


class NetD(nn.Module):
    """Discriminator."""
    def __init__(self):
        super(NetD, self).__init__()
        
        self.label_emb = nn.Embedding(10, 10)

        do_rate = 0.3
        c_0, c_1, c_2, c_3 = 794, 1024, 512, 256

        # FC-LReLU-Do-FC-LReLU-Do-FC-LReLU-Do :: (B, Feat=794) -> (B, Feat=256)
        self.model = nn.Sequential(
            nn.Linear(c_0, c_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(do_rate),
            nn.Linear(c_1, c_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(do_rate),
            nn.Linear(c_2, c_3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(do_rate),
        )
        # -FC-σ :: (B, Feat=256) -> (B, Feat=1)
        self.main = nn.Sequential(
            nn.Linear(c_3, 1),
            nn.Sigmoid()
        )

    def forward(self,x, y):
        """
        Args:
        Returns:
            out :: (B, Feat=256) - Intermediate features for VAE D-feature L2 loss
            dl  :: (B, Feat=  1) - Discrimination
        """
        # Reshape :: (B, ...) -> (B, Feat=784)
        x = x.view(x.size(0), 784)
        c = self.label_emb(y)
        x = torch.cat([x, c], 1)

        out = self.model(x)
        dl = self.main(out)

        return out, dl


def loss_function(recon_x, x, mu, logvar, bsz=100):
    """VAE loss = L2 loss + KL loss."""
    #BCE = F.binary_cross_entropy(recon_x.view(-1,784), x.view(-1, 784))
    MSE = F.mse_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= bsz * 784

    return MSE + KLD
