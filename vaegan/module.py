"""VAE/GAN modules."""

import torch
from torch import nn, Tensor, empty_like, cat, exp # pylint:disable=no-name-in-module
from torch.nn import functional as F

from .utils import idx2onehot


class Encoder(nn.Module):
    """VAE Encoder, cat(sample,onehot)-FC1-ReLU-FC21/FC22-μ/logσ."""

    def __init__(self):
        super().__init__() # pyright:ignore[reportUnknownMemberType]

        self.training = True

        self.fc1  = nn.Linear(794, 400)
        self.fc21 = nn.Linear(400,  20)
        self.fc22 = nn.Linear(400,  20)
        self.relu = nn.ReLU()

    def _encode(self, sample: Tensor, idx_class: Tensor, idx_alt_class: None | Tensor, alpha: float):
        """cat(sample,onehot)-FC1-ReLU, then -FC21 for μ and -FC22 for logσ.
        
        Args:
            sample        :: (B, Feat) - Samples
            idx_class     :: (B,)      - Class indice
            idx_alt_class              - Alternative class index for breanding
            alpha                      - Class index blending rate
        """

        # Conditioning :: (B,) -> (B, Feat)
        emb_class = idx2onehot(idx_class, 10, idx_alt_class, alpha)
        con = cat((sample, emb_class), dim=-1)

        # Encode:: (B, Feat=794) -> (B, Feat=400) -> (B, Feat=20) x2
        h1 = self.relu(self.fc1(con))
        mu, logvar = self.fc21(h1), self.fc22(h1)

        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor):
        """Reparameterization trick, z = μ + n * exp(logσ), n ~ N(0,1)"""
        if self.training:
            eps = empty_like(mu).normal_()
            z_q = mu + eps * exp(logvar * 0.5)
            return z_q
        else:
            return mu

    def forward(self, sample: Tensor, idx_class: Tensor, idx_alt_class: None | Tensor = None, alpha: float = 1.):
        """
        Args:
            sample        :: (B, 1, X=28, Y=28) - Sample
            idx_class     :: (B,)               - Class label 
            idx_alt_class ::                    - Alternative class index for breanding
            alpha                               - Index brending rate
        Returns:
            z_q           :: (B, Feat=20, 1, 1) - Latent
            mu            :: (B, Feat=20, 1, 1) - μ    distribution parameter
            logvar        :: (B, Feat=20, 1, 1) - logσ distribution parameter
        """

        # Reshape :: (B, 1, X=28, Y=28) -> (B, Feat=784) - Flatten
        sample = sample.view(-1,28*28)

        # Parameterization :: (B, Feat=784) -> (B, Feat=20) x2
        mu, logvar = self._encode(sample, idx_class, idx_alt_class, alpha)

        # Sampling :: (B, Feat=20) x2 -> (B, Feat=20)
        z_q = self.reparameterize(mu, logvar)

        # Reshape :: (B, Feat=20) -> (B, Feat=20, 1, 1)
        z_q    =    z_q.unsqueeze(-1).unsqueeze(-1)
        mu     =     mu.unsqueeze(-1).unsqueeze(-1)
        logvar = logvar.unsqueeze(-1).unsqueeze(-1)

        return z_q, mu, logvar


class Decoder(nn.Module):
    """VAE Decoder, cat(z,onehot)-FC-ReLU-FC-σ."""
    def __init__(self):
        super().__init__() # pyright:ignore[reportUnknownMemberType]

        self.fc3 = nn.Linear( 30, 400)
        self.fc4 = nn.Linear(400, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, latent: Tensor, idx_class: Tensor, idx_alt_class: None | Tensor = None, alpha: float = 1.):
        """
        Args:
            latent        :: (B, Feat=20, 1, 1) - Latent variables
            idx_class     :: (B,)               - Class index
            idx_alt_class ::                    - Alternative class index for breanding
            alpha                               - Index brending rate
        """

        # Reshape :: (B, Feat=20, 1, 1) -> (B, Feat=20)
        latent = latent.view(-1, 20)

        # Conditioning :: (B, Feat=20) & (B,) -> (B, Feat=30)
        emb_class = idx2onehot(idx_class, 10, idx_alt_class, alpha)
        latent_cond = cat((latent, emb_class), dim=-1)

        # Transform :: (B, Feat=30) -> (B, Feat=784)
        sample_pred = self.sigmoid(self.fc4(self.relu(self.fc3(latent_cond))))

        return sample_pred


class Discriminator(nn.Module):
    """Discriminator, [FC-LReLU-Do]x3-FC-σ."""
    def __init__(self):
        super().__init__() # pyright:ignore[reportUnknownMemberType]


        do_rate = 0.3
        c_0, c_1, c_2, c_3 = 794, 1024, 512, 256
        feat_cond = 10

        # Conditioning :: (B,) -> (B, Feat=10)
        self.label_emb = nn.Embedding(10, feat_cond)

        # [FC-LReLU-Do]x3 :: (B, Feat=794) -> (B, Feat=256)
        self.model = nn.Sequential(
            # L1
            nn.Linear(c_0, c_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(do_rate),
            # L2
            nn.Linear(c_1, c_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(do_rate),
            # L3
            nn.Linear(c_2, c_3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(do_rate),
        )
        # FC-σ :: (B, Feat=256) -> (B, Feat=1)
        self.main = nn.Sequential(
            nn.Linear(c_3, 1),
            nn.Sigmoid()
        )

    def forward(self, sample: Tensor, class_label: Tensor):
        """
        Args:
            sample      :: ()            - Sample, real or fake
            class_label :: ()            - Class label
        Returns:
            feat        :: (B, Feat=256) - Intermediate features for VAE D-feature reconstruction loss
            d_sample    :: (B, Feat=  1) - Discrimination
        """

        # Reshape :: (B, ...) -> (B, Feat=784) - Flatten
        sample = sample.view(sample.size(0), 784)

        # Conditioning :: -> (B, Feat=10) -> (B, Feat=784+10=794)
        cond = self.label_emb(class_label)
        sample = cat([sample, cond], 1)

        # Discrimination
        feat = self.model(sample)
        d_sample = self.main(feat)

        return feat, d_sample


def loss_function(recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor, bsz: int = 100):
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
