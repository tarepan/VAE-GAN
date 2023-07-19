"""VAE/GAN modules."""

import torch
from torch import nn, Tensor, empty_like, cat, exp # pylint:disable=no-name-in-module
from torch.nn import functional as F


class Encoder(nn.Module):
    """VAE Encoder, cat(sample,onehot)-FC1-ReLU-FC21/FC22-μ/logσ."""

    def __init__(self):
        super().__init__() # pyright:ignore[reportUnknownMemberType]

        self.training = True

        self.fc1  = nn.Linear(794, 400)
        self.fc21 = nn.Linear(400,  20)
        self.fc22 = nn.Linear(400,  20)
        self.relu = nn.ReLU()

    def _encode(self, sample: Tensor, cond: Tensor):
        """cat(sample,onehot)-FC1-ReLU, then -FC21 for μ and -FC22 for logσ.
        
        Args:
            sample        :: (B, Feat) - Samples
            cond          :: (B, Feat) - Conditioning vector
        """

        # Conditioning :: (B,) -> (B, Feat)
        sample_cond = cat((sample, cond), dim=-1)

        # Encode:: (B, Feat=794) -> (B, Feat=400) -> (B, Feat=20) x2
        h1 = self.relu(self.fc1(sample_cond))
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

    def forward(self, sample: Tensor, cond: Tensor):
        """
        Args:
            sample :: (B, Feat=784) - Sample
            cond   :: (B, Feat= 10) - Conditioning vector
        Returns:
            z_q    :: (B, Feat=20) - Latent
            mu     :: (B, Feat=20) - Distribution parameter μ
            logvar :: (B, Feat=20) - Distribution parameter logσ
        """

        # Parameterization/Sampling :: (B, Feat=784) & (B, Feat=10) -> (B, Feat=20) x2 -> (B, Feat=20)
        mu, logvar = self._encode(sample, cond)
        z_q = self.reparameterize(mu, logvar)

        return z_q, mu, logvar


class Decoder(nn.Module):
    """VAE Decoder, cat(z,cond)-FC-ReLU-FC-σ."""
    def __init__(self):
        super().__init__() # pyright:ignore[reportUnknownMemberType]

        self.net = nn.Sequential(
            nn.Linear( 30, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )

    def forward(self, latent: Tensor, cond: Tensor):
        """
        Args:
            latent        :: (B, Feat=20) - Latent
            cond          :: (B, Feat=10) - Conditioning vector
        Returns:
                          :: (B, Feat=784=28*28)
        """
        return self.net(cat((latent, cond), dim=-1))


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
            sample      :: (B, Feat=784) - Sample, real or fake
            class_label :: (B,)          - Class label
        Returns:
            feat        :: (B, Feat=256) - Intermediate features for VAE D-feature reconstruction loss
            d_sample    :: (B, Feat=  1) - Discrimination
        """

        # Conditioning :: (B,) -> (B, Feat=10) -> (B, Feat=784+10=794)
        cond = self.label_emb(class_label)
        sample_cond = cat([sample, cond], 1)

        # Discrimination
        feat = self.model(sample_cond)
        d_sample = self.main(feat)

        return feat, d_sample


def reconstruction_loss(pred: Tensor, gt: Tensor) -> Tensor:
    """VAE reconstruction loss.
    
    Args:
        pred :: () - Prediction
        gt   :: () - Ground-truth
    Returns:
             :: (1,) - Loss
    """
    return F.mse_loss(pred, gt)


def kl_loss(mu: Tensor, logvar: Tensor, bsz: int = 100):
    """VAE KL loss."""

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(σ^2) - μ^2 - σ^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= bsz * 784

    return KLD
