"Train VAE/GAN"

import torch
import torch.utils.data
from torch import nn, optim, empty_like, ones, zeros # pylint:disable=no-name-in-module
from torchvision import datasets, transforms # pyright:ignore[reportMissingTypeStubs]
from torchvision.utils import save_image     # pyright:ignore[reportMissingTypeStubs,reportUnknownVariableType]

from .module import Encoder, Decoder, Discriminator, loss_function
from .utils import idx2onehot


def main():
    """Train a VAE/GAN."""

    # Configs
    bsz = 128
    n_epoch = 200

    dataset_train = datasets.MNIST('../data', download=True,                transform=transforms.ToTensor())
    # dataset_test  = datasets.MNIST('../data',                train = False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=bsz, shuffle=True, drop_last=True) # pyright:ignore[reportUnknownVariableType]
    # test_loader  = torch.utils.data.DataLoader(dataset_test,  batch_size=bsz, shuffle=True, drop_last=True)

    encoder = Encoder()
    decoder = Decoder()
    disc    = Discriminator()
    criterion = nn.BCELoss()

    optim_disc = optim.Adam(disc.parameters(),    lr=1e-4)
    optim_enc  = optim.Adam(encoder.parameters(), lr=1e-4)
    optim_dec  = optim.Adam(decoder.parameters(), lr=1e-4)

    encoder, decoder = encoder.cuda(), decoder.cuda()
    disc = disc.cuda()
    criterion =criterion.cuda()

    _ones  = ones(bsz).cuda()
    _zeros = zeros(bsz).cuda()

    for epoch in range(n_epoch):
        print(f'start epoch #{epoch}')

        for i, (image, digit) in enumerate(train_loader):
            #### Step ################################################

            # Data
            ## real :: (B, 1, X=28, Y=28)
            real  = image.cuda()
            ## digit :: (B,)
            digit = digit.cuda()

            # Common_Forward
            ## Embedding
            cond = idx2onehot(digit, 10)
            ## Encode
            z_q, mu, logvar = encoder(real, cond)
            ## Prior sampling
            z_p = empty_like(mu).normal_()
            ## Decode
            fake_zq = decoder(z_q, cond)
            fake_zp = decoder(z_p, cond)

            # D_Forward
            d_feat_fake, d_fake_zq = disc(fake_zq, digit)
            d_feat_real, d_real    = disc(real,    digit)
            _,           d_fake_zp = disc(fake_zp, digit)
            # D_Loss/Backward/Optim
            loss_adv_d_real    = criterion(d_real.squeeze(1),    _ones)
            loss_adv_d_fake_zq = criterion(d_fake_zq.squeeze(1), _zeros)
            loss_adv_d_fake_zp = criterion(d_fake_zp.squeeze(1), _zeros)
            loss_disc = loss_adv_d_real + loss_adv_d_fake_zq + loss_adv_d_fake_zp
            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            optim_disc.step()

            # G_Loss/Backward/Optim, with adversarial encoder learning
            loss_adv_g_zq = criterion(d_fake_zq.squeeze(1), _ones)
            loss_adv_g_zp = criterion(d_fake_zp.squeeze(1), _ones)
            loss_vae = loss_function(d_feat_fake, d_feat_real, mu, logvar)
            loss_g = loss_adv_g_zq + loss_adv_g_zp + loss_vae
            encoder.zero_grad()
            decoder.zero_grad()
            loss_g.backward()
            optim_dec.step()
            optim_enc.step()

            # Logging
            if i % 2000 == 0:
                save_image(real,                          './results/cvaegan_results/train2_real_samples2.png', normalize=True)
                save_image(fake_zq.data.view(-1,1,28,28), './results/cvaegan_results/train2_fake_samples2.png', normalize=True)
            #### /Step ###############################################

        if epoch % 25 == 0:
            save_image(fake_zq.data.view(-1,1,28,28), f'./results/cvaegan_results/train2_fake_samples2_{epoch}.png', normalize=True) # pyright:ignore[reportUnboundVariable]

    # torch.save(encoder, './pretrained models/encoder3.pth')
    # torch.save(disc,    './pretrained models/disc3.pth')
    # torch.save(decoder, './pretrained models/decoder3.pth')


if __name__ == "__main__":
    main()
