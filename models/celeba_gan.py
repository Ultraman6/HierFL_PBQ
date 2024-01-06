import copy

import torch
import torch.nn as nn
import torch.optim as optim
from typing_extensions import override


# Generator Network
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1)


class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, noise, real_images):
        # Generate fake images from noise
        fake_images = self.generator(noise)
        # Get discriminator outputs for both real and fake images
        real_output = self.discriminator(real_images)
        fake_output = self.discriminator(fake_images.detach())
        return real_output, fake_output, fake_images

    @override
    def state_dict(self):
        weights_d = self.discriminator.cpu().state_dict()
        weights_g = self.generator.cpu().state_dict()
        weights = {"generator": weights_g, "discriminator": weights_d}
        return copy.deepcopy(weights)

    @override
    def load_state_dict(self, state_dict):
        if 'generator' in state_dict and 'discriminator' in state_dict:
            self.generator.load_state_dict(copy.deepcopy(state_dict['generator']))
            self.discriminator.load_state_dict(copy.deepcopy(state_dict['discriminator']))
        else:
            raise KeyError("State dict does not contain the expected keys: 'generator' and 'discriminator'")



def synchronize_gan_parameters(clients):
    # 初始化生成器和判别器的参数
    init_gen_parameters = list(clients[0].model.shared_layers.sharegenerator.parameters())
    init_dis_parameters = list(clients[0].model.shared_layers.discriminator.parameters())

    for client in clients:
        # 获取每个客户端的生成器和判别器的参数
        user_gen_parameters = list(client.model.shared_layers.generator.parameters())
        user_dis_parameters = list(client.model.shared_layers.discriminator.parameters())

        # 同步生成器参数
        for init_param, user_param in zip(init_gen_parameters, user_gen_parameters):
            user_param.data[:] = init_param.data[:]

        # 同步判别器参数
        for init_param, user_param in zip(init_dis_parameters, user_dis_parameters):
            user_param.data[:] = init_param.data[:]




# Create the generator and discriminator
netG = Generator()
netD = Discriminator()

# Loss function and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

