"""
Title: WGAN Architecture and Training Program
Author: Enting Zhou
Date: 06/15/2022
Availability: https://github.com/ETZET/MCMC_GAN
"""
import os.path

import torch
from torch import nn
from torch.autograd import Variable
from torch import autograd
import time
import wandb
import matplotlib

matplotlib.use('Agg')


class WGAN_SIMPLE(nn.Module):
    """
    Generative Model Architecture

    Model Architecture cited from Scheiter, M., Valentine, A., Sambridge, M., 2022. Upscaling
    and downscaling Monte Carlo ensembles with generative models, Geophys. J. Int., ggac100.

    This model use gradient penalty to enforce 1-Lipschitz constraint.
    Citation: Gulrajani, Ahmed & Arjovsky. Improved training of wasserstein gans. Adv. Neural Inf. Process. Syst.
    """

    def __init__(self, ndim, nhid=200, nlatent=100, device="cpu"):
        """
        :param ndim: Number of feature in input data
        :param nhid: Number of hidden units per layer
        :param device: device on which a torch.Tensor is or will be allocated
        :param gen: Generator that consist of four layers of dropout layers with linear output
        :param disc: Discriminator that consist of four layers of dropout layers with linear output
        """
        super().__init__()

        self.ndim = ndim
        self.nlatent = nlatent
        self.device = device
        
        self.gen = nn.Sequential(
            nn.Linear(self.nlatent,nhid),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(nhid,nhid),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(nhid,nhid),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(nhid,ndim),
        )
        
        self.disc = nn.Sequential(
            nn.Linear(self.ndim,nhid),
            nn.LeakyReLU(0.1),
            nn.Linear(nhid,nhid),
            nn.LeakyReLU(0.1),
            nn.Linear(nhid,nhid),
            nn.LeakyReLU(0.1),
            nn.Linear(nhid,1),
        )
        
        self.gen.apply(init_weights)
        self.disc.apply(init_weights)
        
        self.gen.to(device)
        self.disc.to(device)
            

    def optimize(self, dataloader, output_path, use_wandb=False, lr=1e-4, beta1=0.5, lambda_term=10, epochs=200, kkd=1, kkg=1, device="cpu"):

        optimizer_gen = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizer_disc = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(beta1, 0.999))

        for epoch in range(epochs):
            for i, data in enumerate(dataloader):
                tic = time.time()
                # Update Discriminator
                for k in range(kkd):
                    optimizer_disc.zero_grad()

                    real_data = data.to(device).float()
                    b_size = real_data.size(0)
                    fake_data = self.gen(torch.randn(b_size, self.nlatent, device=device).float())

                    gradient_penalty = self.calculate_gradient_penalty(real_data, fake_data, lambda_term)
                    D_loss_real = torch.mean(self.disc(real_data))
                    D_loss_fake = torch.mean(self.disc(fake_data))
                    score_disc = -D_loss_real + D_loss_fake + gradient_penalty

                    score_disc.backward()
                    optimizer_disc.step()

                # Update Generator
                for k in range(kkg):
                    optimizer_gen.zero_grad()

                    real_data = data.to(device)
                    b_size = real_data.size(0)
                    fake_data = self.gen(torch.randn(b_size, self.nlatent, device=device))
                    score_gen = -torch.mean(self.disc(fake_data))
                    score_gen.backward()

                    optimizer_gen.step()

                toc = time.time()
                # logging
                if i % 5 == 0:
                    print(
                        '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t Wasserstein Distance: %.4f\t  Elapsed time per Iteration: %.4fs'
                        % (epoch, epochs, i, len(dataloader),
                           score_disc, score_gen, (D_loss_real - D_loss_fake), (toc - tic)))
                    if use_wandb:
                        wandb.log({'D_loss': score_disc, 'Wasserstein Distance': (D_loss_real - D_loss_fake),
                                   'G_loss': score_gen})
            # model saving
            model_save_path = os.path.join(output_path,"model")
            if not os.path.exists(model_save_path):
                os.mkdir(model_save_path)
            if epoch % 10 == 0 or epoch == epochs - 1:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                }, "{}/model_epoch{}.pth".format(model_save_path, epoch))

    def calculate_gradient_penalty(self, real_images, fake_images, lambda_term):
        batch_size = real_images.shape[0]
        eta = torch.FloatTensor(batch_size, 1).uniform_(0, 1)
        eta = eta.expand(batch_size, real_images.size(1)).to(self.device)

        interpolated = eta * real_images + ((1 - eta) * fake_images).to(self.device)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.disc(interpolated.float())

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                                  create_graph=True, retain_graph=True)[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
        return grad_penalty

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
