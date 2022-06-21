from numpy import save
import torch
from torch import nn
from torch.autograd import Variable
from torch import autograd
import tqdm
import time
import wandb
import pickle
import numpy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_circle_data(ndim=2, nsamples=100000, chain=0):
    if ndim == 2:
        if nsamples == 100000:
            with open(r'datasets/mcmc_samples_pcp_dim_2.obj', 'rb') as file:
                data = pickle.load(file)
        else:
            with open(r'datasets/mcmc_samples_pcp_dim_2_reduced_' + str(nsamples) + '.obj', 'rb') as file:
                data = pickle.load(file)
        data = data[:, chain, :]
    else:
        with open(r'datasets/mcmc_samples_pcp_dim_' + str(ndim) + '.obj', 'rb') as file:
            data = pickle.load(file)
    return torch.Tensor(data)


class DCGAN(nn.Module):
    def __init__(self, nlatent=100, ngf=64, ndf=64):
        super().__init__()

        self.nlatent = nlatent

        self.gen = nn.Sequential(
            # Input: latent vector of length nlatent
            nn.ConvTranspose2d(self.nlatent, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ngf*8) x 2 x 2
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: 1 x 32 x 32
        )

        self.disc = nn.Sequential(
            # Input: 1 x 32 x 32
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ndf*8) x 2 x 2
            nn.Conv2d(ndf * 8, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

        self.gen.apply(self.weights_init)
        self.disc.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def optimize(self, dataloader, Glr=1e-4, Dlr=1e-4, epochs=10, betas=(0.5, 0.999), device="cpu", scaler=None,
                 savepath=None):

        # Initialize BCELoss function
        loss = nn.BCELoss()

        wandb.watch(models=self, criterion=loss, log="gradients", log_freq=5)

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(8, self.nlatent, 1, 1, device=device)

        # apply smoothing
        real_label = 0.7
        fake_label = 0

        optimizer_gen = torch.optim.Adam(self.gen.parameters(), lr=Glr, betas=betas)
        optimizer_disc = torch.optim.Adam(self.disc.parameters(), lr=Dlr, betas=betas)

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        for epoch in range(epochs):
            for i, data in enumerate(dataloader):
                tic = time.time()
                ### Update Discriminator
                optimizer_disc.zero_grad()

                ### real data batch
                real_data = data.to(device)
                b_size = real_data.size(0)
                # real_label = numpy.random.uniform(low=0.7,high=1.0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = self.disc(real_data).view(-1)
                errD_real = loss(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                ### fake data batch
                fake_data = self.gen(torch.randn(b_size, self.nlatent, 1, 1, device=device))
                label.fill_(fake_label)
                # pass through Discriminator
                output = self.disc(fake_data.detach()).view(-1)
                errD_fake = loss(output, label)
                errD_fake.backward()
                # record output
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake

                optimizer_disc.step()

                ### Update Generator
                optimizer_gen.zero_grad()

                fake_data = self.gen(torch.randn(b_size, self.nlatent, 1, 1, device=device))
                # real_label = numpy.random.uniform(low=0.7,high=1.0)
                real_label = 1.0
                label.fill_(real_label)
                output = self.disc(fake_data).view(-1)

                errG = loss(output, label)
                errG.backward()

                D_G_z2 = output.mean().item()

                optimizer_gen.step()

                toc = time.time()

                # Output training stats
                if i % 2 == 0:
                    print(
                        '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f  Elapsed time per Epoch: %.4fs'
                        % (epoch, epochs, i, len(dataloader),
                           errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, (toc - tic)))
                if epoch % 5 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'gen_optimizer_state_dict': optimizer_gen.state_dict(),
                        'dist_optimizer_state_dict': optimizer_disc.state_dict(),
                        'D loss': errD.item(),
                        'G loss': errG.item()
                    }, "{}/DCGAN_epoch{}.model".format(savepath, epoch))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                ###### Statistics logging
                with torch.no_grad():
                    fake = self.gen(fixed_noise).detach().cpu()
                fig, axes = plt.subplots(2, 4, figsize=(22, 8), dpi=100)
                for i in range(4):
                    real_data_cpu = real_data.detach().cpu()
                    axes[1][i].pcolormesh(scaler.inverse_transform(fake[i][0]))
                    axes[0][i].pcolormesh(scaler.inverse_transform(real_data_cpu[i][0]))
                fig.savefig("{}/{}.png".format('./figures', 'G(z)'), format='png')
                plt.close(fig)
                wandb.log({'D_loss': errD.item(), 'G_loss': errG.item()})
                wandb.log({"Visualization": wandb.Image("{}/{}.png".format('./figures', 'G(z)'))})

                iters += 1
        return G_losses, D_losses


class Generator(nn.Module):
    def __init__(self, nlatent=100):
        super().__init__()

        self.nlatent = nlatent

        self.main = nn.Sequential(
            # Input: latent vector of length nlatent
            nn.ConvTranspose2d(self.nlatent, 1024, 4, 2, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # Size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # Size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Size: (ngf) x 16 x 16
            nn.ConvTranspose2d(256, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: 1 x 32 x 32
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        return self.main(x)

    def feature_extraction(self, x):
        x = self.main(x)
        return x.view(-1, 1024 * 4 * 4)


class WGAN_GP(object):

    def __init__(self, device="cpu"):
        self.device = device

        self.G = Generator().to(device)
        self.D = Discriminator().to(device)

        self.G.apply(self.weights_init)
        self.D.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def calculate_gradient_penalty(self, real_images, fake_images, lambda_term):
        batch_size = real_images.shape[0]
        eta = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(batch_size, real_images.size(1), real_images.size(2), real_images.size(3)).to(self.device)

        interpolated = eta * real_images + ((1 - eta) * fake_images).to(self.device)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                                  create_graph=True, retain_graph=True)[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
        return grad_penalty

    def optimize(self, dataloader, Glr=1e-4, Dlr=1e-4, epochs=10, betas=(0.5, 0.999), lambda_term = 10, device="cpu", scaler=None,
                 savepath=None):

        # Initialize BCELoss function
        loss = nn.BCELoss()

        wandb.watch(models=self.G, criterion=loss, log="gradients", log_freq=5)
        wandb.watch(models=self.D, criterion=loss, log="gradients", log_freq=5)

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(32, self.G.nlatent, 1, 1, device=device)

        # apply smoothing
        one = torch.tensor(1, dtype=torch.float).to(device)
        mone = torch.tensor(-1, dtype=torch.float).to(device)

        optimizer_gen = torch.optim.Adam(self.G.parameters(),lr=Glr,betas=betas)
        optimizer_disc = torch.optim.Adam(self.D.parameters(),lr=Dlr,betas=betas)

        # Lists to keep track of progress
        d_loss_real = 0
        d_loss_fake = 0
        Wasserstein_D = 0
        iters = 0

        for epoch in range(epochs):
            for i, data in enumerate(dataloader):
                tic = time.time()

                ### Update Discriminator
                optimizer_disc.zero_grad()

                ### generate data
                real_data = data.to(device)
                b_size = real_data.size(0)

                ### update discriminator
                d_loss_real = self.D(real_data).mean()
                d_loss_real.backward(mone)

                ### fake data batch
                fake_data = self.G(torch.randn(b_size, self.G.nlatent, 1, 1, device=device))
                d_loss_fake = self.D(fake_data)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                ### gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(real_data, fake_data,lambda_term=lambda_term)
                gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                optimizer_disc.step()

                ### Update Generator
                optimizer_gen.zero_grad()

                fake_data = self.G(torch.randn(b_size, self.G.nlatent, 1, 1, device=device))
                g_loss = self.D(fake_data).mean()
                g_loss.backward(mone)
                g_cost = -g_loss
                optimizer_gen.step()

                toc = time.time()

                # Output training stats
                if i % 2 == 0:
                    print(
                        '[%d/%d][%d/%d]\tLoss_D_real: %.4f\tLoss_D_fake: %.4f\tLoss_G: %.4f  Elapsed time per Epoch: %.4fs'
                        % (epoch, epochs, i, len(dataloader),
                           d_loss_real, d_loss_fake, g_loss, (toc - tic)))
                if epoch % 5 == 0:
                    torch.save({
                        'epoch': epoch,
                        'G_state_dict': self.G.state_dict(),
                        'D_state_dict': self.D.state_dict(),
                        'gen_optimizer_state_dict': optimizer_gen.state_dict(),
                        'dist_optimizer_state_dict': optimizer_disc.state_dict(),
                        'D loss': d_loss_real + d_loss_fake,
                        'G loss': g_loss
                    }, "{}/DCGAN_epoch{}.model".format(savepath, epoch))

                ##### Statistics logging
                if i % 50 == 0:
                    with torch.no_grad():
                        fake = self.G(fixed_noise).detach().cpu()
                    fig, axes = plt.subplots(4,8,figsize=(22,8),dpi=100)
                    axes = axes.flatten()
                    for i in range(32):
                        axes[i].pcolormesh(scaler.inverse_transform(fake[i][0]))
                    fig.savefig("{}/{}.png".format('./figures','G(z)_WGAN'),format='png')
                    plt.close(fig)


                wandb.log({'D_loss_real': d_loss_real,'D_loss_fake': d_loss_fake, 'G_loss': g_loss})
                wandb.log({"Visualization": wandb.Image("{}/{}.png".format('./figures','G(z)'))})

                iters += 1
