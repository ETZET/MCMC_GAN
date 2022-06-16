import torch
from torch import nn

import tqdm
import time
import wandb
import pickle
import matplotlib.pyplot as plt

def get_circle_data(ndim=2,nsamples=100000,chain=0):
    if ndim == 2:
        if nsamples == 100000:
            with open(r'datasets/mcmc_samples_pcp_dim_2.obj','rb') as file:
                data = pickle.load(file)
        else:
            with open(r'datasets/mcmc_samples_pcp_dim_2_reduced_'+str(nsamples)+'.obj','rb') as file:
                data = pickle.load(file)
        data = data[:,chain,:]
    else:
        with open(r'datasets/mcmc_samples_pcp_dim_'+str(ndim)+'.obj','rb') as file:
            data = pickle.load(file)
    return torch.Tensor(data)



class DCGAN(nn.Module):
    def __init__(self,nlatent=100,ngf=64,ndf=64):
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
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ngf*2) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            # Size: (ngf) x 16 x 16
            nn.ConvTranspose2d( ngf, 1, 4, 2, 1, bias=False),
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


    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def optimize(self,dataloader,lr=1e-4,epochs=10,betas=(0.5,0.999),device="cpu",scaler=None):

        # Initialize BCELoss function
        loss = nn.BCELoss()

        wandb.watch(models=self,criterion=loss,log="gradients",log_freq=5)

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(8, self.nlatent, 1, 1, device=device)

        # apply smoothing
        real_label = 0.9
        fake_label = 0

        optimizer_gen = torch.optim.Adam(self.gen.parameters(),lr=lr,betas=betas)
        optimizer_disc = torch.optim.Adam(self.disc.parameters(),lr=lr,betas=betas)

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
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = self.disc(real_data).view(-1)
                errD_real = loss(output,label)
                errD_real.backward()
                D_x = output.mean().item()

                ### fake data batch
                fake_data = self.gen(torch.randn(b_size,self.nlatent,1,1,device=device))
                label.fill_(fake_label)
                # pass through Discriminator
                output = self.disc(fake_data.detach()).view(-1)
                errD_fake = loss(output,label)
                errD_fake.backward()
                # record output
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake

                optimizer_disc.step()


                ### Update Generator
                optimizer_gen.zero_grad()

                fake_data = self.gen(torch.randn(b_size,self.nlatent,1,1,device=device))
                label.fill_(real_label)
                output = self.disc(fake_data).view(-1)

                errG = loss(output,label)
                errG.backward()

                D_G_z2 = output.mean().item()

                optimizer_gen.step()

                toc = time.time()

                # Output training stats
                if i % 2 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f Elapsed time: %.4fs'
                        % (epoch, epochs, i, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, (toc-tic)))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                ###### Statistics logging
                with torch.no_grad():
                    fake = self.gen(fixed_noise).detach().cpu()
                fig, axes = plt.subplots(2,4,figsize=(22,8),dpi=100)
                for i in range(4):  
                    real_data_cpu = real_data.detach().cpu()
                    axes[1][i].pcolormesh(scaler.inverse_transform(fake[i][0]))
                    axes[0][i].pcolormesh(scaler.inverse_transform(real_data_cpu[i][0]))
                fig.savefig("{}/{}.png".format('./figures','G(z)'),format='png')
                wandb.log({'D_loss': errD.item(), 'G_loss': errG.item()})
                wandb.log({"Visualization": wandb.Image("{}/{}.png".format('./figures','G(z)'))})

                iters += 1
        return G_losses, D_losses
    
