import os
import pickle
import torch
import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import seaborn as sns
from process_data import *
from generative_model import DCGAN, WGAN_GP


dataroot = "./data/samples_sep"
savepath = "./model"
workers = 1 
batch_size = 128
nz = 100
ngf = 64
ndf = 64
num_epochs = 200
Glr = 0.0002
Dlr= 0.0002
beta1 = 0.5
ngpu = 1

if __name__ == "__main__":

    # load the precomputed scaler
    scaler_file = open('./data/sample/scaler.pkl','rb')
    scaler = pickle.load(scaler_file)

    map_dataset = AfricaPatch_Flat('./data/Rayleigh_P30_flat.csv',scaler)

    dataloader = DataLoader(map_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("currently using device:", device)
    # Training Loop
    # model = DCGAN(nlatent=nz,ngf=ngf,ndf=ndf).to(device)
    #
    # # check device and topology
    # for n, p in model.named_parameters():
    #     print(p.device, '', n)
    #
    # # data logging
    # wandb.init(project="mcmc-gan")
    #
    # torch.save(model.state_dict(),'./model/dcgan_untrained.model')
    # model.optimize(dataloader,epochs=num_epochs,Glr=Glr,Dlr=Dlr,betas=(beta1,0.999),scaler=scaler,device=device,savepath=savepath)
    # torch.save(model.state_dict(),'./model/dcgan_trained.model')

    model = WGAN_GP(device=device)

    # check device and topology
    for n, p in model.G.named_parameters():
        print(p.device, '', n)
    for n, p in model.D.named_parameters():
        print(p.device, '', n)

    # data logging
    wandb.init(project="mcmc-wgan")

    # torch.save(model.state_dict(), './model/dcgan_untrained.model')
    model.optimize(dataloader, epochs=num_epochs, Glr=Glr, Dlr=Dlr, betas=(beta1, 0.999), scaler=scaler, device=device,
                   savepath=savepath)
    # torch.save(model.state_dict(), './model/dcgan_trained.model')
