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
from generative_model import DCGAN


dataroot = "./data/samples_sep"
workers = 1 
batch_size = 128
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 10
lr = 0.0002
beta1 = 0.5
ngpu = 1

if __name__ == "__main__":

    # load the precomputed scaler
    scaler_file = open('./data/sample/scaler.pkl','rb')
    scaler = pickle.load(scaler_file)

    map_dataset = AfricaPatch(root_dir=dataroot,velName="Rayleigh",\
    velType="P",velPd="30",scaler=scaler, num_files=45000)

    dataloader = DataLoader(map_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("currently using device:", device)
    # Training Loop
    model = DCGAN().to(device)

    # check device and topology
    for n, p in model.named_parameters():
        print(p.device, '', n)

    # data logging
    wandb.init(project="mcmc-gan")

    torch.save(model.state_dict(),'./model/dcgan_untrained.model')
    model.optimize(dataloader,epochs=num_epochs,lr=lr,scaler=scaler,device=device)
    torch.save(model.state_dict(),'./model/dcgan_trained.model')
