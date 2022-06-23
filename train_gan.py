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
from generative_model import DCGAN, WGAN_GP, WGAN_SIMPLE

dataroot = "./data/samples_sep"
savepath = "./model"
workers = 1
batch_size = 128
nz = 100
ngf = 64
ndf = 64
num_epochs = 200
Glr = 0.0002
Dlr = 0.0002
beta1 = 0.5
ngpu = 1


def train_DCGAN_W_GP():
    data = np.genfromtxt('./data/sample/Rayleigh_P30_flat.csv', delimiter=',', skip_header=True)
    scaler = MinMaxScaler((32, 32))
    scaler.fit(data)

    map_dataset = AfricaPatch_Flat('./data/Rayleigh_P30_flat.csv', scaler)
    dataloader = DataLoader(map_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("currently using device:", device)

    model = WGAN_GP(device=device)

    # check device and topology
    for n, p in model.G.named_parameters():
        print(p.device, '', n)
    for n, p in model.D.named_parameters():
        print(p.device, '', n)

    # data logging
    wandb.init(project="mcmc-wgan")

    model.optimize(dataloader, epochs=num_epochs, Glr=Glr, Dlr=Dlr, betas=(beta1, 0.999), scaler=scaler, device=device,
                   savepath=savepath)

def train_WGAN_SIMPLE(recompute_scaler=True):
    data = np.genfromtxt('./data/Rayleigh_P30_downsampled_flat_extended.csv', delimiter=',', skip_header=True)

    if recompute_scaler:
        scaler = MinMaxScaler((1,data.shape[1]))
        scaler.fit(data)
        scaler_file = open('./data/whole_scaler_extended.pkl', 'wb')
        pickle.dump(scaler, scaler_file)
    else:
        scaler_file = open('./data/whole_scaler_extened.pkl', 'rb')
        scaler = pickle.load(scaler_file)

    data = scaler.transform(data)

    map_dataset = Africa_Whole_Flat(data)
    dataloader = DataLoader(map_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("currently using device:", device)

    model = WGAN_SIMPLE(ndim=data.shape[1],device=device)

    config = dict(
        learning_rate=0.0002,
        momentum=0.5,
        training_epoch = 200,
        architecture = "WGAN_MLP",
        data = "Rayleigh P30 last 3 runs",
        device = device
    )
    wandb.init(project="mcmc-wgan-simple",
               notes = "hyperparameter tuning",
               config=config)

    model.optimize(dataloader, epochs=config['training_epoch'], lr=config['learning_rate'], beta1=config['momentum'], device=device)




if __name__ == "__main__":
    train_WGAN_SIMPLE(recompute_scaler=True)
