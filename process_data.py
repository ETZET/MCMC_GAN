import pickle
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import seaborn as sns

class AfricaPatch(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,root_dir,velName,velType,velPd,scaler,num_files=None):
        self.root_dir = root_dir
        self.velName = velName
        self.velType = velType
        self.velPd = velPd
        if num_files==None:
            self.num_maps = len([f for f in os.listdir(root_dir) \
                if f.endswith('.csv') and os.path.isfile(os.path.join(root_dir, f))])
        else:
            self.num_maps = num_files
        self.scaler = scaler
        
    def __len__(self):
        return self.num_maps


    def __getitem__(self, idx):
        sample = torch.zeros((1,32,32))
        path = os.path.join(self.root_dir,"{}_{}{}_{}.csv".format(self.velName,\
                self.velType,self.velPd,idx))
        sample_np = self.scaler.transform(np.genfromtxt(path,delimiter=",",skip_header=True))
        sample[0,:,:] = torch.from_numpy(sample_np)
        return sample

class MinMaxScaler():

    def __init__(self, transform_shape: tuple):
        self.min = np.empty(transform_shape)
        self.max = np.empty(transform_shape)
        self.diff = np.empty(transform_shape)
        self.transform_shape = transform_shape
    
    def fit(self, data: np.ndarray):
        self.min = np.min(data,axis=0).reshape(self.transform_shape)
        max = np.max(data,axis=0).reshape(self.transform_shape)
        self.diff = max-self.min
    
    def transform(self, data: np.ndarray, range=(-1,1)):
        range_min = range[0]
        range_max = range[1]
        data_std = (data - self.min) / self.diff
        data_scaled = data_std * (range_max-range_min) + range_min
        return data_scaled
    
    def inverse_transform(self, data_scaled: np.ndarray, range=(-1,1)):
        range_min = range[0]
        range_max = range[1]
        data_std = (data_scaled - range_min) / (range_max - range_min)
        data = data_std * self.diff + self.min
        return data



if __name__ == "__main__":
    # data = np.genfromtxt('./data/sample/Rayleigh_P30_flat.csv',delimiter = ',',skip_header=True)

    # scaler = MinMaxScaler((32,32))
    # scaler.fit(data)
    # scaler_file = open('./data/sample/scaler.pkl','wb')
    # pickle.dump(scaler,scaler_file)

    # # load the precomputed scaler
    scaler_file = open('./data/sample/scaler.pkl','rb')
    scaler = pickle.load(scaler_file)

    path = os.path.join("./data/samples_sep","{}_{}{}_{}.csv".format("Rayleigh",\
                "P","30",20000))
    sample = np.genfromtxt(path,delimiter=",",skip_header=True)
    sample_scaled = scaler.transform(sample)
    sample_inverst = scaler.inverse_transform(sample_scaled)
    fig,axes = plt.subplots(1,3, figsize=(15,8))
    axes[0].pcolormesh(sample)
    axes[1].pcolormesh(sample_scaled)
    axes[2].pcolormesh(sample_inverst)
    plt.show()
    
    