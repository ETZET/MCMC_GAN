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

    def __init__(self, root_dir, velName, velType, velPd, scaler, num_files=None):
        self.root_dir = root_dir
        self.velName = velName
        self.velType = velType
        self.velPd = velPd
        if num_files == None:
            self.num_maps = len([f for f in os.listdir(root_dir) \
                                 if f.endswith('.csv') and os.path.isfile(os.path.join(root_dir, f))])
        else:
            self.num_maps = num_files
        self.scaler = scaler

    def __len__(self):
        return self.num_maps

    def __getitem__(self, idx):
        sample = torch.zeros((1, 32, 32))
        path = os.path.join(self.root_dir, "{}_{}{}_{}.csv".format(self.velName,
                                                                   self.velType, self.velPd, idx))
        sample_np = self.scaler.transform(np.genfromtxt(path, delimiter=",", skip_header=True))
        sample[0, :, :] = torch.from_numpy(sample_np)
        return sample


class AfricaPatch_Flat(Dataset):
    def __init__(self, directory, scaler, dim=32):
        self.dir = directory
        self.data = np.genfromtxt(directory, delimiter=',', skip_header=True)
        self.scaler = scaler
        self.dim = 32

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = torch.zeros((1, self.dim, self.dim))
        sample_np = self.data[idx, :].reshape((self.dim, self.dim))
        sample_scaled = self.scaler.transform(sample_np)
        sample[0, :, :] = torch.from_numpy(sample_scaled)
        return sample

class Africa_Whole_Flat(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample_np = self.data[idx, :]
        sample = torch.from_numpy(sample_np)
        return sample

class MinMaxScaler():

    def __init__(self, transform_shape: tuple):
        self.min = np.empty(transform_shape)
        self.max = np.empty(transform_shape)
        self.diff = np.empty(transform_shape)
        self.transform_shape = transform_shape

    def fit(self, data: np.ndarray):
        self.min = np.min(data, axis=0).reshape(self.transform_shape)
        max = np.max(data, axis=0).reshape(self.transform_shape)
        self.diff = max - self.min

    def transform(self, data: np.ndarray, range=(-1, 1)):
        range_min = range[0]
        range_max = range[1]
        data_std = (data - self.min) / self.diff
        data_scaled = data_std * (range_max - range_min) + range_min
        return data_scaled

    def inverse_transform(self, data_scaled: np.ndarray, range=(-1, 1)):
        range_min = range[0]
        range_max = range[1]
        data_std = (data_scaled - range_min) / (range_max - range_min)
        data = data_std * self.diff + self.min
        return data


if __name__ == "__main__":
    # data = np.genfromtxt('./data/sample/Rayleigh_P30_flat.csv', delimiter=',', skip_header=True)
    #
    # scaler = MinMaxScaler((32, 32))
    # scaler.fit(data)
    # scaler_file = open('./data/sample/scaler.pkl', 'wb')
    # pickle.dump(scaler, scaler_file)

    data = np.genfromtxt('./data/Rayleigh_P30_downsampled_flat.csv', delimiter=',', skip_header=True)
    scaler = MinMaxScaler((1, data.shape[1]))
    scaler.fit(data)

    sample = data[200,:]
    sample_scaled = scaler.transform(sample)
    sample_inverst = scaler.inverse_transform(sample_scaled)
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    axes[0].plot(sample)
    axes[1].plot(sample_scaled)
    axes[2].plot(sample_inverst)
    plt.show()
