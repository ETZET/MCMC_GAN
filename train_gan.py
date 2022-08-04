"""
Title: GAN Training Script
Author: Enting Zhou
Date: 06/15/2022
Availability: https://github.com/ETZET/MCMC_GAN
"""
import os
import pickle
import argparse
import torch
import numpy as np
import wandb
from process_data import MinMaxScaler
from generative_model import WGAN_SIMPLE
import h5py


def train_wgan_simple(args):
    """
    user training program
    :param args: Namespace, provide training parameters
    """
    # read data
    print("Reading Data...")
    hf = h5py.File(args.input_path,'r')
    data = np.array(hf.get('velmap'))
    # use later 50 % to train
    burnin = int(data.shape[0]*0.5)
    data = data[burnin:,:]
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print("currently using device:", device)

    config = dict(
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        training_epoch=args.epochs,
        batch_size=args.batch_size,
        architecture="WGAN_MLP",
        data=args.input_path,
        device=device
    )

    # Normalize data to range of (-1,1)
    print("Scaling input data...")
    scaler = MinMaxScaler()
    scaler.fit(data)
    with open(os.path.join(args.output_path,'whole_scaler_extended.pkl'), 'wb') as f:
        pickle.dump(scaler,f)
    data = scaler.transform(data)

    # initialize model
    model = WGAN_SIMPLE(ndim=data.shape[1], device=device,nhid=300)

    if args.use_wandb:
        wandb.init(project="mcmc-wgan-simple",
                   config=config)

    model.optimize(data,output_path=args.output_path, use_wandb=args.use_wandb,
                   batch_size=config['batch_size'], epochs=config['training_epoch'],
                   lr=config['learning_rate'], beta1=config['momentum'],
                   device=device)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="GAN", help="Model Name")
    parser.add_argument("-i", "--input-path", type=str, required=True, help="Model Name")
    parser.add_argument("-o", "--output-path", type=str,required=True, help="Model Name")
    parser.add_argument("--epochs", type=int, default=200, help="Model Name")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.0002, help="Model Name")
    parser.add_argument("-b", "--batch-size", type=int, default=128, help="Model Name")
    parser.add_argument("--momentum", type=float, default=0.5, help="Model Name")
    parser.add_argument("--use-wandb", type=bool, default=False, help="Model Name")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    # optimization
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    train_wgan_simple(args)
