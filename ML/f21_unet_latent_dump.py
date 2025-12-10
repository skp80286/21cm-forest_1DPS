'''
Use unet model to extract the latent features and write them into files
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import argparse
import glob
from datetime import datetime

import F21DataLoader as dl
import f21_predict_base as base
import plot_results as pltr
import Scaling
import PS1D
import F21Stats as f21stats
from UnetModelWithDense import UnetModel

import numpy as np
import sys
import os

import matplotlib.pyplot as plt


def dump_latent(datafile, dir):
    logger.info(f"Loading file {i+1}/{len(datafiles)}: {datafile}")
    file_name = os.path.basename(datafile)  # Extract the filename from the path
    file_name_no_ext = os.path.splitext(file_name)[0]  # Remove the extension

    X_train, y_train, _, keys, freq_axis = base.load_dataset([datafile], max_workers=1, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False, skip_ps=True)
    if args.input_points_to_use is not None:
        X_train = X_train[:, :args.input_points_to_use]
        freq_axis = freq_axis[:, :args.input_points_to_use]
    bandwidth = freq_axis[0][-1] - freq_axis[0][0]

    # Denoising
    logger.info(f" Denoising signal: los original shape={X_train.shape}, bandwidth={bandwidth}")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    latent = model.get_latent_features(X_train_tensor)
    logger.info(f" Latent features shape={latent.shape}")
    # Save the latent features to a file
    latent_file = f'{dir}/{file_name_no_ext}.npy' 
    np.save(latent_file, latent)
    logger.info(f" Saved latent features to {latent_file}")

# main code starts here
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.determinisitc=True
torch.backends.cudnn.benchmark=False

parser = base.setup_args_parser()
parser.add_argument('--dataset', type=str, default='full', help='one of full, test_only, small_set')
parser.add_argument('--kernel1', type=int, default=5, help='5,3,7, etc')
parser.add_argument('--kernel2', type=int, default=3, help='5,3,7, etc')
parser.add_argument('--latentdim', type=int, default=256, help='256, 512, etc')
parser.add_argument('--dropout', type=float, default=0.2, help='value between 0 and 1')
parser.add_argument('--pooling', type=str, default='max', help='max, avg, etc')
parser.add_argument('--activation', type=str, default='relu', help='relu, elu, leaky, etc')

args = parser.parse_args()
#if args.input_points_to_use not in [2048, 128]: raise ValueError(f"Invalid input_points_to_use {args.input_points_to_use}")
if args.input_points_to_use >= 2048: 
    step = 4
else: 
    step = 2

output_dir = base.create_output_dir(args=args)
logger = base.setup_logging(output_dir)

## Loading data
datafiles = []
if args.target.startswith("PSOJ352"):
    args.telescope = 'uGMRT'
    args.rms = 6.0
    datafiles += base.get_rms_datafile_list(type='signalandnoise', args=args)
    args.telescope = 'uGMRT'
    args.rms = 3.3
    datafiles += base.get_rms_datafile_list(type='signalandnoise', args=args)
else:
    datafiles = base.get_datafile_list(type='noisy', args=args)

if args.maxfiles is not None: datafiles = datafiles[:args.maxfiles]

small_dataset_points = [[-3.00,0.11],[-2.00,0.11],[-1.00,0.11],[-3.00,0.25],[-2.00,0.25],[-1.00,0.25],[-3.00,0.52],[-2.00,0.52],[-1.00,0.52], [-3.00,0.80],[-2.00,0.80],[-1.00,0.80]]#,[0.00,0.80]]
test_points = [[-3,0.11], [-3,0.80], [-1,0.11], [-1,0.80], [-2,0.52], [-3.6,0.80], [-3.6,0.51], [-3.6,0.24]]

train_files = []
test_files = []
small_dataset_files = []
for nof in datafiles:
    is_test_file = False
    for p in test_points:
        if nof.find(f"fX{p[0]:.2f}_xHI{p[1]:.2f}") >= 0:
            test_files.append(nof)
            is_test_file = True
            break
    if not is_test_file:
        train_files.append(nof)
    for p in small_dataset_points:
        if nof.find(f"fX{p[0]:.2f}_xHI{p[1]:.2f}") >= 0:
            small_dataset_files.append(nof)
            break

# Initialize the network
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

logger.info("####")
logger.info(f"### Using \"{device}\" device ###")
logger.info("####")

## Load the trained Unet model
logger.info(f"Loading model from file {args.modelfile}, output_size={args.input_points_to_use}, dropout=0.2, step={step}, kernel1={args.kernel1}, kernel2={args.kernel2}, latentdim={args.latentdim}")
model = UnetModel(input_size=args.input_points_to_use, input_channels=1, output_size=args.input_points_to_use, dropout=0.2, step=step, kernel1=args.kernel1, kernel2=args.kernel2, latentdim=args.latentdim, pooling=args.pooling, activation=args.activation)
model.load_model(args.modelfile)

logger.info(f"Loading dataset {len(datafiles)}")

fileset = None
if args.dataset == "full": fileset = train_files + test_files
elif args.dataset == "test_only": fileset = test_files
elif args.dataset == "small_dataset": fileset = small_dataset_files
 

latent_dir = f'{output_dir}/latent'
test_latent_dir = f'{output_dir}/test_latent'
os.mkdir(latent_dir)
os.mkdir(test_latent_dir)

if args.dataset == "full":
    logger.info(f'Dumping latent features for training')
    for i,datafile in enumerate(train_files):
        logger.info(f"Loading file {i+1}/{len(train_files)}: {datafile}")
        dump_latent(datafile, dir=latent_dir)

logger.info(f'Dumping latent features for testing')
for i,datafile in enumerate(test_files):
    logger.info(f"Loading file {i+1}/{len(test_files)}: {datafile}")
    dump_latent(datafile, dir=test_latent_dir)

logger.info(f"Dump completed. Output:{output_dir}")
