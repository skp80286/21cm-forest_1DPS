import torch
import torch.nn as nn
import torch.utils.data as tud
from BayesianCNN21cm import BayesianCNN21cm
from BayesianCNN21cm import create_bayesian_model,train_model, test_model, report_test_scores, sample_posterior, plot_posterior, save_model, load_model
import argparse
import glob
from datetime import datetime
import os

import F21DataLoader as dl
import f21_predict_base as base
import plot_results as pltr
import Scaling
import PS1D
import F21Stats as f21stats

# ----------------------------------------------------------
# 1. Create synthetic example dataset
# ----------------------------------------------------------

# main code starts here
parser = base.setup_args_parser()
parser.add_argument('--test_multiple', action='store_true', help='Test 1000 sets of 10 LoS for each test point and plot it')
parser.add_argument('--test_reps', type=int, default=10000, help='Test repetitions for each parameter combination')
parser.add_argument('--loss', type=str, default='mse', help="")
parser.add_argument('--epochsbatch', type=int, default=10, help='10,20, etc')
parser.add_argument('--kernel1', type=int, default=5, help='5,3,7, etc')
parser.add_argument('--kernel2', type=int, default=3, help='5,3,7, etc')
parser.add_argument('--dropout', type=float, default=0.1, help='value between 0 and 1')
parser.add_argument('--pooling', type=str, default='avg', help='max, avg, etc')
parser.add_argument('--activation', type=str, default='elu', help='relu, elu, leaky, etc')
parser.add_argument('--model_dir', type=str, default=None, help='Director to load model from (test_only)')
args = parser.parse_args()

output_dir = base.create_output_dir(args=args)
if args.runmode == "test_only": model_dir = args.model_dir
logger = base.setup_logging(output_dir)

logger.info(f"input_points={args.input_points_to_use}, kernel1={args.kernel1}, kernel2={args.kernel2}, dropout={args.dropout}")

datafiles = []
args.telescope = 'uGMRT'
args.rms = 6.0
datafiles += base.get_rms_datafile_list(type='signalandnoise', args=args)
args.telescope = 'uGMRT'
args.rms = 3.3
datafiles += base.get_rms_datafile_list(type='signalandnoise', args=args)

#test_points = [[-3.00,0.11],[-2.00,0.11],[-1.00,0.11],[-3.00,0.25],[-2.00,0.25],[-1.00,0.25],[-3.00,0.52],[-2.00,0.52],[-1.00,0.52], [-3.00,0.80],[-2.00,0.80],[-1.00,0.80]]#,[0.00,0.80]]
test_points = [[-3.6,0.8],[-3.6,0.51],[-3.6,0.24]]
train_files = []
test_files = []
sotrain_files = []
sotest_files = []

for snof in datafiles:
    is_test_file = False
    for p in test_points:
        if snof.find(f"fX{p[0]:.2f}_xHI{p[1]:.2f}") >= 0:
            test_files.append(snof)
            is_test_file = True
            break
    if not is_test_file:
        train_files.append(snof)

criterion = nn.MSELoss()

if args.runmode == "train_test":
    logger.info(f"Loading train dataset {len(train_files)}")
    X_train, y_train, _, keys, freq_axis = base.load_dataset(train_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
    logger.info(f"Loaded datasets X_train:{X_train.shape} y_train:{y_train.shape}")
    
    #run(X_train, X_test, y_train, None, y_test, None, None, args.epochs, args.trainingbatchsize, lr=0.00001, kernel1=args.kernel1, kernel2=args.kernel2, dropout=args.dropout, step=step, input_points_to_use=args.input_points_to_use, showplots=args.interactive, criterion=criterion)
    
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    
    train_dataset = tud.TensorDataset(X_train, y_train)

    
    # ----------------------------------------------------------
    # 2. Create the model (configurable)
    # ----------------------------------------------------------
    
    k1,k2 = args.kernel1, args.kernel2
    model = create_bayesian_model(
        input_length=args.input_points_to_use,
        in_channels=1,
        conv_channels=(32, 64, 128, 256),        # 4 convolutional blocks
        kernel_sizes=((k1, k2), (k1, k2), (k1, k2), (k1, k2)),
        fc_layers=(256, 128, 64),                # 3 dense layers
        activation=args.activation,                        # relu, elu, leakyrelu
        pooling=args.pooling,                           # max or avg pooling
        dropout=args.dropout,
    )
    
    logger.info(model)
    
    
    # ----------------------------------------------------------
    # 3. Train the model
    # ----------------------------------------------------------
    
    trained_model = train_model(
        model,
        train_dataset,
        num_epochs=args.epochs,
        batch_size=args.trainingbatchsize,
        lr=1e-3,
        weight_decay=1e-5,
    )
    
    logger.info(f"Saving trained model to: {output_dir}")
    save_model(trained_model, f'{output_dir}/trained_model')
    model_dir = output_dir

logger.info(f"Loading trained model from {model_dir}")
trained_model, _ = load_model(f'{model_dir}/trained_model')
logger.info(f"Loading test dataset {len(test_files)}")
X_test, y_test, _, keys, _ = base.load_dataset(test_files, psbatchsize=1, limitsamplesize=800, save=False)
logger.info(f"Loaded dataset X_test:{X_test.shape} y_test:{y_test.shape} ")
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

test_dataset = tud.TensorDataset(X_test, y_test)
# ----------------------------------------------------------
# 4. Test the model
# ----------------------------------------------------------

y_true, y_pred = test_model(
    trained_model,
    test_dataset,
    batch_size=64,
)


# ----------------------------------------------------------
# 5. Print test metrics
# ----------------------------------------------------------

metrics = report_test_scores(y_true, y_pred)
y_true_np =y_true.detach().cpu().numpy()
y_pred_np =y_pred.detach().cpu().numpy()
logger.info(f'y_true_np.shape={y_true_np.shape}, {y_pred_np.shape}')
#logger.info(f'y_true={y_true_np}\n\n {y_pred_np}')
base.save_predictions_to_csv(y_pred_np, y_true_np, filename=f'{output_dir}/test_results.csv')
logger.info(metrics)


# ----------------------------------------------------------
# Posterior for a single test spectrum
# ----------------------------------------------------------
idx = 0  # choose a test index
spectrum = X_test[idx]         # shape (3584,)
true_vals = (y_test[idx, 0].item(), y_test[idx, 1].item())

samples = sample_posterior(model, spectrum, n_samples=1000)
logger.info(f"Posterior mean: {samples.mean(dim=0)}")
logger.info(f"Posterior std : {samples.std(dim=0)}")

# Posterior plots
plot_posterior(samples, true_values=true_vals,
param_names=("xHI", "logfX"), output_dir=output_dir)
