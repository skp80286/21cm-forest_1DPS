import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import f21_predict_base as base
from scipy.stats import gaussian_kde

from sbi.inference import NPE, MCMCPosterior
from sbi.utils import BoxUniform
from xgboost import XGBRegressor
from sbi.inference import simulate_for_sbi
from sklearn.multioutput import MultiOutputRegressor

def xgb_simulator(theta):
    """
    theta: tensor shape (N,2)  [xHI, logfX]
    returns: tensor shape (N,512)
    """
    return torch.tensor(emulator.predict(theta.cpu().numpy()), dtype=torch.float32)

def parse_labels_from_filename(fname):
    logfX = float(re.search(r"logfX(-?\d+\.?\d*)", fname).group(1))
    xHI   = float(re.search(r"xHI(\d+\.?\d*)", fname).group(1))
    return xHI, logfX

def load_training_data(override_path, samples, args):
    # little hack to load _diffseed files only for testing
    #print(f'Loading training data for target {args.target}')
    if args.target.startswith('PSOJ352'): 
        files = base.get_rms_datafile_list('signalandnoise', args, extn='npy', override_path=override_path)
    else:
        args.extra_file_tag=''
        files = base.get_datafile_list('noisy', args, extn='npy', override_path=override_path)

    numgroups = samples//args.training_sample_group_size
    X_train = np.zeros((numgroups*len(files), args.latentdim), dtype=np.float32)
    y_train = np.zeros((numgroups*len(files), 2), dtype=np.float32)
    logger.info(f'Created X_train.shape={X_train.shape}, y_train.shape={y_train.shape}')
    
    for i, file in enumerate(files):
        curr_xHI = float(file.split('xHI')[1].split('_')[0])
        curr_logfX = float(file.split('fX')[1].split('_')[0])

        """
        ##
        ## Override the xHI and logfX with the accurate values from the simulation data file
        ##
        if curr_xHI > 1 or curr_xHI < 0 or curr_logfX > 1 or curr_logfX < -4:
            logger.error(f'Invalid: curr_xHI={curr_xHI}, curr_logfx={curr_logfX}, file={file}')
            logger.error(file.split('xHI')[1])
            logger.error(file.split('xHI')[1].split('_')[0])
        data = np.fromfile(file)
        logger.info(f'####data:{np.array2string(data[:30], formatter={'float_kind':lambda x: f"{x:.2f},"})}')
        sofilepattern = file.replace('^.*/F21_noisy',f'{args.path}/F21_signalonly').replace('^.*/F21_signalandnoise', '{args.path}/F21_signalonly')
        sofiles = glob.glob(sofilepattern)
        if len(sofiles) == 1:
            data = np.fromfile(sofiles[0])
            #data = np.fromfile(str('%sF21_signalonly_21cmFAST_200Mpc_z%.1f_fX%s_xHI%s_%s_%s_rms%.4fmJy_%.1fkHz%s.%s' %
            #       (path, args.redshift,args.log_fx, args.xHI, args.telescope, args.target, args.rms, args.spec_res,args.extra_file_tag,extn))
            logger.info(f'####sodata:{np.array2string(data[:30], formatter={'float_kind':lambda x: f"{x:.2f},"})}')
            curr_logfX = data[2]
            curr_xHI = data[3]
            if curr_xHI > 1 or curr_xHI < 0 or curr_logfX > 1 or curr_logfX < -4:
                logger.error(f'Invalid from sofile: curr_xHI={curr_xHI}, curr_logfx={curr_logfX}, sofile={sofiles[0]}')
        else:
            logger.info(f'Did not find signalonly file for curr_xHI={curr_xHI}, curr_logfx={curr_logfX}')
        """

        y_train[i*numgroups:(i+1)*numgroups, 0] = curr_xHI
        y_train[i*numgroups:(i+1)*numgroups, 1] = curr_logfX
        currps = np.load(file)[:samples,:args.latentdim]
        logger.info(f'loaded training data from file {file}. shape: {currps.shape}')
        logger.info(f'Loading data into X_train from rows: {i*numgroups} to {(i+1)*numgroups}')
        if args.training_sample_group_size > 1:
            currps_grouped = currps.reshape(-1, 10, currps.shape[1]).mean(axis=1)
        else:
            currps_grouped = currps

        if i == 0:
            logger.info(f"Original array shape: {currps.shape}")
            logger.info(f"Shape after grouping and taking mean: {currps_grouped.shape}")
            logger.info(f"currps sample:\n{currps[:10,2]}")
            logger.info(f"currps sample grouped:\n{currps_grouped[0][3]}")
        X_train[i*numgroups:(i+1)*numgroups, :] = currps_grouped[:,:]
    return torch.tensor(X_train, device=device), torch.tensor(y_train, device=device)


def load_test_data(override_path, samples, args):
    # little hack to load _diffseed files only for testing
    #args.extra_file_tag=''
    args.extra_file_tag='_diffseed'
    if args.target.startswith('PSOJ352'):
        #args.extra_file_tag='_seed370'
        files = base.get_rms_datafile_list('signalandnoise', args, extn='npy', filter="test_only", override_path=override_path)
    else:
        files = base.get_datafile_list('noisy', args, extn='npy', override_path=override_path)


    test_sets = []
    true_thetas = []

    for i, file in enumerate(files):
        curr_xHI = float(file.split('xHI')[1].split('_')[0])
        curr_logfX = float(file.split('fX')[1].split('_')[0])


        latents = np.load(file).astype(np.float32)[:, :args.latentdim]

        test_sets.append({
            "x": torch.tensor(latents, device=device),
            "true_theta": (curr_xHI, curr_logfX)
        })
        true_thetas.append((curr_xHI, curr_logfX))

    return test_sets, true_thetas

def posterior_using_xgb_sim(x, theta, prior):
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train(
        training_batch_size=1024,
        learning_rate=5e-4,
        max_num_epochs=50
    )
    posterior = inference.build_posterior(density_estimator, sample_with="mcmc")
    return posterior


def train_npe(x, theta, prior):
    inference = NPE(prior=prior, density_estimator="maf", device=device)

    inference.append_simulations(theta, x)
    density_estimator = inference.train(
        training_batch_size=1024,
        learning_rate=5e-4,
        max_num_epochs=50
    )

    #posterior = inference.build_posterior(density_estimator)
    posterior = inference.build_posterior(density_estimator, sample_with="mcmc")
    return posterior

def infer_posterior_for_test(posterior, x_test, n_samples=2000):
    all_samples = []

    for x in x_test:
        samples = posterior.sample(
            (n_samples,),
            x=x.unsqueeze(0)
        )
        all_samples.append(samples.cpu().numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    return all_samples

def compute_credible_levels(samples, levels=[0.683, 0.954, 0.997], gridsize=200):
    """
    samples: array of shape (N, 2)
    returns:
        X, Y : grid
        Z    : KDE evaluated on grid
        z_levels : density thresholds for given credible levels
    """
    x = samples[:, 0]
    y = samples[:, 1]

    kde = gaussian_kde(np.vstack([x, y]))

    # Grid
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    X, Y = np.meshgrid(
        np.linspace(xmin, xmax, gridsize),
        np.linspace(ymin, ymax, gridsize)
    )

    Z = kde(np.vstack([X.ravel(), Y.ravel()]))
    Z = Z.reshape(X.shape)

    # Sort densities (descending)
    Z_flat = Z.ravel()
    idx = np.argsort(Z_flat)[::-1]
    Z_sorted = Z_flat[idx]

    # Cumulative probability
    cumsum = np.cumsum(Z_sorted)
    cumsum /= cumsum[-1]

    # Find density thresholds
    z_levels = []
    for cl in levels:
        z_levels.append(Z_sorted[np.searchsorted(cumsum, cl)])

    return X, Y, Z, z_levels

def plot_sigma_contours(samples, color, level, label=None):
    """
    Draws 1σ, 2σ, 3σ credible contours.
    samples shape: (N,2) with [xHI, logfX]
    """
    X, Y, Z, z_levels = compute_credible_levels(samples, levels=[level])

    plt.contour(
        X, Y, Z,
        levels=z_levels,
        colors=color,
        linewidths=[2.5],
        linestyles=["solid"], #, "dashed", "dotted"]
    )

def save_posteriors_npy(posteriors, true_thetas):
    """
    posteriors: list of arrays, each shape (Nsamples, 2)
                columns: [xHI, logfX]
    true_thetas: list of tuples (xHI, logfX)
    """
    data = {
        "posteriors": posteriors,
        "true_thetas": np.array(true_thetas, dtype=np.float32)
    }

    filename = f"{output_dir}/posteriors.npy"
    np.save(filename, data, allow_pickle=True)
    logger.info(f"Saved posterior data to {filename}")

def load_posteriors_npy(posterior_dir):
    filename = f"{posterior_dir}/posteriors.npy"

    data = np.load(filename, allow_pickle=True).item()

    posteriors = data["posteriors"]
    true_thetas = data["true_thetas"]

    print(f"Loaded {len(posteriors)} posteriors from {filename}")
    return posteriors, true_thetas


def plot_three_posteriors_sigma(posteriors, true_thetas):
    posterior_means = [
        (p[:,0].mean(), p[:,1].mean())
        for p in posteriors
    ]

    for level in [0.683, 0.954, 0.997]:
        colors = ["red", "green", "blue", "orange", "violet"]
    
        plt.figure(figsize=(8,6))
    
        for i, samples in enumerate(posteriors):
            color = colors[i]
    
            # Draw 1σ / 2σ / 3σ contours
            plot_sigma_contours(samples, color=color, level=level)
    
            xHI = samples[:,0]
            logfX = samples[:,1]
    
    
            # True value (star)
            xHI_true, logfX_true = true_thetas[i]
            plt.scatter(
                xHI_true, logfX_true,
                marker="*", s=180, color=color, edgecolor="black", zorder=5
            )
    
            # Posterior mean (circle)
            plt.scatter(
                xHI.mean(), logfX.mean(),
                marker="o", s=100, color=color, edgecolor="black", zorder=5
            )
    
        plt.xlim(0, 1)
        plt.ylim(-4, 1)
        plt.xlabel("xHI")
        plt.ylabel("logfX")
        if args.target.startswith('PSOJ352'):
            title_suffix = f"RMS={args.rms:.2f}"
        else:
            title_suffix = f"{args.telescope} {args.t_int:.0f}h"

        plt.title(f"{level*100:.1f}% Credible Posterior Contours {args.target} {title_suffix}")
        plt.grid(alpha=0.2)
        plt.tight_layout()
    
        filename = f"{output_dir}/posterior_sigma_{level}.png"
        plt.savefig(filename, dpi=300, format="png")
        logger.info(f"Saved posterior-sigma plot to {filename}")


def plot_three_posteriors(posteriors, true_thetas):
    colors = ["red", "green", "blue", "orange", "violet"]
    labels = ["xHI=0.24", "xHI=0.51", "xHI=0.80"]

    plt.figure(figsize=(8,6))

    for i, samples in enumerate(posteriors):
        xHI = samples[:,0]
        logfX = samples[:,1]

        sns.kdeplot(
            x=xHI, y=logfX,
            levels=5,
            color=colors[i],
            linewidths=1.5
        )

        # True value
        tx, tl = true_thetas[i]
        plt.scatter(tx, tl, marker="*", s=160,
                    color=colors[i], edgecolor="black")

        # Posterior mean
        plt.scatter(xHI.mean(), logfX.mean(),
                    marker="o", s=100,
                    color=colors[i], edgecolor="black")

    plt.xlim(0, 1)
    plt.ylim(-4, 1)
    plt.xlabel("xHI")
    plt.ylabel("logfX")
    plt.title("SBI NPE Posterior Distributions")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    filename = f"{output_dir}/posterior_levels.png"
    plt.savefig(filename, dpi=300, format="png")
    logger.info(f"Saved posterior-levels plot to {filename}")


if __name__ == "__main__":
    global logger
    global output_dir
    global args
    global device
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.determinisitc=True
    torch.backends.cudnn.benchmark=False

    parser = base.setup_args_parser()

    parser.add_argument('--datapath', type=str, help='PS data path')
    parser.add_argument('--testdatapath', type=str, help='test PS data path')
    parser.add_argument('--training_sample_group_size', type=int, default=10, help='Number of samples of spectrum to be grouped')
    parser.add_argument('--latentdim', type=int, default=512, help='Size of latent feature vector')
    parser.add_argument('--posterior_dir', type=str, default="tmp_out", help='directory to load posterior file from')
    parser.add_argument('--use_xgb_simulator', action='store_true', help='Use XGBoostRegressor as simulator for latent features')

    args = parser.parse_args()

    output_dir = base.create_output_dir(args=args)
    logger = base.setup_logging(output_dir)
    device = (
        "cuda"
        if torch.cuda.is_available()
        #else "mps"
        #if torch.backends.mps.is_available()
        else "cpu"
    )

    logger.info("####")
    logger.info(f"### Using \"{device}\" device ###")
    logger.info("####")

    if args.runmode == "train_test":
        # Load training and test data
        logger.info(f"Loading training data from {args.datapath}...")
        X_train, theta_train = load_training_data(override_path=args.datapath, samples=args.limitsamplesize, args=args)
        logger.info(f"Training data shape: X={X_train.shape}, y={theta_train.shape}")
    
        logger.info(f"Loading test data from {args.testdatapath}...")
        test_sets, true_thetas = load_test_data(override_path=args.testdatapath, samples=args.limitsamplesize, args=args)
        logger.info(f"Test data length: test_sets={len(test_sets)}")
    
        prior = BoxUniform(
            low=torch.tensor([0.0, -4.0], device=device),
            high=torch.tensor([1.0,  1.0], device=device)
        )
        
        if args.use_xgb_simulator:
            logger.info(f"Fitting XGB emulator")
            emulator = MultiOutputRegressor(XGBRegressor())
            emulator.fit(theta_train, X_train)
            logger.info(f"Starting simulation for sbi")
            theta, x = simulate_for_sbi(
                simulator=xgb_simulator,
                prior=prior,
                num_simulations=50000
            )
            logger.info(f"Creating posterior")
            posterior = posterior_using_xgb_sim(x, theta, prior)
        else:
            # Train
            logger.info(f"Training NPE")
            posterior = train_npe(X_train, theta_train, prior)

        logger.info(f"Created posterior")

        posteriors = []
        """
        if args.target.startswith('PSOJ352'):
            true_thetas = [
                (0.24, -3.6),
                (0.51, -3.6),
                (0.80, -3.6)
            ]
        else:
            true_thetas = [
                (0.11, -1.0),
                (0.11, -3.0),
                (0.80, -3.0)
                (0.80, -3.0),
                (0.52, -2.0)
            ]
        """
        
        for test in test_sets:
            samples = infer_posterior_for_test(posterior, test["x"])
            posteriors.append(samples)

        save_posteriors_npy(posteriors, true_thetas)
    else:
        posteriors, true_thetas = load_posteriors_npy(args.posterior_dir)
    
    plot_three_posteriors(posteriors, true_thetas)

    plot_three_posteriors_sigma(posteriors, true_thetas)


