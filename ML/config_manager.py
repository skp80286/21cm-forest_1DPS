'''
Configuration management module for f21 inference using power spectrum data.
Handles argument parsing and configuration setup.
'''

import f21_predict_base as base


def setup_argument_parser():
    """
    Set up the argument parser with all necessary arguments.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = base.setup_args_parser()
    
    # Add specific arguments for this script
    parser.add_argument('--datapath', type=str, help='PS data path')
    parser.add_argument('--testdatapath', type=str, help='test PS data path')
    parser.add_argument('--training_sample_group_size', type=int, default=10, 
                       help='Number of samples of spectrum to be grouped')
    parser.add_argument('--testing_sample_group_size', type=int, default=10, 
                       help='Number of samples of spectrum to be grouped while testing')
    parser.add_argument('--pstype', type=str, default="noisy", 
                       help='noisy or denoised')
    
    return parser


def set_default_paths(args):
    """
    Set default data paths based on telescope and integration time.
    
    Args:
        args: Parsed arguments object
    """
    if args.datapath is None:
        # Set the datapath based on configuration
        if args.telescope == 'uGMRT' and args.t_int == 50 and args.pstype == 'noisy':
            args.datapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/noisy_g50/f21_ps_dum_train_test_uGMRT_t50.0_20250410153928/ps/"
        elif args.telescope == 'uGMRT' and args.t_int == 500 and args.pstype == 'noisy':
            args.datapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/noisy_500/f21_ps_dum_train_test_uGMRT_t500.0_20250511105815/ps/"
        elif args.telescope == 'SKA1-low' and args.t_int == 50 and args.pstype == 'noisy':
            args.datapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/noisy_ska/f21_ps_dum_train_test_SKA1-low_t50.0_20250511105922/ps/"
        elif args.telescope == 'uGMRT' and args.t_int == 50 and args.pstype == 'denoised':
            args.datapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/denoised_50/mixed_f21_unet_ps_dum_train_test_uGMRT_t50.0_20250607223018/ps/"
        elif args.telescope == 'uGMRT' and args.t_int == 500 and args.pstype == 'denoised':
            args.datapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/denoised_500/mixed_f21_unet_ps_dum_train_test_uGMRT_t500.0_20250604091744/ps/"
        elif args.telescope == 'SKA1-low' and args.t_int == 50 and args.pstype == 'denoised':
            args.datapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/denoised_ska/mixed_f21_unet_ps_dum_train_test_SKA1-low_t50.0_20250608062755/ps/"
    
    if args.testdatapath is None:
        # Set the testdatapath based on configuration
        if args.telescope == 'uGMRT' and args.t_int == 50 and args.pstype == 'noisy':
            args.testdatapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/noisy_g50/f21_ps_dum_train_test_uGMRT_t50.0_20250410153928/test_ps/"
        elif args.telescope == 'uGMRT' and args.t_int == 500 and args.pstype == 'noisy':
            args.testdatapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/noisy_500/f21_ps_dum_train_test_uGMRT_t500.0_20250511105815/test_ps/"
        elif args.telescope == 'SKA1-low' and args.t_int == 50 and args.pstype == 'noisy':
            args.testdatapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/noisy_ska/f21_ps_dum_train_test_SKA1-low_t50.0_20250511105922/test_ps/"
        elif args.telescope == 'uGMRT' and args.t_int == 50 and args.pstype == 'denoised':
            args.testdatapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/denoised_50/mixed_f21_unet_ps_dum_train_test_uGMRT_t50.0_20250607223018/test_ps/"
        elif args.telescope == 'uGMRT' and args.t_int == 500 and args.pstype == 'denoised':
            args.testdatapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/denoised_500/mixed_f21_unet_ps_dum_train_test_uGMRT_t500.0_20250604091744/test_ps/"
        elif args.telescope == 'SKA1-low' and args.t_int == 50 and args.pstype == 'denoised':
            args.testdatapath = "../../../21cm-forest/code/saved_output/train_test_psbs_dump/denoised_ska/mixed_f21_unet_ps_dum_train_test_SKA1-low_t50.0_20250608062755/test_ps/"


def setup_environment():
    """
    Set up the environment with proper random seeds and device configuration.
    
    Returns:
        str: Device to use ('cuda', 'mps', or 'cpu')
    """
    import torch
    import numpy as np
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Determine device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    return device 