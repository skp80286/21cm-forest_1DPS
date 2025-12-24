#!python

import matplotlib as mpl
import matplotlib.pyplot as plt   # data visualization
import seaborn as sns             # statistical data visualization

import logging

import torch

import importlib
import glob
import numpy as np
import plot_results as pltr
import F21DataLoader as dl
import f21_predict_base as base
import F21Stats as f21stats

def plot_los(los_test, samples=1, showplots=False, saveplots=True, label='', output_dir='tmp_out', freq_axis=None):
    
    for i, (noisy) in enumerate(los_test[:samples]):
        if freq_axis is None: freq_axis=range(len(noisy))
        
        plt.rcParams['figure.figsize'] = [5., 2.]
        plt.figure(frameon=True)

        #plt.title(f'{label}')
        #chisq_noisy = np.sum((noisy - test)**2 / test)
        plt.plot(freq_axis, noisy, c='black', linewidth=2.)
        #plt.plot(freq_axis, test, label='Signal', c='orange')
        #chisq_denoised = np.sum((pred - test)**2 / test)
        #plt.plot(freq_axis, pred+0.1, label=f'Denoised+0.1: χ²={chisq_denoised:.2f}')
        #plt.xlabel(r'$\nu_{obs}$[MHz]'), 
        #plt.ylabel(r'$F_{21}=e^{-\tau_{21}}$')
        #plt.legend(loc='best')#lower right')
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        #plt.ylim(0.92,1.05)
        plt.tight_layout()
        if saveplots: 
            plt.savefig(f"{output_dir}/reconstructed_los_{label}.png", format="png", bbox_inches='tight')
            logger.info(f"Saved denoised los plot to {output_dir}/reconstructed_los_{label}.png")
        if i> 5: break
        if showplots: plt.show()
        #print(f'denoising {label}: χ²={chisq_noisy:.2f} χ²={chisq_denoised:.2f}')
        plt.close()

filepath = '/user1/21cm_forest/21cmFAST_los/F21_noisy/'
rms = 3.3
xs=[0.25, 0.52, 0.8]
fs=[-3.6]
txs=[0.24, 0.51, 0.8]

for x in xs:
    for f in fs:
        files = glob.glob(f"{filepath}/F21_signalonly_21cmFAST_200Mpc_z6.0_fX{f:.2f}_xHI{x:.2f}_uGMRT_PSOJ352*15_rms{rms:.4f}mJy_6.1kHz.dat")
        print(f"Found {len(files)} files.")
        los_so, _,  _,  _, freq_axis = base.load_dataset(files, max_workers=1, psbatchsize=1, limitsamplesize=1000, save=False, skip_ps=True)
        print(f"x={x}, f={f}, so: shape:{los_so.shape}")
        pltr.plot_denoised_los(los_so[10:11], None, None, showplots=True, saveplots=True, freq_axis=freq_axis[0]/1e6, x=x, f=f, label='Training ')

for x in txs:
    for f in fs:
        filepattern = f"{filepath}/F21_signalonly_21cmFAST_200Mpc_z6.0_fX{f:.2f}_xHI{x:.2f}_uGMRT_PSOJ352*15_rms{rms:.4f}mJy_6.1kHz*.dat"
        print(f"loading files: {filepattern}")
        files = glob.glob(filepattern)
        print(f"Found {len(files)} files.")
        los_so, _,  _,  _, freq_axis = base.load_dataset(files, max_workers=1, psbatchsize=1, limitsamplesize=1000, save=False, skip_ps=True)
        print(f"x={x}, f={f}, so: shape:{los_so.shape}")
        pltr.plot_denoised_los(los_so[10:11], None, None, showplots=True, saveplots=True, freq_axis=freq_axis[0]/1e6, x=x, f=f, label='Testing ')

