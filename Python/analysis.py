import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl

import gvar as gv
from tqdm import tqdm
from lsqfit import nonlinear_fit
import scipy.optimize

from dataclasses import dataclass,field
from pathlib import Path
from time import time_ns
import numpy as np
import h5py as h5
import itertools
import logging
import yaml
from correlator import CorrelatorData, diagonalize_unitary
from fitter import Fitter

logger =  logging.getLogger(__name__)  

# Define the FZJ color scheme
FZJ_COLORS = {
    "darkgreen": ( 10/255, 93/255,  0/255),
    "darkred"  : (219/255, 49/255, 49/255),
    "blue"     : (  2/255, 61/255,107/255),
    "lightblue": (173/255,189/255,227/255),
    "violet"   : (175/255,130/255,185/255),
    "gold": (255/255, 215/255, 0/255),
    "purple": (128/255, 0/255, 128/255),
    "brown"    : (165/255, 42/255, 42/255),
    "olive": (128/255, 128/255, 0/255),
    "maroon": (128/255, 0/255, 0/255),
    "silver": (192/255, 192/255, 192/255),
    "black"    : (  0/255,  0/255,  0/255),
    "magenta": (255/255, 0/255, 255/255),
    "orange"   : (250/255,180/255, 90/255),
    "pink"     : (255/255,105/255,180/255),
    "teal": (0/255, 128/255, 128/255),
    "cyan": (0/255, 255/255, 255/255),
    "navy": (0/255, 0/255, 128/255),
    "gray"     : (215/255,215/255,215/255),
    "yellow"   : (250/255,235/255, 90/255),
    "red"      : (235/255, 95/255,115/255),
    "green"    : (185/255,210/255, 95/255),
}

# use the defined colors as default colors in matplotlib
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=list(FZJ_COLORS.values()))


def fname(*, param, path='.', name='Perylene', ext='.h5'):
    """
        params:
            - param: result of the argument parser or isle.util.parameters

        For convenience, the single star ensures that fname is only called
        via keyword arguments
    """
    if path[-1] == '/':
        path = path[:-1]
    if ext[0] != '.':
        ext = '.' + ext
    return f"{path}/{name}_" + f"Nt{param.Nt}_" + f"beta{param.beta:g}_" + f"U{param.U:g}_" + f"mu{param.mu:g}" + ext


def read_data(h5path, Nt, Nx, Nconf, isIsleData = False):
    logger.info("Reading data...")

    C_sp = np.zeros((Nconf,Nt,Nx,Nx),dtype=complex)
    C_sh = np.zeros((Nconf,Nt,Nx,Nx),dtype=complex)
    actVals = np.zeros(Nconf,dtype=complex)
    
    if isIsleData:
        with h5.File(h5path,'r') as h5f:
            logger.info("Reading Action...")
            actVals = h5f["weights/actVal"][0:Nconf]

            logger.info("Reading C_sp...")
            C_sp = np.moveaxis(h5f["correlation_functions/single_particle/destruction_creation"][0:Nconf,:,:,:],3,1)
            C_sh = np.moveaxis(h5f["correlation_functions/single_hole/destruction_creation"][0:Nconf,:,:,:],3,1)
    else:
        actVals = np.zeros(Nconf,dtype=complex)
        C_sp = np.zeros((Nconf,Nt,Nx,Nx),dtype=complex)
        C_sh = np.zeros((Nconf,Nt,Nx,Nx),dtype=complex)
        with h5.File(h5path,'r') as h5f:
            for n in range(Nconf):
                actVals[n] = h5f[f"twoSite/markovChain/{n}/actVal"][()]
                C_sp[n,:,:,:] = h5f[f"twoSite/markovChain/{n}/correlators/single/particle"][()].reshape(Nt,Nx,Nx)
                C_sh[n,:,:,:] = h5f[f"twoSite/markovChain/{n}/correlators/single/hole"][()].reshape(Nt,Nx,Nx)

    return C_sp, C_sh, actVals

def plot_stats(data):
    logger.info("Plotting HMC Statistics")
    fig,axs = plt.subplots(2,2,figsize=(16,9))

    axs[0,0].plot(np.arange(data.Nconf),data.correlator.actVals.real, '.', color = FZJ_COLORS["blue"], label=r"$\Re{S}$")
    axs[0,0].plot(np.arange(data.Nconf),data.correlator.actVals.imag, '.', color = FZJ_COLORS["green"], label=r"$\Im{S}$")
    ax_hist = axs[0,0].inset_axes([1.0, 0, 0.25, 1], sharey=axs[0,0])
    ax_hist.hist(data.correlator.actVals.real, bins=20, color=FZJ_COLORS["blue"], alpha=0.5,orientation='horizontal')
    ax_hist.hist(data.correlator.actVals.imag, bins=20, color=FZJ_COLORS["green"], alpha=0.5,orientation='horizontal')
    ax_hist.set_xticks([])
    ax_hist.tick_params(axis='y', labelleft=False)
    ax_hist.tick_params(axis='x', labelbottom=False)

    axs[0,0].set_xlabel(r"Measurement ID",fontsize=16)
    axs[0,0].set_ylabel(r"Action Value",fontsize=16)

    axs[0,0].legend()

    Sigma = gv.gvar(data.statistical_power_est[-1], data.statistical_power_err[-1])

    axs[1,0].errorbar( data.NconfCuts, data.statistical_power_est, yerr=data.statistical_power_err, color=FZJ_COLORS["lightblue"], capsize=2,
        label = r"$\left\vert \left\langle\Sigma\right\rangle\right\vert_{N_\mathrm{cut}}$"
    )
    axs[1,0].axhline( y=data.statistical_power_est[-1] , color=FZJ_COLORS["blue"], label = rf"$\left\vert \left\langle\Sigma\right\rangle\right\vert = {Sigma}$" )
    axs[1,0].axhspan( 
        ymin = data.statistical_power_est[-1] - data.statistical_power_err[-1], 
        ymax = data.statistical_power_est[-1] + data.statistical_power_err[-1],
        color=FZJ_COLORS["blue"],
        alpha=0.5
    )
    axs[1,0].set_xlabel(r"$N_\mathrm{conf}$-cut off",fontsize=16)
    axs[1,0].set_ylabel(r"$\left\vert \left\langle\Sigma\right\rangle\right\vert $",fontsize=16)
    axs[1,0].legend()


    axs[1,1].set_axis_off()
    axs[1,1].text(0.05,0,
        rf"""
        Perylene Simulation
            - $N_\mathrm{{conf}} = {data.Nconf}$
            - $N_\mathrm{{bst}} = {data.Nbst}$
            - $N_t={data.Nt}$
            - $\beta={data.beta:g}$
            - $U={data.U:g}$
            - $\mu={data.mu:g}$
            - $Q = {gv.gvar(data.charge_est, data.charge_err)}$
        """,
        fontsize = 16
    )


    fig.suptitle(rf"HMC Statistics",fontsize=20)

    fig.tight_layout()

    fig.savefig(fname(param=data, path="./", name="Simulation_Stats", ext=".pdf") )

def plot_C_sp(data):
    logger.info("Plotting Correlation Functions")
    fig,axs = plt.subplots(2,1,figsize=(16,9))

    abscissa = np.arange(data.Nt) * data.delta

    for k in range(data.Nx):
        erb = axs[0].errorbar(abscissa, data.C_sp_est[:,k], yerr=data.C_sp_err[:,k], fmt='.:',capsize=2)
        axs[1].errorbar(abscissa[1:]-data.beta/2, data.Cs_sp_est[:,k], yerr=data.Cs_sp_err[:,k], fmt='.:',capsize=2,color=erb[0].get_color()) 

    axs[0].set_xlabel(r"$\tau$",fontsize=16)
    axs[0].set_ylabel(r"$C_\mathrm{sp}(\tau)$",fontsize=16)
    axs[0].set_yscale('log')

    axs[1].set_xlabel(r"$\tau - \frac{\beta}{2}$",fontsize=16)
    axs[1].set_ylabel(r"$\frac{C_\mathrm{sp}(\tau) +C_\mathrm{sp}(-\tau)}{2}$",fontsize=16)
    axs[1].set_yscale('log')
    
    title = rf"$N_t = {data.Nt:g}, \, \beta = {data.beta:g}, \, U = {data.U:g}, \, \mu = {data.mu:g}$"

    fig.suptitle(rf"Single Particle Correlator: {title}", fontsize= 20 )

    fig.tight_layout()

    fig.savefig(fname(param=data, path="./", name="C_sp", ext=".pdf") )

def plot_meff(data):
    logger.info("Plotting Effective Masses")
    fig,axs = plt.subplots(2,1,figsize=(16,9))

    abscissa = np.arange(0,data.Nt) * data.delta

    legend_handles = []
    meff_list = []
    for k in range(data.Nx):
        shift = 0# (k - data.Nx//2) * data.delta / data.Nx
        ebars = [None,None]
        ebars[0] = axs[0].errorbar(
            abscissa[1:-1]+shift, 
            data.meff_exp_est[1:-1,k], 
            yerr=data.meff_exp_err[1:-1,k], 
            fmt='.:',capsize=2
        )
        ebars[1] = axs[1].errorbar(
            abscissa[1:-2]+data.delta-data.beta/2+shift, 
            data.meff_cosh_est[:,k], 
            yerr=data.meff_cosh_err[:,k], 
            fmt='.:',capsize=2
        )

        # find best fit
        nState_bestfit = 0
        tstart_bestfit = 0
        tend_bestfit = 0
        AIC_bestfit = np.inf
        useCoshModel = None
        sign = None
        # key: {nState}/{tstart}/{tend}/{k}/{res or useCoshModel}
        for key,item in data.fit_results.items():
            nState,tstart,tend,k_,element = key.split("/")

            nState = int(nState)
            tstart = int(tstart)
            tend = int(tend)
            k_ = int(k_)

            if k_ != k: continue
            if element != "res": continue

            AIC = data.fit_params_est[f"{nState}/{tstart}/{tend}/{k}/AIC"]

            if AIC < AIC_bestfit:
                nState_bestfit = nState
                tstart_bestfit = tstart
                tend_bestfit   = tend
                AIC_bestfit    = AIC

        useCoshModel =  data.fit_results[f"{nState_bestfit}/{tstart_bestfit}/{tend_bestfit}/{k}/useCoshModel"]
        axsIndex = 1 if useCoshModel else 0

        sign = -data.fit_results[f"{nState_bestfit}/{tstart_bestfit}/{tend_bestfit}/{k}/sign"]

        # This is |E0|/delta
        E0 = gv.gvar(
            data.modelAverage_est["E0"][k],
            data.modelAverage_err["E0"][k]
        ) / data.delta

        logger.info(f"{k=}: {E0}")

        label_ = rf"E_0"
        if useCoshModel:
            label = rf"${label_} = {sign*E0}$"
        else:
            # E0 only get's a sign in the label but not in the plot!
            E0 = sign*E0
            label = rf"${label_} = {E0}$"

        handle = axs[axsIndex].axhline( 
            y = E0.mean, 
            xmin = tstart_bestfit/len(abscissa),
            xmax = (tend_bestfit-1)/len(abscissa),
            color=ebars[axsIndex][0].get_color(), 
            label = label
        )

        axs[axsIndex].axhspan( 
            ymin = E0.mean + E0.sdev, 
            ymax = E0.mean - E0.sdev,
            xmin = tstart_bestfit/len(abscissa),
            xmax = (tend_bestfit-1)/len(abscissa),
            color=handle.get_color(),
            alpha=0.2
        )

        legend_handles.append(handle)
        if useCoshModel:
            meff_list.append(sign*E0.mean)
        else:
            # E0 only get's a sign in the label but not in the plot!
            meff_list.append(E0.mean)

    order = np.argsort(meff_list)

    axs[0].set_xlabel(r"$\tau$",fontsize=16)
    axs[0].set_ylabel(r"$\frac{m_\mathrm{eff}^\mathrm{exp}(\tau)}{\delta}$",fontsize=16)
    axs[1].set_xlabel(r"$\tau - \frac{\beta}{2}$",fontsize=16)
    axs[1].set_ylabel(r"$\frac{m_\mathrm{eff}^\mathrm{cosh}(\tau)}{\delta}$",fontsize=16)
    #axs.set_yscale('symlog')

    axs[1].legend(handles=[legend_handles[i] for i in order], ncol=4, bbox_to_anchor=(0.5, -0.2), loc="upper center")  # Adjust the bbox_to_anchor and loc
    
    fig.tight_layout()

    fig.savefig(fname(param=data, path="./", name="meff", ext=".pdf") )

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='Analysis %(asctime)s~> %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
    logger.info("Start")


    Nbst = 100

    with open("example_param.yml", 'r') as file:
        meta_yml = yaml.safe_load(file)

        Nt = meta_yml['system']['Nt']
        beta = meta_yml['system']['beta']
        U = meta_yml['system']['U']
        mu = meta_yml['system']['mu']
        Nx = meta_yml['system']['nions']
        Nconf = meta_yml['HMC']['Nconf']

        adj = meta_yml['system']['adjacency']
        hop = meta_yml['system']['hopping']
        kappa = np.zeros((Nx,Nx))

        for x in range(len(adj)):
            y,z = adj[x]

            kappa[y,z] = hop
        kappa += kappa.T

    e,u = np.linalg.eigh(kappa)
    
    C_sp, C_sh, actVals = read_data("example.h5", Nt, Nx, Nconf, False)

    corrData = CorrelatorData(
            Nt = Nt,
            beta = beta,
            U = U,
            mu = mu,
            Nx = Nx,
            Nconf = Nconf,
            actVals = actVals,
            Nbst = Nbst,
            C_sp = C_sp,
            C_sh = C_sh,
            diagonalize = lambda C: diagonalize_unitary(C, amplitudes=u)
    )

    fitter = Fitter(corrData)

    plot_stats(fitter)
    
    plot_C_sp(corrData)

    plot_meff(fitter)

    logger.info("End")
