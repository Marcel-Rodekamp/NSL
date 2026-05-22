# cmake creates a symlink of this script for development purposes. Therefore, the directory of the symlink needs to be identified for some functionality.
import os
import sys
symlink_dir = os.path.dirname(os.path.abspath(__file__))
# add symlink directory to path for PyNSL import
sys.path.insert(0, symlink_dir)

import PyNSL as nsl
import yaml
import h5py as h5
import torch
import numpy as np

def main():

    # Example of how to read parameters from a yaml file
    # with open('../Executables/Examples/example_param.yml', 'r') as f:
        # yml = yaml.safe_load(f)

    # Python dictionary containing HMC parameters
    params = {
        'name': "two_sites",
        'Nx': 2,
        'Nt': 16,
        'beta': 10.,
        'U': 2.,
        'mu': 0.0,
        'offset': 0.0,
        'save frequency': 2,
        'Ntherm': 100,
        'Nconf': 100,
        'trajectory length': 2.,
        'Nmd': 5,
        'h5file': "example.hmc.h5",
    }

    # Set torch device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    # Print parameters
    for key, value in params.items():
        print(f"{key:20} {value}")
    
    # Create lattice from yaml file
    lattice = nsl.Lattice.Generic('../Executables/Examples/example_param.yml')
    Nx = lattice.sites()
    # Move lattice to device
    lattice.to(device)

    # Initialize h5 file and write metadata
    with h5.File(params['h5file'], 'w') as h5f:
        basenode = h5f.create_group(params['name'])
        write_meta(params, basenode)

    # Create Hubbardmodel action as sum of gauge and fermion action
    action = nsl.Action.SumAction(nsl.Action.HubbardGaugeAction(params), nsl.Action.HubbardFermionAction(lattice, params))
    
    # Create starting field
    init_phi = torch.randn((params['Nt'], Nx), device=torch.device(device))
    init_phi = torch.tensor(init_phi, dtype=torch.complex128) + 1.j*params['offset']
    print(init_phi)
    # Set initial configuration
    init_config = {"phi": init_phi}
    
    # Create real valued Leapfrog integrator and HMC
    lfr = nsl.Integrator.LeapfrogRealForce(action, params['trajectory length'].real, params['Nmd'])
    hmc = nsl.MCMC.HMC(lfr, action, params['h5file'])
    
    # Create initial Markov state from initial configuration
    init_state = nsl.MCMC.MarkovState(init_config, action(init_config), 1.)
    
    # Thermalize (i.e. generate beginning of Markov chain but do not save configurations)
    thermalized_state = hmc.thermalize(init_state, params['Ntherm'], params['save frequency'])

    # Generate Markov chain saving Nconf configurations with frequency saveFrequency
    mc = hmc.generate(thermalized_state, Nconf=params['Nconf'], saveFrequency=params['save frequency'])

    # Print field at position 42 of the Markov chain as sanity check
    print(mc[42].configuration["phi"])

def write_meta(params, basenode):
    basenode["Meta/lattice"] = str(params["name"])

    basenode["Meta/params/U"] = complex(params["U"])

    basenode["Meta/params/beta"] = complex(params["beta"])

    basenode["Meta/params/mu"] = complex(params["mu"])

    basenode["Meta/params/Nt"] = np.uint64(params["Nt"])

    basenode["Meta/params/spatialDim"] = np.uint64(params["Nx"])

    basenode["Meta/params/nMD"] = np.uint64(params["Nmd"])

    basenode["Meta/params/saveFreq"] = np.uint64(params["save frequency"])

    basenode["Meta/params/trajLength"] = float(params["trajectory length"])

    action = "hubbardExp"
    # h5file[basenode + "/Meta/action"] = np.string_(action)

if __name__ == '__main__':
    main()