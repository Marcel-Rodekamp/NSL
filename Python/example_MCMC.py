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

    with open('/home/physics/PhD/Projects/NSL/Executables/Examples/example_param.yml', 'r') as f:
        yml = yaml.safe_load(f)

    params = {
        'name': str(yml['system']['name']),
        'beta': float(yml['system']['beta']),
        'Nt': int(yml['system']['Nt']),
        'Nx': int(yml['system']['nions']),
        'U': float(yml['system']['U']),
        'mu': float(yml['system']['mu']) if yml['system']['mu'] else 0.0,
        'offset': float(yml['system']['offset']),
        'save frequency': int(yml['HMC']['save frequency']),
        'Ntherm': int(yml['HMC']['Ntherm']),
        'Nconf': int(yml['HMC']['Nconf']),
        'trajectory length': float(yml['Leapfrog']['trajectory length']),
        'Nmd': int(yml['Leapfrog']['Nmd']),
        'h5file': str(yml['fileIO']['h5file']),
    }

    params['device'] = "GPU" if torch.cuda.is_available() else "CPU"

    #TODO Logger
    for key, value in params.items():
        print(f"{key:20} {value}")
    # print(yml)
    # adjacency = torch.tensor(yml['system']['adjacency']).transpose(0,1)
    # positions = torch.tensor(yml['system']['positions'])
    # s = torch.sparse_coo_tensor(
    #    adjacency,
    #    torch.ones_like(adjacency[0], dtype=float),
    #    size=(len(positions), len(positions)))
    # adjacency = s.to_dense()
    # adjacencyL = adjacency.clone()
    # adjacencyL = adjacencyL.transpose(0,1)
    # adjacency += adjacencyL
    # # print(adjacency)
    # lattice = nsl.Lattice.SpatialLattice(params['name'], adjacency, positions)
    
    lattice = nsl.Lattice.Generic('/home/physics/PhD/Projects/NSL/Executables/Examples/example_param.yml')
    dim = lattice.sites()

    lattice.to(params["device"])

    params['lattice'] = lattice

    with h5.File(params['h5file'], 'w') as h5f:
        basenode = h5f.create_group(params['name'])
        write_meta(params, basenode)
     

    Nx = lattice.sites()

    print(f"Setting up a Hubbard action with beta={np.real(params['beta'])}, Nt={int(params['Nt'])}, U={np.real(params['U'])}, mu={np.real(params['mu'])}, on a {str(params['lattice'])} lattice.")
    hga = nsl.Action.HubbardGaugeAction(params)
    print(hga.eval(torch.ones((params['Nt'], Nx), dtype=torch.float64)))
    hfa = nsl.Action.HubbardFermionAction(params)
    print(hfa.eval())
    action = nsl.Action.HubbardGaugeAction(params)
    print("Action object created from py::dict.")
    print(action.eval(torch.ones((params['Nt'], Nx), dtype=torch.float64)))

def write_meta(params, basenode):
    basenode["/Meta/lattice"] = str(params["lattice"].name)

    basenode["/Meta/params/U"] = complex(params["U"])

    basenode["/Meta/params/beta"] = complex(params["beta"])

    basenode["/Meta/params/Nt"] = np.uint64(params["Nt"])

    basenode["/Meta/params/spatialDim"] = np.uint64(params["lattice"].sites())

    basenode["/Meta/params/nMD"] = np.uint64(params["Nmd"])

    basenode["/Meta/params/saveFreq"] = np.uint64(params["save frequency"])

    basenode["/Meta/params/trajLength"] = float(params["trajectory length"])

    action = "hubbardExp"
    # h5file[basenode + "/Meta/action"] = np.string_(action)

if __name__ == '__main__':
    main()