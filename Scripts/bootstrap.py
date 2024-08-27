import numpy as np
import argparse
import h5py as h5
from pathlib import Path

def get_observables(node, path=''):
    observables = set()
    for key, val in node.items():
        new_path = path + '/' + key if path else key
        if isinstance(val, h5.Dataset):
            observables.add(new_path)
        elif isinstance(val, h5.Group):
            observables.update(get_observables(val, new_path))
    return observables

class Bootstrap:
    def __init__(self, ncfgs, n=200):
        self.n_bs = n
        self.indices = np.random.choice(range(ncfgs), size=(n, ncfgs), replace=True)

    def __call__(self, measurement):
        
        result = np.zeros((self.n_bs, *measurement[0].shape), dtype=measurement.dtype)
        
        for i, bs in enumerate(self.indices):
            result[i] = measurement[bs].mean(axis=0)
            
        return result
    
    def residual(self, measurement):
        
        return self(measurement) - measurement.mean(axis=0)
    
    def residual_squared(self, measurement):
        return self.residual(measurement)**2
    
def main():
    parser = argparse.ArgumentParser(description='Bootstrap resampling')
    parser.add_argument('file', type=Path, help='H5 file with measurements')
    parser.add_argument('-b', '--base', type=str, help='Base path in H5 file', default=None)
    parser.add_argument('-n', '--nbs', type=int, help='Number of bootstrap samples', default=200)
    parser.add_argument('-o', '--observables', nargs='+', help='Observables to bootstrap', type=str, default=None)
    args = parser.parse_args()

    with h5.File(args.file, 'r') as h5f:
        if args.base is None:
            base_keys = list(h5f.keys())
            if len(base_keys) > 1:
                raise ValueError("Multiple base keys found in H5 file. Please specify one with --base")
            else:
                base_path = base_keys[0]
        else:
            base_path = args.base

        if args.observables is None:
            observables = set()
            for cfg in h5f[base_path]["markovChain"].keys():
                observables.update(get_observables(h5f[base_path]["markovChain"][cfg]))
            observables -= set(['acceptanceProbability', 'actVal', 'phi', 'markovTime'])
            observables = sorted(list(observables))
        else:
            observables = args.observables

        data = {obs: [] for obs in observables}
        for cfg in h5f[base_path]["markovChain"].keys():
            for obs in observables:
                if obs in h5f[base_path]["markovChain"][cfg]:
                    data[obs].append(h5f[base_path]["markovChain"][cfg][obs][()])

    results = {}
    for obs in observables:
        data[obs] = np.array(data[obs])
        n = data[obs].shape[0]
        bs = Bootstrap(n, args.nbs)
        data_bs = bs(data[obs])
        results[obs] = {'mean': data_bs.mean(axis=0), 'std': data_bs.std(axis=0)}
        print(obs)
        if "correlators" in obs:
            print("Diagonalizing correlators")
            data_bs = data_bs.reshape(args.nbs, -1, 2, 2)
            data_mean = data_bs.mean(axis=0)
            nt = data_bs.shape[1]
            mean_d = np.zeros((nt, nt, 2, 2), dtype=complex)
            std_d = np.zeros((nt, nt, 2, 2), dtype=complex)
            for t in range(nt):
                _, vecs = np.linalg.eig(data_mean[t])
                diag_bs = np.einsum('xi, ntij, jy->ntxy', np.linalg.inv(vecs), data_bs, vecs)
                mean_d[t] = diag_bs.mean(axis=0)
                std_d[t] = diag_bs.std(axis=0)
            results[obs][f"mean_d"] = mean_d
            results[obs][f"std_d"] = std_d


    with h5.File(args.file, 'a') as h5f:
        node = h5f.require_group(base_path + '/bootstrap')
        for obs in observables:
            if obs in node:
                del node[obs]
            for key, val in results[obs].items():
                node.create_dataset(obs + '/' + key, data=val)

if __name__ == '__main__':
    main()