import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import argparse
from pathlib import Path

def get_observables(node, path=''):
    observables = set()
    for key, val in node.items():
        new_path = path + '/' + key if path else key
        if isinstance(val, h5.Dataset):
            observables.add(path)
        elif isinstance(val, h5.Group):
            observables.update(get_observables(val, new_path))
    return observables

def main():
    parser = argparse.ArgumentParser(description='Plot k-correlators')
    parser.add_argument('files', type=Path, nargs='+', help='H5 file with measurements')
    parser.add_argument('-b', '--base', type=str, help='Base path in H5 file', default=None)
    parser.add_argument('-o', '--observables', type=str, nargs='+', help='Observables to plot', default=None)
    parser.add_argument('-d', '--diagonalize', action='store_true', help='Diagonalize correlators')
    args = parser.parse_args()

    if args.observables is None:
        observables = set()
        for file in args.files:
            with h5.File(file, 'r') as h5f:
                if args.base is None:
                    base_keys = list(h5f.keys())
                    if len(base_keys) > 1:
                        raise ValueError("Multiple base keys found in H5 file. Please specify one with --base")
                    else:
                        base_path = base_keys[0]
                else:
                    base_path = args.base
                observables.update(get_observables(h5f[base_path]["bootstrap"]))
    else:
        observables = args.observables

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for obs in sorted(observables):
        print(obs)
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for ifile, file in enumerate(args.files):
            with h5.File(file, 'r') as h5f:
                if args.base is None:
                    base_keys = list(h5f.keys())
                    if len(base_keys) > 1:
                        raise ValueError("Multiple base keys found in H5 file. Please specify one with --base")
                    else:
                        base_path = base_keys[0]
                else:
                    base_path = args.base
        
                mean = h5f[base_path]["bootstrap"][obs]['mean'][()]
                std = h5f[base_path]["bootstrap"][obs]['std'][()]
                
                mean = mean.reshape(-1, 2, 2)
                std = std.reshape(-1, 2, 2)
                if args.diagonalize:
                    eigs, vecs = np.linalg.eig(mean)
                    # print(eigs, vecs)
                    mean = np.einsum('txi, tij, tjy->txy', np.linalg.inv(vecs), mean, vecs)

                for i in range(2):
                    for j in range(2):
                        ax[i, j].errorbar(np.arange(mean.shape[0]), mean[:, i, j].real, yerr=std[:, i, j], marker='o', markersize=2, color=colors[ifile] ,label=f'{ifile} Real')
                        ax[i, j].errorbar(np.arange(mean.shape[0]), mean[:, i, j].imag, yerr=std[:, i, j], marker='o', markersize=2, color=colors[ifile] ,label=f'{ifile} Imaginary', linestyle='--')
                        ax[i, j].set_title(f'({i}, {j})')
                        ax[i, j].set_xlabel('$\\tau$')
                        ax[i, j].set_ylabel('$C(\\tau)$')
                        ax[i, j].set_yscale('log')
                        ax[i, j].legend()
        fig.suptitle(obs)

        plt.show()


if __name__ == '__main__':
    main()