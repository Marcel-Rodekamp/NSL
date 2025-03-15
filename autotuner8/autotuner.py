'''
Author: Philippos Papaphilipou (original version for CNS)
        Petar Sinilkov new version adapted for NSL

Notes: 
- If numpy, scipy or matplotlib are not available in the nodes, a local install
  of python is an alternative solution, such as with the Anaconda installer. 
- In case the model fails to converg, try different initial guess numbers, different CDFs,
  such as alpha.cdf((x+a[0])*a[1], a[2]) from scipy.stats, or different ftting parameters
- For notes on the confidence interval and fitting method see the _functions.py file 
'''

from scipy.optimize import least_squares
from skew_normal import cdf_skewnormal
import numpy as np
np.printoptions(precision=3, linewidth=400)
from _functions import *
import os
import sys
import subprocess
import math
import pickle
import scipy.stats
import psutil
import yaml
import h5py as h5
sys.dont_write_bytecode = True  # to avoid pyc files

fitting_verbose_level = 1

np.random.seed(1)

import matplotlib as mpl
mpl.use('Agg') # To work without Xserver
from matplotlib import pyplot as pl

graph_dir = "./autotune-graphs/"
out_dir = "./stdout/"
meas_dir, conf_dir, base_name = "", "", ""

base_name = "C60"
conf_dir = "thermal"

optimal_acceptance = 0.75
significance_interval = 0.25

min_trajectories = 10
max_trajectories = 50

max_counter = 1000

# p = psutil.Process()
# cpuList = p.cpu_affinity()
# print("# cpu list = ",cpuList)

import argparse
parser = argparse.ArgumentParser() #help='Process data down to a manageable size by cutting, binning, and bootstrapping.'
parser.add_argument("yaml", type=str, help="Initialization YAML file")
parser.add_argument("executable", type=str, help="NSL Executable")
parser.add_argument("--GPU", default=False, action='store_true', help="Using GPU")
parser.add_argument("--steps", type=int, default=25, help="Number of steps")
parser.add_argument("--overwrite", default=False, action='store_true', help="Overwrite existing data")
args = parser.parse_args()

with open(args.yaml) as stream:
    ymlFile = yaml.safe_load(stream)

h5file = ymlFile["fileIO"]["h5file"]
if os.path.exists(h5file) and args.overwrite:
    os.remove(h5file)

base_name = ymlFile["system"]["name"]
conf_dir = "thermal"

binary, conffile = args.executable, args.yaml

trajectories_count = 0

enable_graphs = True

hasenbusch_timescale = 1

stage_counter = 0

# Prepare the autotuner file
last_guess = None

graph_counter = 0
counter_stopping_condition = 0
steps = args.steps
all_steps = []
last_counter = 0
all_accept_rates = dict()

tuneFlag = 0
os.system(f"mkdir -p {graph_dir}")

def plot_fit(independent_var, dependent_var, errors, u_test, y_test, next_steps):
    pl.errorbar(independent_var, dependent_var, yerr=errors, fmt='o', markersize=4, label='data', capsize=2)
    #pl.plot(independent_var[-2:], dependent_var[-2:], 'o', markersize=4, label='art. data')
    pl.plot(u_test, y_test, label='fitted model')
    pl.axhline(y=optimal_acceptance, color='r', linestyle='--', label='optimal')
    pl.axvline(x=next_steps, color='g', linestyle='--', label=f'next steps {next_steps}')
    pl.xlabel('Number of steps')
    pl.ylabel('Acceptance rate')
    pl.ylim(-0.2, 1.2)
    pl.xlim(-2, sorted(independent_var)[-2]+2)
    pl.legend(loc='lower right')
    pl.grid(color='#dddddd', linestyle='--', linewidth=0.5)

    pl.gcf().set_size_inches(12.14, 7.5)
    global graph_counter
    figname = f"{graph_dir}/{ymlFile['fileIO']['h5file'].split('/')[-1].replace('.h5','')}-{str(graph_counter).zfill(2)}.pdf"
    pl.savefig(figname, dpi=None, facecolor='w', edgecolor='w',
            format='pdf', transparent=False, bbox_inches=None, pad_inches=0.1)#, rameon=None)
    pl.clf()
    # graph_counter += 1

if os.path.exists(ymlFile["fileIO"]["h5file"]):
    with h5.File(ymlFile["fileIO"]["h5file"], 'r') as f:
        try:
            tuneFlag = f[f"{base_name}/Meta/params/tuneFlag"][()]
            steps = f[f"{base_name}/Meta/params/nMD"][()]
            stage_counter = f[f"{base_name}/autotune/stage-counter"][()]
            last_guess = f[f"{base_name}/autotune/last-guess"][()]
            for k,v in f[f"{base_name}/autotune/all-accept-rates"].items():
                all_accept_rates[k] = v[()]
            min_trajectories = 7
            max_trajectories = 50
            significance_interval = 0.20
        except KeyError:
            last_guess = None

            graph_counter = 0
            counter_stopping_condition = 0        

        last_counter = len(f[f"{base_name}/{conf_dir}"])

        graph_counter = len(all_steps)

if tuneFlag == 0:

    probabilities_at_nmd = {0: None, 1: None, 200: None}
    expectations_at_nmd = {0: 0., 1: 0., 200: 1.}
    errors_at_nmd = {0: 0.0001, 1: 0.1, 200: 0.01}
    start = last_counter

    while start < max_counter:
        # Run
        # p.cpu_affinity(cpuList)
        # result = subprocess.run([f"{binary}", "--file", f"{args.yaml}"], check=True)
        result = subprocess.run([f"{binary}", "--file", f"{args.yaml}", *(["--GPU"] if args.GPU else []) ], check=True)
        print(result)

        # os.system(f"./{binary} --file {args.yaml}")
        # p.cpu_affinity([cpuList[0]])
        # counter += 1

        # Count accepted traj.
        probabilities = []
        accepted_prob_total = 0

        with h5.File(ymlFile["fileIO"]["h5file"], 'a') as f:
            if f[f"{base_name}/Meta/params/thermalFlag"][()] == 0:
                f[f"{base_name}/Meta/params/thermalFlag"][...] = 1
            trajectories = f[f"{base_name}/{conf_dir}"].keys()
            # trajectories = sorted([int(traj) for traj in trajectories])
            counter = len(trajectories)
            for traj in range(start, counter):
                acceptance_probability = min(1, f[f"{base_name}/{conf_dir}/{traj}/acceptanceProbability"][()])
                # acceptance_rate = f[f"{base_name}/{conf_dir}/{traj}/acceptanceRate"][()]

                # if traj >= counter - trajectories_count:
                probabilities.append(acceptance_probability)
                    # accepted_count += acceptance_rate

                accepted_prob_total += acceptance_probability
                # accepted_count_total += acceptance_rate
        start = counter
        # accept_rate = accepted_count/trajectories_count
        # accept_rate_total = accepted_count_total/counter
        # accept_rate_prob = np.mean(probabilities)
        # conf_int = wald_interval(accepted_count, trajectories_count, 0.95)
        
        if probabilities_at_nmd.get(steps) is None:
            probabilities_at_nmd[steps] = probabilities
        else:
            probabilities_at_nmd[steps] += probabilities

        try:
            probabilities = np.array(probabilities_at_nmd[steps])
        except:
            print(probabilities_at_nmd[steps])
            raise ValueError("Error in probabilities array")
        accept_rate_prob = probabilities.mean()
        conf_int_prob = conf_int_N_draws(
            accept_rate_prob, probabilities.std(), len(probabilities), 0.95)
        if probabilities.std() < 1e-6:
            conf_int_prob = (0.01, accept_rate_prob, accept_rate_prob)

        expectations_at_nmd[steps] = accept_rate_prob
        errors_at_nmd[steps] = conf_int_prob[0]
        # info_string = "Traj: %d  acc.r.: (bin)%f (prob)%f | Current: steps: %d  traj: %d  acc.r/conf.int: (bin)%f/%f(%f,%f) (prob)%f/%f(%f,%f)"
        # values_to_print = (counter, accept_rate_total, accept_rate_prob_total, steps, trajectories_count,
        #                 accept_rate, conf_int[0], conf_int[1], conf_int[2], accept_rate_prob, conf_int_prob[0], conf_int_prob[1], conf_int_prob[2])
        print(f"At Nmd={steps}: expected probability: {accept_rate_prob:.2%} \nconfidence interval: [{conf_int_prob[1]:.2%} {conf_int_prob[2]:.2%}]")
        
        # Try to change the number of steps
        if optimal_acceptance < conf_int_prob[1] or optimal_acceptance > conf_int_prob[2]:

            # Initial guesses for fitting, e.g based on a result of a small lattice
            guesses = []
            guesses.append(np.array([-10, .2, 0.5]))
            guesses.append(
                np.array([-9.27128658e+01,   2.16389117e-02,   7.24033137e-01]))
            guesses.append(np.array([2e-1 for i in range(3)]))
            try:
                if last_guess is not None:
                    guesses.append(last_guess)
            except ValueError:
                raise ValueError(f"Last guess is not a valid array: {last_guess}")
                # guesses.append(last_guess)

            x, y = zip(*sorted(expectations_at_nmd.items()))
            independent_var = np.array(x)
            dependent_var = np.array(y)
            errors = np.array([errors_at_nmd[steps] for steps in x])
            # Utilize the Levenberg-Marquardt algorithm by using the scipy library
            best_res = least_squares(function, guesses[0], ftol=1e-10, gtol=1e-10, xtol=1e-10, args=(
                independent_var, dependent_var, errors), verbose=fitting_verbose_level, loss='soft_l1', max_nfev=1000)
            for x0 in guesses[1:]:
                res = least_squares(function, x0, ftol=1e-10, gtol=1e-10, xtol=1e-10, args=(
                    independent_var, dependent_var, errors), verbose=fitting_verbose_level, loss='soft_l1', max_nfev=1000)
                if res.cost < best_res.cost:
                    best_res = res
            last_guess = res.x

            # Get a set of predictions from the model
            u_test = np.linspace(0, 2000, 20000)
            y_test = model_to_fit(best_res.x, u_test)


            # Nmd selection based on the model
            next_steps = steps
            diff = abs(accept_rate_prob-optimal_acceptance)
            for step in range(len(u_test)):
                if abs(y_test[step]-optimal_acceptance) < diff and (u_test[step] > steps or optimal_acceptance < conf_int_prob[2]):
                    diff = abs(y_test[step]-optimal_acceptance)
                    next_steps = int(
                        u_test[step]/hasenbusch_timescale+0.5)*hasenbusch_timescale
            if next_steps == steps:
                if optimal_acceptance < conf_int_prob[1]:
                    next_steps = min(steps-1, int(steps/1.2))
                else:
                    next_steps = max(steps+1, int(steps*1.2))
            steps = next_steps
            print("New Nmd selected from model: ".upper() + str(steps))
            with h5.File(ymlFile["fileIO"]["h5file"], 'a') as f:
                f[f"{base_name}/Meta/params/nMD"][...] = steps
            if counter_stopping_condition >= 3 or (all_steps.count(steps) > 3 and optimal_acceptance < conf_int_prob[2]):
                if stage_counter != 0:
                    print("Stopping condition already met.")
                    break
                stage_counter += 1

                with h5.File(ymlFile["fileIO"]["h5file"], 'a') as f:
                    f[f"{base_name}/Meta/params/thermalFlag"][...] = 0
                    f.create_group(f"{base_name}/autotune/all-accept-rates")
                    f[f"{base_name}/autotune/stage-counter"] = stage_counter
                    f[f"{base_name}/autotune/last-guess"] = np.asarray(last_guess)
                    for k,v in all_accept_rates.items():
                        f[f"{base_name}/autotune/all-accept-rates"].create_dataset(str(k), data=v)

                result = subprocess.run([f"{binary}", "--file", f"{args.yaml}"], check=True)
                print(result)

                with h5.File(ymlFile["fileIO"]["h5file"], 'a') as f:
                    f[f"{base_name}/Meta/params/thermalFlag"][...] = 1
                    print(f[f"{base_name}/Meta/params/thermalFlag"][...])

                counter_stopping_condition = 0
                all_steps = []
                min_trajectories = 7
                max_trajectories = 50
                significance_interval = 0.20

            # Plot fitting graphs
            if enable_graphs is True:
                plot_fit(independent_var, dependent_var, errors, u_test, y_test, next_steps)
                graph_counter += 1
        else:
            print(f"Optimal acceptance rate {optimal_acceptance:.2%} is within the confidence interval [{conf_int_prob[1]:.2%} {conf_int_prob[2]:.2%}]")
            if conf_int_prob[0] < 0.065:
                print(f"Confidence is sufficient. Stopping condition met.")
                try:
                    graph_counter = "final"
                    plot_fit(independent_var, dependent_var, errors, u_test, y_test, steps)
                except:
                    print("Plotting failed")
                break
            elif len(probabilities) > 500:
                print(f"Tuner got stuck at {steps}. Stopping here.")
                break
            else:
                print(f"Confidence is not sufficient yet {conf_int_prob[0]:.2%} > 6.5%. Generating more data.")
                x, y = zip(*sorted(expectations_at_nmd.items()))
                independent_var = np.array(x)
                dependent_var = np.array(y)
                errors = np.array([errors_at_nmd[steps] for steps in x])
                try:
                    plot_fit(independent_var, dependent_var, errors, u_test, y_test, steps)
                except:
                    print("Plotting failed")

        


    with h5.File(ymlFile["fileIO"]["h5file"], 'a') as f:
        f[f"{base_name}/Meta/params/tuneFlag"][...] = 1

    print("Final Nmd: " + str(steps))

result = subprocess.run([f"{binary}", "--file", f"{args.yaml}", *(["--GPU"] if args.GPU else []) ], check=True)
print(result)
