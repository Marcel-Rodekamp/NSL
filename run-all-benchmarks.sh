#!/bin/bash
#SBATCH --account=training2310
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4

ml Stages/2022 GCCcore/.11.2.0 PyTorch/1.11-CUDA-11.5 h5py Boost CMake

BASE=$(pwd)

for LEVEL in "top-level" "mc-step" "integrator" "action"
do
    echo "$LEVEL Profiling"
    for BENCH in "L6_MD22" "L9_MD30"
    do
        mkdir -p $BASE/profiling/$BENCH/$LEVEL
        pushd $BASE/profiling/$BENCH/$LEVEL
        for i in {0..20}
        do
            echo "Running example_MCMC_$BENCH Trial $i"
            srun $BASE/build-$LEVEL/Executables/Examples/example_MCMC_$BENCH -p
            #$BASE/build-$LEVEL/Executables/Examples/example_MCMC_$BENCH -p
	    touch NSL_profile.log
            mv NSL_profile.log $i.log
        done
        popd
    done
done
