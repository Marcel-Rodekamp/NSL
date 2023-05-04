#!/bin/bash -x
#SBATCH --account=training2310
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4
#SBATCH --reservation=gpuhack23


BASE=/p/home/jusers/pederiva2/jureca/nsl-profiling/NSL
EXE_DIR=$BASE/build/Executables/Examples
ml Stages/2022 GCCcore/.11.2.0 PyTorch/1.11-CUDA-11.5 h5py Boost CMake

# Initialize Folders
mkdir -p $BASE/build
mkdir -p $BASE/profiling

for L in "6" "9"
do
    for MD in "22" "30"
    do
        BENCH="L${L}_MD${MD}"
        echo "Performing Benchmark $BENCH"

        echo "Top Level Profiling"
        pushd $BASE/build
        cmake -S $BASE -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DLATTICE_SIZE=${L} -DMOL_DYN_STEPS=${MD}"
        cmake --build . -j 20
        popd

        mkdir -p $BASE/profiling/$BENCH/top-level
        pushd $BASE/profiling/$BENCH/top-level
        for i in {0..20}
        do
            # srun $EXE_DIR/example_MCMC -p > /dev/null
            touch NSL_profile.log
            mv NSL_profile.log $i.log
        done
        popd

        echo "MC Step Level Profiling"
        pushd $BASE/build
        cmake -S $BASE -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DPROFILE_LEVEL_MC_STEP"
        cmake --build . -j 20
        popd

        mkdir -p $BASE/profiling/$BENCH/mc-step
        pushd $BASE/profiling/$BENCH/mc-step
        for i in {0..20}
        do
            # srun $EXE_DIR/example_MCMC -p > /dev/null
            touch NSL_profile.log
            mv NSL_profile.log $i.log
        done
        popd

        echo "Integrator Level Profiling"
        pushd $BASE/build
        cmake -S $BASE -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DPROFILE_LEVEL_MC_STEP -DPROFILE_LEVEL_INTEGRATOR"
        cmake --build . -j 20
        popd

        mkdir -p $BASE/profiling/$BENCH/integrator
        pushd $BASE/profiling/$BENCH/mc-step
        for i in {0..20}
        do
            # srun $EXE_DIR/example_MCMC -p > /dev/null
            touch NSL_profile.log
            mv NSL_profile.log $i.log
        done
        popd

        echo "Action Level Profiling"
        pushd $BASE/build
        cmake -S $BASE -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DPROFILE_LEVEL_MC_STEP -DPROFILE_LEVEL_INTEGRATOR -DPROFILE_LEVEL_ACTION"
        cmake --build . -j 20
        popd

        mkdir -p $BASE/profiling/$BENCH/integrator
        pushd $BASE/profiling/$BENCH/mc-step
        for i in {0..20}
        do
            # srun $EXE_DIR/example_MCMC -p > /dev/null
            touch NSL_profile.log
            mv NSL_profile.log $i.log
        done
        popd
    done
done

