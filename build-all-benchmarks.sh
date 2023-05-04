ml Stages/2022 GCCcore/.11.2.0 PyTorch/1.11-CUDA-11.5 h5py Boost CMake

BASE=/p/home/jusers/pederiva2/jureca/nsl-profiling/NSL
declare -A COMPILER_FLAGS
COMPILER_FLAGS=(["top-level"]="" ["mc-step"]="-DPROFILE_LEVEL_MC_STEP" ["integrator"]="-DPROFILE_LEVEL_MC_STEP -DPROFILE_LEVEL_INTEGRATOR" ["action"]="-DPROFILE_LEVEL_MC_STEP -DPROFILE_LEVEL_INTEGRATOR -DPROFILE_LEVEL_ACTION")


for LEVEL in "top-level" "mc-step" "integrator" "action"
do
    echo "Building $LEVEL Profiling"
    mkdir -p $BASE/build-$LEVEL
    pushd $BASE/build-$LEVEL
    echo "Building NSL"
    rm -rf *
    cmake -S $BASE -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="${COMPILER_FLAGS[$LEVEL]}"
    cmake --build . -j 20 --target NSL

    for BENCH in "L6_MD22" "L9_MD30"
    do
        echo "Building example_MCMC_$BENCH"
        cmake --build . --target example_MCMC_$BENCH
    done
    popd
done