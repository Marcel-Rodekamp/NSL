# Nanosystem Simulation Library

Nanosystem Simulation Library (NSL) implements statistical simulations for systems on the nanoscale.

This library provides a merge of the two software packages CNS and [isle](https://github.com/evanberkowitz/isle).
Additional features become added to provide a self-containing simulation library for nanostructures such as

<!-- Add all the systems we have implemented here -->
* Graphene
* Carbon Nanotubes

## Installation

The default branch is the `devel` branch. 
Only tested and working features are merged into this. 
It is ment as a starting point for the development of new features.
The `main` branch holds the newest major version of NSL.
While each major version contains its separate branch as `release/MajorVersionID.0`.
Subreleases, if they appear may contain bug fixes and can be found at 
`release/MajorVersionID.n` where n counts upwards from 1.

### Quick Build

The quick build instruction assume that all prerequisites, detailed below, are already met.

1. Obtain a copy of the library
```
git clone git@github.com:Marcel-Rodekamp/NSL.git && cd NSL
```
2. Create a build directory and change into it
```
mkdir build && cd build
```
3. Call cmake (cmake version ≥ 3.18) to generate the build files
   * Possible build types:
     * `Debug`: Build for developing and debugging, turn on runtime assertions
     * `Release`: Build for production, turn off runtime assertions
``` 
cmake -DCMAKE_BUILD_TYPE=Debug ..
```
4. Build the directory (add `-j 4` for parallel build on 4 cores)
```
make
```

### Build on macOS with Homebrew (tested)

These commands were run exactly as written on a fresh clone on 2026-11-17:
macOS 26.6 on Apple silicon, Homebrew `cmake` 4.4.5, `ninja` 1.13.2, `hdf5` 2.2.0,
`pytorch` 2.15.0, Apple clang 21. Configuring took 27 s, compiling everything 6 min on 8 cores,
the tests 2.5 min.

```sh
# 1. tools and libraries (once); Apple's compiler comes with the Xcode command line tools
#    (`xcode-select --install` if `clang --version` fails)
brew install cmake ninja hdf5 pytorch

# 2. the code (or cd into your existing checkout)
git clone https://github.com/Marcel-Rodekamp/NSL.git
cd NSL
git checkout newMeasurements

# 3. configure a Release build in ./build; this downloads Catch2, HighFive, spdlog,
#    yaml-cpp and CLI11 on the first run
cmake --fresh -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DTorch_DIR="$(brew --prefix)/share/cmake/Torch" \
      -DHIGHFIVE_USE_BOOST=OFF \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5

# 4. compile everything ...
cmake --build build -j 8
#    ... or only the programs you need
cmake --build build -j 8 --target propagator autotune_RadialMCMC_PPP

# 5. run the tests
ctest --test-dir build -j 4
```

The programs are then in `build/Executables/` (e.g. `build/Executables/Hubbard/propagator`,
`build/Executables/PPP/autotune_RadialMCMC_PPP`), the tests in `build/Tests/`.
After `git pull`, step 4 alone recompiles what changed.

Why the options:

* `Torch_DIR` points CMake to Homebrew's `libtorch`.
* `HIGHFIVE_USE_BOOST=OFF`: otherwise HighFive looks for Boost components that NSL does not use.
* `CMAKE_POLICY_VERSION_MINIMUM=3.5`: CMake 4 refuses the old `cmake_minimum_required` of the
  downloaded HighFive; this lets it configure anyway.
* `--fresh` discards an existing CMake cache. Use it (i.e. repeat steps 3 and 4) after
  `brew upgrade` has updated `pytorch`, `hdf5` or their dependencies; an old build directory
  otherwise links against libraries that no longer exist.
* CMake picks Apple's compiler (`/usr/bin/c++`) even when Homebrew's LLVM `clang++` comes first in
  `PATH`; that is the tested setup. If you have set `CC`/`CXX` to another compiler, unset them.

Known test failures (also before the changes on `newMeasurements`): `test_Tensor_mat_exp`,
`test_FermionMatrix_hermitian`, `test_fermionMatrixHubbardExp` and `test_force` fail in their
single-precision (`complex<float>`) cases, `test_BiCGStab` breaks down with NaN residuals, and
`test_CG`, `test_LinAlg_matvec` and `test_leapfrog` fail now and then because they use unseeded
random input.

The Python tools in `Executables/PPP/python` need
`python3 -m pip install numpy scipy h5py pyyaml matplotlib`.

### Prerequisites

Sometimes a few things are needed before the build can work.
We utilize features from C++20 which requires more recent compilers. 
Currently, NSL is tested to work with 

1. `g++ (GCC) 10.3.0`
2. `g++ (GCC) 11.1.0`
3. `clang++ 13.0.0` with targets:
   * `x86_64-apple-darwin20.6.0`
   * `arm64-apple-darwin20.6.0`

Additionally, the backend depends on [PyTorch's C++ ABI `libtorch`](https://pytorch.org/cppdocs/installing.html)
which is automatically shipped with any PyTorch installation. 
Please ensure that PyTorch is installed on your system.

If PyTorch has been installed via [pip3](https://docs.python.org/3/installing/index.html) and it is not installed in the system defaults (that is the default case with pip!) you are required to tell cmake where to find PyTorch.
This can be done in step 3 of the quick build by specifying `Torch_DIR`, for example with
```
cmake -DTorch_DIR=$(pip3 show torch | grep Location | cut -d ' ' -f 2)/torch/share/cmake/Torch -DCMAKE_BUILD_TYPE=Debug ..
```

Here are some system specific details to install NSL

#### 1. Linux

##### 1.1 Arch Linux

Arch linux is rather flexible to obtain the newest versions of any software.
This allows to install the newest g++ compilers & PyTorch via the package manager.
For [g++](https://archlinux.org/packages/core/x86_64/gcc/) this would mean
```
sudo pacman -S gcc 
```
Once `g++ 11` is outdated and newer versions fail to build NSL (hopefully this never happens)
one can fall back to a `g++11` installation using the AUR. 
For more information visit the [official arch linux documentation](https://wiki.archlinux.org/title/GNU_Compiler_Collection).

[PyTorch](https://archlinux.org/packages/community/x86_64/python-pytorch/) can be installed in a similar way
```
sudo pacman -S python-pytorch
```
or with [GPU support](https://archlinux.org/packages/community/x86_64/python-pytorch-cuda/)
```
sudo pacman -S python-pytorch-cuda
```

For (nvidia) GPU support also ensure that the cuda-toolkit is installed. 
For more details on nvidia on arch linux visit the official [nvidia arch wiki](https://wiki.archlinux.org/title/NVIDIA)
as well as the [nvidia installation documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

If PyTorch is built with pip3, specify `Torch_DIR` as explained above.

##### 1.2 Ubuntu

We tested NSL on an Ubuntu 20.04 system. 
Older versions might fail or this documentation does not cover the required details.

First, the default version of cmake is not sufficient and must be updated following 
the [forum](https://askubuntu.com/a/1157132) the following steps do the trick

1. Ensure the defaul cmake is removed
```
sudo apt remove --purge --auto-remove cmake
sudo apt update
```
2. Get the signin key
```
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
```
3. Add the repository for cmake (20.04 specific command)
```
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
sudo apt update
```
4. Install cmake
```
sudo apt install cmake
```

To verify ensure that `cmake --version` returns a version larger or equal then `3.18`

The next step is to install `g++11`, following [this blog article](https://linuxize.com/post/how-to-install-gcc-on-ubuntu-20-04/)

1. Install the gnu compiler with major version 11
```
sudo apt install gcc-11 g++-11
```
2. Make those the default choice __*WARNING! This could break dependencies.*__
```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave  /usr/bin/g++ g++ /usr/bin/g++-11
```

Now you need to install PyTorch. 
Here it is recommended to use pip.

1. Install Python pip
```
sudo apt install python3-pip
```
2. Now install PyTorch:
```
sudo pip3 install torch torchvision torchaudio
```

This automatically detects weather GPU support is available and installs the correct 
pytorch version.

Now you can use the quick build to compile NSL, don't forget to specify `Torch_DIR` as explained above.

#### 2. macOS 

For a tested Homebrew setup see *Build on macOS with Homebrew (tested)* above.

Ensure that your clang version matches one of the above.

PyTorch can be installed using `pip3`
```
pip3 install torch torchvision torchaudio
```

If PyTorch is build with pip3, specify `Torch_DIR` as explained above.

#### 3. Windows

A build for windows is only given using the [Windows Subsystem for Linux](https://docs.microsoft.com/de-de/windows/wsl/install)
which can be installed through the powershell using 
```
wsl --install
```
This should install an Ubuntu 20.04 kernel, hence further prerequisites can be found at section __1.2 Ubuntu__.

## Dependencies

* [Pytorch](https://pytorch.org/)
* [Catch 2](https://github.com/catchorg/Catch2)

## Autotuner

The autotuner in the base directory **autotuner8** will run out of the box w/o any extra compilation.
However, it will use an in-house pythone routine for calculating the skew-normal distribution at the upper tail end of the distribution.
The reason for this is that scipy's implementation of the skew-normal distribution is buggy in this regime! See [here](https://github.com/scipy/scipy/issues/7746).

However, even the in-house python code is buggy, and this can in rare cases cause an issue with the autotuning procedure.  The best way
to avoid this issue is to make a python bind to the owens.f90 routine (included now in this repo), which seems to work amazingly. See [here](https://people.math.sc.edu/Burkardt/f_src/owens/owens.html)

To install this code, do the following:

* go to the directory NSL/autotuner8
* run > f2py -c -m owens owens.f90

That should make the binding!

## Stopping runs safely (SLURM walltime)

Every executable that calls `NSL::init` handles three signals (see `src/NSL/stop.hpp`):

* **SIGUSR1**: finish the current HMC trajectory (or the current propagator component), store it,
  and stop.
* **SIGTERM, SIGINT**: stop at once. If an HDF5 write is in progress, it is completed and flushed
  first, so everything stored before the stop is complete.

A run stopped this way exits with code **75** ("stopped early, resumable"); running the same command
again continues where it stopped:

* HMC chains continue from the last stored configuration.
* A plain-run thermalization stopped by SIGUSR1 is stored in `<name>/thermalization` and continued.
  After SIGTERM it starts again.
* `propagator` measures the components that are not yet stored.
* `autotuner8/autotuner.py` passes the signal on to the running call and exits with code 75.

In a SLURM batch script, ask for the warning some time before the walltime:

```sh
#SBATCH --signal=USR1@600
```

Write it without `B:`, so that the `srun` job step receives the signal. The margin (here 600 s) must exceed the time of one
trajectory (times `save frequency`) or of one measured component. `autotuner8/batch-scripts/submit_chain_batch.sh`
takes it from `STOP_MARGIN` in `env.sh`. Use it only with NSL executables, since SIGUSR1 terminates programs that do not handle it.

**Failed updates.** An HMC trajectory or radial update whose energy difference is not finite is rejected,
and its acceptance probability is stored as 0, never NaN. An update that throws an exception or yields NaN is rejected
and logged as a warning; 10 such failures in a row abort the run with an error message and exit code 1.

## Stability routines

NSL can now simulate at room temperature!!!  To perform such simulations, one needs to use particular stability methods.  The possible choices are

* DIRECTINVERSE
* SVD
* QR

**DIRECTINVERSE** has been the standard method up to now, and is really only good with $\beta$ up to 15 (max!). Both **QR** and **SVD** are good to $\beta=90$ and above (this assumes a nearest neighbor connectivity of 3). To specify which stability method to simulate with, add the following example (change to suit your desired stability method) into the input yaml file:
```
stability: QR
```
If no stability flag is given in the yaml file, then the default stability method will be **QR**. Note that the **QR** method is a factor of 7-8 slower than **DIRECTINVERSE**.  **SVD** is even slower. When in doubt, it is recommended to always use **QR**.
