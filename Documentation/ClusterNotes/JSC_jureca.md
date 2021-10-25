# Jureca

The code base is tested on the super computer [Jureca](https://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JURECA/JURECA_node.html) 
using `stages/2020` .
The following should help to compile and utilize NSL on Jureca.

## Installation

Jurecas environment already comes with all required compilers and versions we need to compile the code.
We assume that you start with the standard environment `StdEnv/2020` this means you should get the following output
```
module list
    Currently Loaded Modules:
        1) GCCcore/.10.3.0 (H)   2) zlib/.1.2.11 (H)   3) binutils/.2.36.1 (H)   4) StdEnv/2020

    Where:
        H:  Hidden Module
```
For compilation you need to load two modules (`module load` can be abbreviated with `ml`)
```
module load CMake
module load PyTorch
```
After the loaded modules should look like
```
module list

    Currently Loaded Modules:
        1) GCCcore/.10.3.0        (H)    16) libpng/.1.6.37      (H)  31) Java/15.0.1                        46) LAME/.3.100                          (H)
        2) zlib/.1.2.11           (H)    17) freetype/.2.10.1    (H)  32) PostgreSQL/12.3                    47) x265/.3.4                            (H)
        3) binutils/.2.36.1       (H)    18) gperf/.3.1          (H)  33) gflags/.2.2.2                 (H)  48) libvpx/1.9.0
        4) StdEnv/2020                   19) util-linux/.2.36    (H)  34) libspatialindex/.1.9.3        (H)  49) FriBidi/1.0.9
        5) ncurses/.6.2           (H)    20) fontconfig/.2.13.92 (H)  35) NASM/.2.15.03                 (H)  50) FFmpeg/.4.3.1                        (H)
        6) CMake/3.18.0                  21) xorg-macros/.1.19.2 (H)  36) libjpeg-turbo/.2.0.5          (H)  51) LibTIFF/.4.1.0                       (H)
        7) imkl/.2021.2.0         (H)    22) libpciaccess/.0.16  (H)  37) Python/3.8.5                       52) Pillow-SIMD/7.0.0.post3-Python-3.8.5
        8) Ninja/1.10.0                  23) X11/20200222             38) protobuf/.3.17.3              (H)  53) magma/2.5.4                          (g)
        9) nvidia-driver/.default (H,g)  24) Tk/.8.6.10          (H)  39) protobuf-python/.3.17.3       (H)  54) NCCL/2.10.3-1-CUDA-11.3              (g)
       10) CUDA/11.3              (g)    25) GMP/6.2.0                40) pybind11/.2.5.0-Python-3.8.5  (H)  55) cuDNN/8.2.1.32-CUDA-11.3             (g)
       11) bzip2/.1.0.8           (H)    26) XZ/.5.2.5           (H)  41) SciPy-Stack/2021-Python-3.8.5      56) LLVM/10.0.1
       12) libreadline/.8.0       (H)    27) libxml2/.2.9.10     (H)  42) typing-extensions/3.7.4            57) PyTorch/1.8.1-Python-3.8.5           (g)
       13) Tcl/8.6.10                    28) libxslt/.1.1.34     (H)  43) MPFR/4.1.0
       14) SQLite/.3.32.3         (H)    29) libffi/.3.3         (H)  44) numactl/2.0.14
       15) expat/.2.2.9           (H)    30) libyaml/.0.2.5      (H)  45) x264/.20200912                (H)

    Where:
        g:  built for GPU
        H:             Hidden Module
```
and the versions of cmake and g++ should be at least 
```
cmake --version
    cmake version 3.18.0

g++ --version
    g++ (GCC) 10.3.0
    Copyright (C) 2020 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

Compilation is then similar to the quick build instructions
1. Obtain a copy of the library, this does not work with ssh due to the cluster policy.
Make sure that you have a [GitHub token](https://github.com/settings/tokens) 
```
git clone https://github.com/Marcel-Rodekamp/NSL.git && cd NSL
```
2. Create a build directory and change into it
```
mkdir build && cd build
```
3. Call cmake to generate the build files
    * Possible build types:
        * `Debug`: Build for developing and debugging, turn on runtime assertions
        * `Release`: Build for production, turn off runtime assertions
``` 
cmake CMAKE_BUILD_TYPE=Debug ..
```
4. Build the directory (add `-j 4` for parallel build on 4 cores)
```
make
```

## Example sbatch

A typical testing suit sbatch file (located in `/pathToNSL/YourBuildDirectory/Tests`) could look like this:
```
#!/bin/bash

#Job parameters
#SBATCH --job-name=test_NSL
#SBATCH --output=./%x_%j.out
#SBATCH --error=./%x_%j.err
#SBATCH --mail-user=userName@provider.com --mail-type=FAIL

#Resources
#SBATCH --account=YourComputeTimeAccount
#SBATCH --time=00:10:00 # ToDo we eventually might need to adjust this time later!
#SBATCH --partition=dc-cpu-devel # dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1

#Print some information
echo -e "Start `date +"%F %T"` | $SLURM_JOB_ID $SLURM_JOB_NAME | `hostname` | `pwd` \n" 

# load required models
ml PyTorch

#Job steps
echo -e "## Calling Tensor/test_tensor:\n"
srun ./Tensor/test_tensor
echo -e "## Calling LinAlg/test_mat_exp:\n"
srun ./LinAlg/test_mat_exp
echo -e "## Calling Lattice/test_bipartite:\n"
srun ./Lattice/test_bipartite  
echo -e "## Calling Lattice/test_complete:\n"
srun ./Lattice/test_complete  
echo -e "## Calling Lattice/test_ring:\n"
srun ./Lattice/test_ring  
echo -e "## Calling Lattice/test_square:\n"
srun ./Lattice/test_square

echo -e "End `date +"%F %T"` | $SLURM_JOB_ID $SLURM_JOB_NAME | `hostname` | `pwd` \n
```
