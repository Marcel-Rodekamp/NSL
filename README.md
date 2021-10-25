# Nanosystem Simulation Library

Nanosystem Simulation Library (NSL) implements statistical simulations for systems on the nanoscale.

This library provides a merge of the two software packages CNS and [isle](https://github.com/evanberkowitz/isle).
Additional features become added to provide a self-containing simulation library for nanostructures such as

<!-- Add all the systems we have implemented here -->
* Graphene
* Carbon Nanotubes

## Installation

The defaul branch is the `devel` branch. 
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

### Prerequisites

Sometimes a few things are needed before the build can work.
We utilize features from C++20 which requires more recent compilers. 
Currently, NSL is tested to work with 

1. `g++ (GCC) 10.3.0`
2. `g++ (GCC) 11.1.0`
3. `clang++ 13.0.0` with targets:
   * `x86_64-apple-darwin20.6.0`
   * `arm64-apple-darwin20.6.0`

Additionally, the backend depends on [PyTorches C++ ABI `libtorch`](https://pytorch.org/cppdocs/installing.html)
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
This allows to install the newest g++ compilers & PyTorch via the packet manager.
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

If PyTorch is build with pip3, specify `Torch_DIR` as explained above.

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
