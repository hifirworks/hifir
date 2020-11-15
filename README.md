# Welcome to *HIF* Project #

## Introduction ##

Welcome to the `HIF` package! `HIF` stands for `H`ybrid (`H`ierarchical) `I`ncomplete `F`actorizations, which is a new *multilevel ILU-like* software framework based on (near) linear time complexity and robust dropping strategy. Yup, you read it right, we have both!

## Installation ##

Since `HIF` is a header-only package, one can simply just

```console
sudo cp src/* /usr/local/include
```

or regular user installation

```console
mkdir -p $HOME/.local/include
cp src/* $HOME/.local/include
```

One can also install `HIF` via `cmake`

```console
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local ..
make install
```

One can reset the default options in `macros.hpp` via `cmake`, for more, please take a look at all options with command

```console
cmake -LA | awk '{if(f)print} /-- Cache values/{f=1}'
```

## License ##

`HIF` is released under GPL version 3. For more details, please refer to the [`LICENSE`](./LICENSE) file.

## Contact ##

This packages is developed and being maintained by the *NumGeom* research group at Stony Brook University.

Active maintainer(s):

1. Qiao (Chiao) Chen, qiao.chen@stonybrook.edu
