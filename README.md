# Welcome to *HIFIR* Project #

|CI | Documentation |
|:---: | :---:|
| [![HIFIR](https://github.com/hifirworks/hifir/actions/workflows/github-actions.yml/badge.svg?branch=main)](https://github.com/hifirworks/hifir/actions/workflows/github-actions.yml) | [![doc](https://img.shields.io/badge/%20doc-ready-blue)](https://hifirworks.github.io/hifir) |

## Introduction ##

Welcome to the `HIFIR` package! `HIFIR` stands for `H`ybrid `I`ncomplete `F`actorization with `I`terative `R`efinement, which is a *multilevel* preconditioner for ill-conditioned and (nearly) singular systems. `HIFIR` has (near) linear time complexity in its factorization and solve and is robust due to its multilevel structure, scalability-oriented dropping, and careful control of the condition numbers of the triangular factors.

Detailed documentation of HIFIR can be found at [https://hifirworks.github.io/hifir](https://hifirworks.github.io/hifir).

## Installation ##

To download the latest version of the code, use the command

```console
git clone -b release -–depth 1 https://github.com/hifirworks/hifir.git hifir
```

Use `git pull` to download any new changes that have been added since `git clone` or last `git pull`. Alternatively, use `git checkout v[GLOBAL].[MAJOR].[MINOR]` to download a specific version.

Another way is to download the current archive at [https://github.com/hifirworks/hifir/releases](https://github.com/hifirworks/hifir/releases).

The C++ interface of `HIFIR` is header-only. To install the code, you can simply do the following on a UNIX system

```console
sudo cp -r /path/to/hifir/src/* /usr/local/include
```

or copy `/path/to/hifir/src/*` to any user-level directory, for example,

```console
cp -r /path/to/hifir/src/* $HOME/.local/include
```

and then add `-I$HOME/.local/include` to the command line of your `C++` compiler.

For Windows users, the simplest and cleanest way is to use [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/) or [Cygwin](https://www.cygwin.com/).

Optionally, one can also use the C interface by building the runtime library `libhifir`; see the [README](./libhifir/README.md) therein for more details.

### Third-Party Dependency: LAPACK ###

The only third-party dependency of HIFIR is LAPACK, which HIFIR uses for its dense-level factorization and solve. We recommend using [OpenBLAS](https://www.openblas.net/), [ATLAS](http://math-atlas.sourceforge.net/), or [Intel oneAPI Math Kernel Library](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html) (formerly Intel MKL). Assuming you have system-administrator permussions, you can install OpenBLAS using the `apt` and `yum` command on Debian and RedHat-based Linux systems, such as

```console
sudo apt install libopenblas-dev
```

and

```console
sudo yum install libopenblas-devel
```

If you prefer to use ATLAS, the corresponding libraries are `libatlas-dev` and `libatlas-devel` on Debian and RedHat-based systems, respectively. Note that if you have multiple versions of BLAS and LAPACK installed, you may need to use the `update-alternatives --config` or `alternatives --config` command to config your system for the proper versions of BLAS and LAPACK; see <https://wiki.debian.org/DebianScience/LinearAlgebraLibraries> and <https://www.redhat.com/sysadmin/alternatives-command> for more detail.

For Intel MKL, please refer to <https://software.intel.com/content/www/us/en/develop/articles/installing-intel-free-libs-and-python-apt-repo.html> and <https://software.intel.com/content/www/us/en/develop/articles/installing-intel-free-libs-and-python-yum-repo.html> for installation on Debian and RedHat-based systems, respectively.

For other platforms, please refer to the official documentation of [OpenBLAS](https://www.openblas.net), [ATLAS](http://math-atlas.sourceforge.net/faq.html#doc), [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html), or the vendor-provided LAPACK for your platform, respectively.

## Copyright and Licenses ##

The `HIFIR` software suite (including `hifir4m` and `hifir4py`) are developed by the NumGeom Research Group at Stony Brook University.

The `HIFIR` software package is released under a dual-license model. For academic users, individual users, or open-source software developers, you can use HIFIR under the GNU Affero General Public License version 3 (AGPLv3+, see [`LICENSE`](./LICENSE)) free of charge for research and evaluation purpose. For commerical users, separate commerical licenses are available through the Stony Brook University. For inquiries regarding commercial licenses, please contact Prof. Xiangmin Jiao at xiangmin.jiao@stonybrook.edu.

## How to Cite `HIFIR` ##

If you use `HIFIR` in your research for nonsingular systems, please cite the `HILUCSI` paper:

```bibtex
@Article{chen2021hilucsi,
  author  = {Chen, Qiao and Ghai, Aditi and Jiao, Xiangmin},
  title   = {{HILUCSI}: Simple, robust, and fast multilevel {ILU} for
             large-scale saddle-point problems from {PDE}s},
  journal = {Numer. Linear Algebra Appl.},
  year    = {2021},
  number  = {6},
  pages   = {e2400},
  volume  = {28},
  doi     = {10.1002/nla.2400}
}
```

If you plan to use HIFIR in solving singular and ill-conditioned systems, please cite the following papers.

```bibtex
@Article{jiao2022approximate,
  author  = {Xiangmin Jiao and Qiao Chen},
  journal = {SIAM J. Matrix Anal. Appl.},
  title   = {Approximate generalized inverses with iterative refinement for
             $\epsilon$-accurate preconditioning of singular systems},
  year    = {2022},
  number  = {1},
  pages   = {40--67},
  volume  = {43},
  doi     = {10.1137/20M1364126}
}
```

```bibtex
@Article{chen2022hifir,
  author  = {Chen, Qiao and Jiao, Xiangmin},
  title   = {{HIFIR}: Hybrid incomplete factorization with iterative refinement
             for preconditioning ill-conditioned and singular systems},
  journal = {ACM Trans. Math. Softw.},
  year    = {2022},
  doi     = {10.1145/3536165}
}
```

## Contacts ##

- Qiao Chen, <qiao.chen@stonybrook.edu>, <benechiao@gmail.com>
- Xiangmin Jiao, <xiangmin.jiao@stonybrook.edu>, <xmjiao@gmail.com>
