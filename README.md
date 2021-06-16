# Welcome to *HIFIR* Project #

## Introduction ##

Welcome to the `HIFIR` package! `HIF` stands for `H`ybrid (`H`ierarchical) `I`ncomplete `F`actorizations with `I`terative `R`efinements, which is a new *multilevel ILU-like* software framework based on (near) linear time complexity and robust dropping strategy. Yup, you read it right, we have both!

## Installation ##

The C++ interface `HIFIR` is a header-only package, one can simply just

```console
sudo cp -r src/* /usr/local/include
```

or regular user installation

```console
mkdir -p $HOME/.local/include
cp -r src/* $HOME/.local/include
```

## License ##

The `HIFIR` software package is released under a dual-license module. For academic users, individual users, or open-source software developers, you can use HIFIR under the GNU Affero General Public License version 3 (AGPLv3+, see [`LICENSE`](./LICENSE)) free of charge for research and evaluation purpose. For commerical users, separate commerical licenses are available through the Stony Brook University. For inqueries regarding commerical licenses, please contact Prof. Xiangmin Jiao at xiangmin.jiao@stonybrook.edu.
## Contacts ##

This packages is developed and being maintained by the *NumGeom* research group at Stony Brook University.

Active maintainer(s):

1. Qiao (Chiao) Chen, <qiao.chen@stonybrook.edu>, <benechiao@gmail.com>
2. Xiangmin (Jim) Jiao, <xiangmin.jiao@stonybrook.edu>

## Citations and Publications ##

If you plan to use HIFIR in solving for nonsingular systems, then please cite the following paper.

```bibtex
@article{chen2021hilucsi,
  author  = {Chen, Qiao and Ghai, Aditi and Jiao, Xiangmin},
  title   = {{HILUCSI}: Simple, robust, and fast multilevel {ILU} for
             large-scale saddle-point problems from {PDE}s},
  journal = {Numer. Linear Algebra Appl.},
  year    = {2021},
  note    = {To appear},
  doi     = {10.1002/nla.2400},
}
```

If you plan to use HIFIR in solving singular and ill-conditioned systems, then
please cite the following papers.

```bibtex
@article{jiao2020approximate,
  author  = {Xiangmin Jiao and Qiao Chen},
  journal = {arXiv},
  title   = {Approximate generalized inverses with iterative refinement for
             $\epsilon$-accurate preconditioning of singular systems},
  year    = {2020},
  note    = {arXiv:2009.01673},
}
```

```bibtex
@article{chen2021hifir,
  author  = {Chen, Qiao and Jiao, Xiangmin},
  title   = {{HIFIR}: Hybrid incomplete factorization with iterative refinement
             for preconditioning ill-conditioned and singular systems},
  journal = {arXiv},
  year    = {2021},
  note    = {arXiv:21...},
}
```
