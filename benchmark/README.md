# Benchmark Setup #

This directory contains the benchmark setup for `PSMILU` regarding reproducing
the results in the paper.

## Prerequisites ##

If you are interested how to reproduce the results, you need to request for the
our input cases. Please contact us for the data inputs.

Building the [driver](./driver.cpp) requires the following dependencies:

1. LAPACK/BLAS
2. HSL_MC64

For 1, virtually any vender versions will work. For 2, you need to request from
[HSL group](http://www.hsl.rl.ac.uk/catalogue/mc64.html). Once you get HSL_MC64,
follow the instruction and compile the package. The benchmark program needs the
double precision version **C interface**.

Next, we refer `MC64_ROOT` as the root directory of HSL_MC64; it should
follow standard \*NIX library structure, i.e. `MC64_ROOT/{include,lib}` contains
the header (`hsl_mc64d.h`) and the static library archive (`libhsl_mc64.a`),
resp.

`LAPACKBLAS_LIB` is used to link LAPACK and BLAS, e.g. `-lopenblas` or
`"-llapack -lblas"` (quotations are mandatory). PSMILU uses the Fortran
interfaces of LAPACK/BLAS, thus you don't need to worry about the header files.

Regarding Fortran name mangling, we assume this program is compiled on x86_64
Linux machines with GNU (or compatible) compilers. If this does not apply to
your case, then you need to modify the [makefile](./Makefile) to customize
the way of instantiating PSMILU package. (Please consult the documentation for
how to use the C++ interface.)

Finally, `PSMILU_INCLUDE` is the root of all PSMILU C++ header files.

## Compilation ##

TL;DR

```console
make \
    PSMILU_INCLUDE=/path/to/PSMILU.hpp \
    MC64_ROOT=/path/to/hsl_mc64 \
    LAPACKBLAS_LIB="-llapack -lblas"
```

## How to Run ##

TL;DR

```console
./driver -h
```

One additional note is for parameter tuning, i.e. `Controls` structure. The program
aggresively reads parameters from `std::cin`, thus you need to either pipe to *stdin*,
redirect *stdin* from file, or use your keyboards. The preferred way is redirecting;
a [template file](./parameters.cfg) is provided.
