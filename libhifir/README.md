# The `libhifir` Library #

`libhifir` is a derived work of HIFIR. It is a runtime library with C and Fortran interfaces, and it is provided for convenience for linking with C/Fortran codes.

## Installation ##

After obtaining the C++ interface, one can build and install the C library `libhifir`. First, navigate to the subfolder libhifir

```console
cd libhifir
```

Then, compile and install the package using the following commands:

```console
make -j
make PREFIX=/path/to/destination install
```

The `-j` options allows building both 32-bit-integer
and 64-bit-integer verions at the same time. By default, `PREFIX=/usr/local`. In addition, set the `LAPACK_LIBS` variable if you have customized installation of LAPACK, such as 
```console
make LAPACK_LIBS="-L/path/to/openblas/lib -lopenblas" -j
```
which is also preferable to set `LDFLAGS=-Wl,-rpath,/path/to/openblas/lib`. Note that `LAPACK_LIBS="-llapack -lblas"` by default.
Note that HIFIR uses OpenMP for computing the Schur complement. You can
disable OpenMP by setting `USE_OPENMP=0`, i.e., `make USE_OPENMP=0 -j`.

Once the installation is complete, we will have

- `${PREFIX}/include/libhifir.h`: This is the header file.
- `${PREFIX}/lib/libhifir_i32.{so,a}`: These are shared and static libraries for 32-bit integer builds.
- `${PREFIX}/lib/libhifir_i64.{so,a}`: These are shared and static libraries for 64-bit integer builds.

To use 64-bit library, set `-DLIBHIFIR_INT_SIZE=64` while compiling your program, 
such as 
```console
cc -DLIBHIFIR_INT_SIZE=64 my_prog.c -lhifir_i64
```

## Examples ##

`libhifir/tests` contains some simple testing programs, which illustrate how to call multilevel triangular solve and matrix-vector multiplication in `libhifir` with both real and complex arithmetics. For comprehensive examples, it is recommended to check the `examples` directory.
