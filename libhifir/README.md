# The `libhifir` Library #

`libhifir` is a derived work of HIFIR. It is a runtime library with C and Fortran interfaces.

After obtaining the C++ interface, one can (optionally) install the C library `libhifir`. First, navigate to the subfolder libhifir

```console
cd libhifir
```

Then, install the package using the following procedure

```console
make -j
make PREFIX=/path/to/destination install
```

`-j` can benefit the build process, as it allows building both 32-bit integer
and 64-bit integer verions as the same time. By default, `PREFIX=/usr/local`. In addition, set USE_OPENMP=0 to disable OpenMP, i.e., `make USE_OPENMP=0 -j`, and modify `LAPACK_LIBS` if you have customized installation of LAPACK, i.e., `make LAPACK_LIBS="-L/path/to/openblas/lib -lopenblas" -j`; it is also preferable to set `LDFLAGS=-Wl,-rpath,/path/to/openblas/lib` before compilation. Note that `LAPACK_LIBS="-llapack -lblas"` by default.

Once the installation is finished, we have

- `${PREFIX}/include/libhifir.h`: This is the header file.
- `${PREFIX}/lib/libhifir_i32.{so,a}`: These are shared and static libraries for 32-bit integer builds.
- `${PREFIX}/lib/libhifir_i64.{so,a}`: These are shared and static libraries for 64-bit integer builds.

In order to use 64bit library, set `-DLIBHIFIR_INT_SIZE=64` while compiling your program, i.e., `cc -DLIBHIFIR_INT_SIZE=64 my_prog.c -lhifir_i64`.
