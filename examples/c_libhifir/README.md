# Examples of `libhifir` #

This directory contains some simple examples of using the C library `libhifir`. For comprehensive examples, we refer the readers to the C++ examples.

## Compilation ##

Type `make` to compile all examples into executables. If your installation of `libhifir` is not standard, then pass in the path variable `PREFIX` during compilation, i.e.,

```console
make PREFIX=/path/to/libhifir
```

Note that if `PREFIX` is specified, then we assume  that `libhifir.h` and `libhifir.so` locate at `PREFIX/include` and `PREFIX/lib`, respectively.

## Running Examples ##

Simply execute the compiled programs to run different examples.
