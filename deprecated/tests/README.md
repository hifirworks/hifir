# PS-MILU Unit Testing #

To add a new test, create a file with `test_*.cpp`; the dependencies are
specified as `include`s.

For instance, let's make a new unit testing `test_foo.cpp`.

```cpp
#include "common.hpp" // must include this before any other psmilu_*.hpp

#include "psmilu_Array.hpp"
#include "psmilu_log.hpp"

#include <gtest/gtest.h>

TEST(...) {...}
```

Our unit testing framework can automatically extract the dependencies, i.e.
`psmilu_Array.hpp` and `psmilu_log.hpp` in this case, on-the-fly. In addition,
both quotation marks and angle brackets work as well with or without space(s)
between `include` and the files.

Now, simply just type

```console
make test_foo
```

**Something won't work!** Comment right after filename will break! The example
below will cause problem!

```cpp
#include "psmilu_Array.hpp"// blah blah
```

## Hint ##

It's recommended that you run test case executable with `Valgrind`, thus you
can check if or not there are memory leaks; e.g.

```console
make test_foo
valgrind ./test_foo.exe
```

## Test Scripts with LAPACK ##

Some tests require you to link against `LAPACK`, e.g. `test_sss_lup.cpp` and
`test_sss_qrcp.cpp`. You need, then, invoke `make` with `LAPACK_LIB` to the
library (assuming `LAPACK` stays in some standard locations that can be found
by linker). For instance, to run LUP test, invoke

```console
make LAPACK_LIB="-llapack -lblas" test_sss_lup
```

If you want to test all scripts, then you **must** add `LAPACK` library, i.e.

```console
make LAPACK_LIB="-llapack -lblas"
```
