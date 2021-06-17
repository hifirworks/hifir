# HIFIR Unit Testing #

To add a new test, create a file with `test_*.cpp`. For instance, let's make a new unit testing `test_foo.cpp`.

```cpp
#include "common.hpp" // must include this before any other hif*.hpp

#include "hif/ds/Array.hpp"
#include "hif/utils/log.hpp"

#include <gtest/gtest.h>

TEST(...) {...}
```

Our unit testing framework can automatically extract all the dependencies. Now, simply just type

```console
make test_foo
```

## Hint ##

It's recommended that you run test case executable with `Valgrind`, thus you can check if or not there are memory leaks; e.g.

```console
make test_foo
valgrind ./test_foo.exe
```
