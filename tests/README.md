# PS-MILU Unit Testing #

To add a new test, create a file with `test_*.cpp`, and in the **first** line,
write down the header dependencies as normal comments, i.e. `//`.

For instance, let's make a new unit testing `test_foo.cpp`.

```cpp
// psmilu_Array.hpp psmilu_CompressedStorage.hpp

#include "common.hpp" // must include this before any other psmilu_*.hpp
// add a line to avoid "smart" header sorting
// include other psmilu_*.hpp

#include <gtest/gtest.h>

TEST(...) {...}
```

Now, simply just type

```console
make test_foo
```
