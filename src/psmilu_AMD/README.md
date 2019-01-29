# [`amd.hpp`](./amd.hpp), Header-only version of AMD #

This directory contains the C++ implementation of original AMD code implemented in C by Dr. Tim Davis. All routines interfaces are preserved and wrapped as static member functions in a top-level class `AMD` with *template* parameter `Int`. Therefore, the C++ AMD, i.e. `amd.hpp`, is *header-only*. It's worth noting that `amd.hpp` requires **C++11**.

As a simple demonstration, the original routine `amd_order` and `amd_l_order` become `AMD<int>::order` and `AMD<long>::order`, resp (assuming 64-bit machines).

Also, this package is self-contained, you can simply embed `amd_config.hpp` and `amd.hpp`. However, there is an implicit namespace defined, which is `psmilu`, to use a different namespace, just pre-define macro `PSMILU_AMD_NAMESPACE`. Or you can do `PSMILU_AMD_USING_NAMESPACE` (equiv to `using namespace PSMILU_AMD_NAMESPACE`) in your code.

```cpp
// original
#include "amd.h"

int status = amd_order(...);

// c++ amd
#include "amd.hpp"
PSMILU_AMD_USING_NAMESPACE;
using amd = AMD<int>;
auto status = amd::order(...); // same parameter list
```

Navigate to the `tests` directory, which contains two demo codes that are adapted based on original `amd_demo.c` and `amd_demo2.c`. Simply type `make` to run them.

The original `amd.h` is distributed under BSD-3 license, as a requirement, the following copyright information is needed:

```
Copyright (C) 1996-2015, Timothy A. Davis, Patrick R. Amestoy, and Iain S. Duff.
```

In addition, the modification has the following copyright announcement:

```
Copyright (C) 2019 The PSMILU AUTHORS
```
