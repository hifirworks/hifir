# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The PSMILU AUTHORS
# ----------------------------------------------------------------------------

# Authors:
#   Qiao,

# This is the Cython header for PyBuilder

from libcpp.memory cimport shared_ptr
cimport psmilu4py as psmilu


# Python class for PyBuilder
cdef class PyBuilder:
    cdef shared_ptr[psmilu.PyBuilder] builder