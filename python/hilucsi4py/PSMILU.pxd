# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The PSMILU AUTHORS
# ----------------------------------------------------------------------------

# Authors:
#   Qiao,

# This is the Cython header for PSMILU

from libcpp.memory cimport shared_ptr
cimport psmilu4py as psmilu


# Python class for PyPSMILU
cdef class PSMILU:
    cdef shared_ptr[psmilu.PyPSMILU] prec