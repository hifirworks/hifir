# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The PSMILU AUTHORS
# ----------------------------------------------------------------------------

# Authors:
#   Qiao,

# This is the Cython header for Options

from libcpp.memory cimport shared_ptr
cimport psmilu4py as psmilu


# Python class for Options
cdef class Options:
    cdef shared_ptr[psmilu.Options] opts