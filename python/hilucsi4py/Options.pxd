# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The PSMILU AUTHORS
# ----------------------------------------------------------------------------

# Authors:
#   Qiao,

# This is the Cython header for Options

cimport psmilu4py as psmilu


# Python class for Options
cdef class Options:
    cdef psmilu.Options opts