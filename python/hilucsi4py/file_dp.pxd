# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The HILUCSI AUTHORS
# ----------------------------------------------------------------------------

# This file is to utilize Cython API feature to generate Python3 print
# wrappers for stdout (1) and stderr (2), which will be wrapped as
# HILUCSI_{STDOUT,STDERR}, resp.

# Authors:
#   Qiao,


cdef api void hilucsi4py_stdout(const char *msg_wo_nl)
cdef api void hilucsi4py_stderr(const char *msg_wo_nl)