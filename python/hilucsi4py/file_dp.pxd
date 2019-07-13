# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The PSMILU AUTHORS
# ----------------------------------------------------------------------------

# This file is to utilize Cython API feature to generate Python3 print
# wrappers for stdout (1) and stderr (2), which will be wrapped as
# PSMILU_{STDOUT,STDERR}, resp.

# Authors:
#   Qiao,


cdef api void psmilu4py_stdout(const char *msg_wo_nl)
cdef api void psmilu4py_stderr(const char *msg_wo_nl)