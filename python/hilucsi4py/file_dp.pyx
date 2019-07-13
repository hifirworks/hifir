# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The PSMILU AUTHORS
# ----------------------------------------------------------------------------

# implementations of print wrappers

# Authors:
#   Qiao,

import sys

cdef void psmilu4py_stdout(const char *msg):
    print(msg.decode('utf-8'), file=sys.stdout)

cdef void psmilu4py_stderr(const char *msg):
    print(msg.decode('utf-8'), file=sys.stderr)
