# -*- coding: utf-8 -*-
###############################################################################
#                 This file is part of HILUCSI4PY project                     #
###############################################################################

# implementations of print wrappers

# Authors:
#   Qiao,

import sys

cdef void hilucsi4py_stdout(const char *msg):
    print(msg.decode('utf-8'), file=sys.stdout)

cdef void hilucsi4py_stderr(const char *msg):
    print(msg.decode('utf-8'), file=sys.stderr)
