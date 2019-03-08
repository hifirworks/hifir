# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The PSMILU AUTHORS
# ----------------------------------------------------------------------------
#
# Authors:
#   Qiao,
#
"""Utility routines for `scipy` interface

.. module:: psmilu4py.scipy.utils
.. moduleauthor:: Qiao Chen, <qiao.chen@stonybrook.edu>
"""

import numpy as np


def as_psmilu4py_data(sp_mat):
    return np.ascontiguousarray(
        sp_mat.indptr, dtype=np.intc), np.ascontiguousarray(
            sp_mat.indices, dtype=np.intc), np.ascontiguousarray(
                sp_mat.data, dtype=np.float64)
