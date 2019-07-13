# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The HILUCSI AUTHORS
# ----------------------------------------------------------------------------
"""Utility module for ``hilucsi4py``

This module contains some helpful utility routines for ``hilucsi4py``, the core
functions are

1. parsing user arbitrary inputs, and
2. ensuring the data types

Notice that we use ``int`` as index type and ``double`` as floating number type,
thus the corresponding numpy dtypes are ``intc`` and ``float64``.

.. module:: hilucsi4py.utils
.. moduleauthor:: Qiao Chen, <qiao.chen@stonybrook.edu>
"""

import numpy as np


def _convert_to_crs(*args, shape=None):
    """Given user inputs, convert them to proper CRS three arrays with size
    """
    if len(args) != 3 and len(args) != 1:
        raise TypeError(
            'input matrix must be CRS\'s three arrays or scipy sparse matrix')
    is_3_arr = len(args) == 3
    if is_3_arr and shape is None:
        raise ValueError('shape is missing')
    if is_3_arr:
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise TypeError('shape must be tuple of dimension 2')
        if shape[0] != shape[1]:
            raise ValueError('the matrix must be squared')
        return args[0], args[1], args[2], shape[0]
    if not hasattr(args[0], 'tocsr'):
        raise TypeError('single-argument input must be scipy sparse matrix')
    A = args[0].tocsr()
    if shape is not None and shape != A.shape:
        raise ValueError(
            'inconsistent user-shape {} and matrix shape {}'.format(
                shape, A.shape))
    shape = A.shape
    if shape[0] != shape[1]:
        raise ValueError('the matrix must be squared')
    return A.indptr, A.indices, A.data, shape[0]


def _as_index_array(v):
    return np.asarray(v, dtype=np.intc)


def _as_value_array(v):
    return np.asarray(v, dtype=np.float64)


def convert_to_crs(*args, shape=None):
    rowptr, colind, vals, n = _convert_to_crs(*args, shape=shape)
    rowptr = _as_index_array(rowptr)
    assert len(rowptr.shape) == 1
    assert rowptr.size == n + 1, 'invalid rowptr size'
    colind = _as_index_array(colind)
    assert len(colind.shape) == 1
    vals = _as_value_array(vals)
    assert len(vals.shape) == 1
    assert colind.size == vals.size, 'colind and vals should have same size'
    assert rowptr[n] == colind.size, 'inconsistent rowptr[n] and colind.size'
    return rowptr, colind, vals


def convert_to_crs_and_b(*args, shape=None):
    last = len(args) - 1
    if not last:
        raise ValueError('invalid inputs')
    rowptr, colind, vals = convert_to_crs(*args[:last], shape=shape)
    b = _as_value_array(args[last])
    if b.size != len(rowptr) - 1:
        raise ValueError(
            'rhs must have size of n ({})'.format(len(rowptr) - 1))
    return rowptr, colind, vals, b.reshape(-1)
