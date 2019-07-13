# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The PSMILU AUTHORS
# ----------------------------------------------------------------------------

# Authors:
#   Qiao,

"""I/O routines to read/write with native PSMILU file formats

.. module:: psmilu4py.io
.. moduleauthor:: Qiao Chen, <qiao.chen@stonybrook.edu>
"""

from libcpp cimport bool
from libcpp.string cimport string as std_string
from libc.stddef cimport size_t
from libcpp.vector cimport vector
cimport psmilu4py as psmilu

import numpy as np


def read_native_psmilu(str filename, *, is_crs=True, is_bin=True):
    """Read a PSMILU native file

    Parameters
    ----------
    filename : str
        file name
    is_crs : bool (optional)
        if ``True`` (default), then load the data as CRS matrix
    is_bin : bool (optional)
        if ``True`` (default), then assume binary file format

    Returns
    -------
    `tuple` of `nrows`, `ncols`, `m`, `indptr`, `indices`, `vals`

    Notes
    -----
    `is_crs` doesn't need to match with the matrix type defined in file
    `filename`. Just choose the storage you prefer without worring about the
    converting files.

    See Also
    --------
    :func:`write_native_psmilu` : write native formats
    """
    cdef:
        std_string fn = filename.encode('utf-8')
        bool is_row = is_crs
        bool isbin = is_bin
        vector[int] indptr
        vector[int] indices
        vector[double] vals
        size_t nrows = 0
        size_t ncols = 0
        size_t m = 0
    psmilu.read_native_psmilu(fn, nrows, ncols, m, indptr, indices, vals,
        is_crs, isbin)
    return nrows, ncols, m, \
        np.asarray(indptr, dtype=np.intc), \
        np.asarray(indices, dtype=np.intc), \
        np.asarray(vals, dtype=np.float64)


def write_native_psmilu(str filename, size_t nrows, size_t ncols,
    int[::1] indptr not None,
    int[::1] indices not None,
    double[::1] vals not None,
    int m0, *,
    is_crs=True, is_bin=True):
    """Write data to PSMILU file formats

    Parameters
    ----------
    nrows : int
        number of rows
    ncols : int
        number of columns
    indptr : array-like
        index pointer/start array
    indices : array-like
        index array
    vals : array-like (double precision)
        numerical values
    m0 : int
        leading symmetric block
    is_crs : bool (optional)
        if ``True`` (default), then load the data as CRS matrix
    is_bin : bool (optional)
        if ``True`` (default), then assume binary file format

    See Also
    --------
    :func:`read_native_psmilu` : read native formats
    """
    # essential checkings to avoid segfault
    if is_crs:
        assert len(indptr) == nrows+1
    else:
        assert len(indptr) == ncols+1
    assert len(vals) == len(indices)
    assert len(vals) == indptr[len(indptr)-1]-indptr[0]
    cdef:
        std_string fn = filename.encode('utf-8')
        bool is_row = is_crs
        bool isbin = is_bin
    psmilu.write_native_psmilu(fn, nrows, ncols, &indptr[0], &indices[0],
            &vals[0], m0, is_row, isbin)
