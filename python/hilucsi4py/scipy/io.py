# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The PSMILU AUTHORS
# ----------------------------------------------------------------------------
"""I/O with native PSMILU formats and `scipy` sparse matrices

.. module:: psmilu4py.scipy.io
.. moduleauthor:: Qiao Chen, <qiao.chen@stonybrook.edu>
"""

from .utils import as_psmilu4py_data, as_sparse_matrix
from .. import io as _io

__all__ = [
    'read_native_psmilu',
    'write_native_psmilu',
]


def read_native_psmilu(filename, is_crs=True, is_bin=True):
    """Read native PSMILU file

    .. note:: This routine can return either ``csc_matrix`` or ``csr_matrix``.

    Parameters
    ----------
    filename : str
        file name of native psmilu file
    is_crs : bool (optional)
        if ``True`` (default), then a ``csr_matrix`` is generated
    is_bin : bool (optional)
        if ``True`` (default), then assume `filename` is binary file

    Returns
    -------
    tuple of two arguments, the first one is `scipy` sparse matrix of
    ``csr_matrix`` if `is_crs` else ``csc_matrix``, and the second one is the
    leading symmetric block size.

    Examples
    --------

    >>> from psmilu4py.scipy import *
    >>> my_file = './A.psmilu'
    >>> A1, m1 = read_native_psmilu(my_file)
    >>> A2, m2 = read_native_psmilu(my_file, is_crs=False)
    >>> assert A1.nnz == A2.nnz
    >>> assert m1 == m2
    """
    if is_crs:
        from scipy.sparse import csr_matrix
        matrix_type = csr_matrix
    else:
        from scipy.sparse import csc_matrix
        matrix_type = csc_matrix
    info = _io.read_native_psmilu(filename, is_crs=is_crs, is_bin=is_bin)
    return matrix_type((info[5], info[4], info[3]), shape=(info[0],
                                                           info[1])), info[2]


def write_native_psmilu(filename, A, m=0, is_bin=True):
    """Write a `scipy` sparse matrix (or dense matrix) to native PSMILU formats

    .. note::

        Unless `A` is strictly ``csc_matrix`` (or its child), we will store `A`
        in CSR format.

    Parameters
    ----------
    filename : str
        file name
    A : matrix
        can be either sparse or dense matrices
    m : int (optional)
        leading symmetric block, default is zero, i.e. asymmetric system
    is_bin : bool (optional)
        if ``True`` (default), then store as a binary file

    Examples
    --------

    >>> import numpy as np
    >>> A = np.random.rand(5,5)
    >>> from scipy.sparse import csc_matrix
    >>> B = csc_matrix(A)
    >>> from psmilu4py.scipy import *
    >>> write_native_psmilu('dense.psmilu', A)
    >>> write_native_psmilu('sparse.psmilu', B) 
    """
    assert m >= 0, 'invalid leading block m{}'.format(m)
    A, is_crs = as_sparse_matrix(A)
    nrows, ncols = A.shape
    indptr, indices, vals = as_psmilu4py_data(A)
    _io.write_native_psmilu(
        filename,
        nrows,
        ncols,
        indptr,
        indices,
        vals,
        m,
        is_crs=is_crs,
        is_bin=is_bin)
