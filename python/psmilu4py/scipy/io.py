# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The PSMILU AUTHORS
# ----------------------------------------------------------------------------
#
# Authors:
#   Qiao,
#
from .utils import as_psmilu4py_data
from .. import io as _io

__all__ = [
    'read_native_psmilu',
    'write_native_psmilu',
]


def read_native_psmilu(filename, is_crs=True, is_bin=True):
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
    from scipy.sparse import csc_matrix
    assert m >= 0, 'invalid leading block m{}'.format(m)
    is_crs = not issubclass(A, csc_matrix)
    if is_crs:
        from scipy.sparse import csr_matrix
        if not issubclass(A, csr_matrix):
            A = A.tocsr()
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
