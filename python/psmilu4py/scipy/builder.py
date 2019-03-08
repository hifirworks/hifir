# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The PSMILU AUTHORS
# ----------------------------------------------------------------------------
#
# Authors:
#   Qiao,
#
"""PSMILU `PyBuilder` and Scipy ``LinearOperator`` compatible

.. module:: psmilu4py.scipy.builder
.. moduleauthor:: Qiao Chen, <qiao.chen@stonybrook.edu>
"""

from scipy.sparse.linalg import LinearOperator
import numpy as np
from .utils import as_psmilu4py_data
from ..PyBuilder import PyBuilder

__all__ = ['ScipyBuilder']


class ScipyBuilder(PyBuilder, LinearOperator):
    def __init__(self, shape):
        PyBuilder.__init__()
        LinearOperator.__init__(shape, None, dtype=np.float64)
        self._buf = None

    def compute(self, A, m=0, opts=None, check=True):
        from scipy.sparse import csc_matrix
        assert A.shape == self.shape, 'mismatched shape'
        assert m >= 0, 'invalid leading block m{}'.format(m)
        nrows, ncols = A.shape
        # ensure data type consistency
        is_crs = not issubclass(A, csc_matrix)
        if is_crs:
            from scipy.sparse import csr_matrix
            if not issubclass(A, csr_matrix):
                A = A.tocsr()  # convert to CRS
        indptr, indices, vals = as_psmilu4py_data(A)
        PyBuilder.compute(
            nrows,
            ncols,
            indptr,
            indices,
            vals,
            m,
            opts=opts,
            check=check,
            is_crs=is_crs)

    def _matvec(self, x):
        # ensure b is contiguous array
        # NOTE that while b passing in, it has already been checked for the
        # shape and size
        b = np.ascontiguousarray(x, dtype=np.float64)
        if len(b.shape) == 2:
            b = b.reshape(-1)
        if self._buf is None or self._buf.size < b.size:
            self._buf = np.empty(b.size)
        y = self._buf[:b.size]
        PyBuilder.solve(b, y)
        return y
