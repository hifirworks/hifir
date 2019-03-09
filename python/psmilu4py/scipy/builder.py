# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The PSMILU AUTHORS
# ----------------------------------------------------------------------------
"""PSMILU `PyBuilder` and Scipy ``LinearOperator`` compatible

.. module:: psmilu4py.scipy.builder
.. moduleauthor:: Qiao Chen, <qiao.chen@stonybrook.edu>
"""

from scipy.sparse.linalg.interface import _CustomLinearOperator
import numpy as np
from .utils import as_psmilu4py_data, as_sparse_matrix
from ..PyBuilder import PyBuilder

__all__ = ['ScipyBuilder']


class ScipyBuilder(PyBuilder, _CustomLinearOperator):
    r"""MILU preconditioner builder for `scipy`

    It's worth noting that the preconditioners can be passed in as a function
    parameter for all KSP solvers in `scipy.sparse.linalg`. The keyvalue
    parameter ``M`` denotes the approximated inverse of linear system A. `scipy`
    has already defined a class ``LinearOperator`` for user to override, and
    the key is to overload the behavior of its matrix-vector operation, i.e.
    ``LinearOperator.matvec``. Following the recommendation of `scipy` doc,
    instead of ``matvec``, we shall overload ``_matvec``.

    .. note::

        Once ``_matvec`` is implemented, ``matmat`` (matrix matrix operation)
        is automatically enabled.

    Notes
    -----
    `scipy` assumes a preconditioner **explicitly** approximates the inverse,
    while `psmilu4py` builder does this in an implicit fashion. In `scipy`,
    the preconditioner is :math:`\boldsymbol{M}^{-1}` thus requiring matrix
    vector multiplication. In `psmilu4py`, we compute ``M`` itself and solve
    for :math:`\boldsymbol{x}=\boldsymbol{M}^{-1}\boldsymbol{b}`.

    Parameters
    ----------
    shape : tuple
        must be length 2 tuple
    """
    def __init__(self, shape):
        PyBuilder.__init__(self)
        _CustomLinearOperator.__init__(self, shape, None, dtype=np.float64)
        self._buf = None

    def compute(self, A, m=0, opts=None, check=True):
        """Compute and build the MILU preconditioner

        Parameters
        ----------
        A : matrix-like
            user input matrix
        m : int
            leading symmetric block
        opts : :py:class:`psmilu4py.Options` (optional)
            control parameters, if ``None``, then use the default values
        check : bool (optional)
            if ``True`` (default), then perform checking for the input system
        """
        assert m >= 0, 'invalid leading block m{}'.format(m)
        # ensure A
        A, is_crs = as_sparse_matrix(A)
        assert A.shape == self.shape, 'mismatched shape'
        nrows, ncols = A.shape
        indptr, indices, vals = as_psmilu4py_data(A)
        PyBuilder.compute(
            self,
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
        # NOTE since matvec can be used as fullback to matmat, we need to make
        # sure that the x is contiguous, otherwise, we have a big problem if
        # the user wants to solver against multiple rhs
        b = np.ascontiguousarray(x, dtype=np.float64)
        if len(b.shape) == 2:
            b = b.reshape(-1)
        if self._buf is None or self._buf.size < b.size:
            self._buf = np.empty(b.size)
        y = self._buf[:b.size]
        PyBuilder.solve(self, b, y)
        return y
