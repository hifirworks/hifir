# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The PSMILU AUTHORS
# ----------------------------------------------------------------------------

# Authors:
#   Qiao,

# This is the implementation for PyBuilder

"""Core MILU preconditioner builder for Python

This is the core module, which builds the multilevel ILU preconditiioner that
can be used in KSP (not limited to) solvers.

.. module:: psmilu4py.PyBuilder
.. moduleauthor:: Qiao Chen, <qiao.chen@stonybrook.edu>
"""

from libcpp cimport bool
from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
cimport psmilu4py as psmilu
from .Options cimport Options


cdef class PyBuilder:
    """Python MILU preconditioner builder

    The interfaces remain the same as the original builder,i.e.
    `psmilu::C_DefaultBuilder`. However, we significantly the parameters by
    hiding the needs of `psmilu::CRS`, `psmilu::CCS`, and `psmilu::Array`.
    Therefore, the interface is very generic and easily adapted to any other
    Python modules without the hassle of complete object types.

    .. note::

        In addition, the only supported data types are ``int`` and ``double``;
        Only C index system (0-based) is supported.

    Examples
    --------

    >>> from psmilu4py import *
    >>> from psmilu4py.PyBuilder import *
    >>> from psmilu4py.io import read_native_psmilu
    >>> import numpy as np
    >>> n1, n2, rowptr, colind, vals, m = read_native_psmilu('mytest.psmilu')
    >>> builder = PyBuilder()
    >>> # compute the preconditioner
    >>> builder.compute(n1, n2, rowptr, colind, vals, m)
    >>> b = np.random.rand(n1)
    >>> x = np.empty_like(b)
    >>> builder.solve(b, x) # solve x=inv(M)*b
    """
    def __init__(self):
        # for docstring purpose
        pass

    def __cinit__(self):
        self.builder.reset(new psmilu.PyBuilder())

    def compute(self, size_t nrows, size_t ncols,
        int[::1] indptr not None,
        int[::1] indices not None,
        double[::1] vals not None,
        int m0, *,  # the rest pars must be kws
        Options opts=None,
        check=True, is_crs=True):
        """Compute/build the preconditioner

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
        opts : :py:class:`psmilu4py.Options` (optional)
            control parameters, if ``None``, then use the default values
        check : bool (optional)
            if ``True`` (default), then perform checking for the input system
        is_crs : bool (optional)
            if ``True``, assume the input is compressed row storage (CRS)

        Attributes
        ----------
        levels : int
            number of levels
        nnz : int
            total number of nonzeros

        See Also
        --------
        :func:`solve`: solve for inv(PyBuilder)*x
        """
        # essential checkings to avoid segfault
        if is_crs:
            assert len(indptr) == nrows+1
        else:
            assert len(indptr) == ncols+1
        assert len(vals) == len(indices)
        assert len(vals) == indptr[len(indptr)-1]-indptr[0]
        cdef:
            size_t m = m0
            Options my_opts = Options()
            bool ck = check
            bool is_row = is_crs
        if opts is not None:
            my_opts.opts = my_opts.opts
        deref(self.builder).compute(nrows, ncols, &indptr[0], &indices[0],
            &vals[0], m, my_opts.opts, ck, is_row)

    def solve(self, double[::1] b not None, double[::1] x not None):
        r"""Core routine to use the preconditioner

        Essentailly, this routine is to perform

        .. math::

            \boldsymbol{x}&=\boldsymbol{M}^{-1}\boldsymbol{b}
        
        Where :math:`\boldsymbool{M}` is our MILU preconditioner.

        Parameters
        ----------
        b : array-like
            right-hand side parameters
        x : array-like (output)
            solution vector
        """
        cdef size_t n = len(b)
        assert n == len(x)
        deref(self.builder).solve(n, &b[0], &x[0])

    def empty(self):
        """Check if or not the builder is empty"""
        return deref(self.builder).empty()

    @property
    def levels(self):
        """int: number of levels"""
        return deref(self.builder).levels()

    @property
    def nnz(self):
        """int: total number of nonzeros of all levels"""
        return deref(self.builder).nnz()
