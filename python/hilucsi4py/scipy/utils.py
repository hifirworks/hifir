# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The PSMILU AUTHORS
# ----------------------------------------------------------------------------
"""Utility routines for `scipy` interface

This module defines two important helper functions that helps transferring
user inputs to proper arrays, which can be accepted by the lower level
builder, i.e. :py:class:`psmilu4py.PyBuilder.PyBuilder`.

Essentially, we assume that the user can pass in arbitrarily object types, thus
we need components to convert these objects to proper numpy arrays. Recall that
the underlying `psmilu4py` builder only accepts numpy arrays with types
``numpy.intc`` and ``numpy.float64``. The converting process must be transparent
and smart; by smart, we want to avoid unnecessary data copying. Luckily, this
can fairly easy to achieve with numpy.

.. module:: psmilu4py.scipy.utils
.. moduleauthor:: Qiao Chen, <qiao.chen@stonybrook.edu>
"""

import numpy as np


def as_sparse_matrix(A):
    """First step convertion, convert any user input operator to sparse matrices

    This routine converts the user input to `scipy` sparse matrix. A is directly
    returned if its type is either ``csc_matrix`` (or its children) or
    ``csr_matrix`` (again, including its children). Otherwise, we will first
    convert the user input to proper numpy array and further convert to
    ``csr_matrix``.

    Parameters
    ----------
    A : matrix-like
        user input of matrix object

    Returns
    -------
    tuple of sparse matrix and boolean flag indicating the matrix storage type
    """
    from scipy.sparse import isspmatrix, csr_matrix
    if isspmatrix(A):
        from scipy.sparse import csc_matrix
        # first check if CCS
        if issubclass(A.__class__, csc_matrix):
            return A, False
        if issubclass(A.__class__, csr_matrix):
            return A, True
        return A.tocsr(), True
    return csr_matrix(np.asarray(A)), True


def as_psmilu4py_data(sp_mat):
    """Second step convertion, convert the three arrays to proper data types

    .. warning::

        This routine implicitly assumes the input must be compatible with either
        ``csc_matrix`` or ``csr_matrix``.

    Parameters
    ----------
    sp_mat : sparse matrix
        sparse matrix of either CCS or CRS

    Returns
    -------
    tuple of three arrays, i.e. (indptr, indices, vals), with integer data type
    of numpy.intc and floating data type numpy.float64.

    See Also
    --------
    :func:`as_sparse_matrix` : convert to sparse matrices
    """
    return np.ascontiguousarray(
        sp_mat.indptr, dtype=np.intc), np.ascontiguousarray(
            sp_mat.indices, dtype=np.intc), np.ascontiguousarray(
                sp_mat.data, dtype=np.float64)
