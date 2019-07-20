# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The HILUCSI AUTHORS
# ----------------------------------------------------------------------------

# Authors:
#   Qiao,

# This is the implementation for Options

"""This is the main module contains the implementation of ``hilucsi4py``

The module wraps the internal components defined in HILUCSI and safely brings
them in Python3. This module includes:

1. multilevel preconditioner,
2. control parameters,
3. KSP solver(s)
4. IO with native HILUCSI binary and ASCII files

.. module:: hilucsi4py
.. moduleauthor:: Qiao Chen, <qiao.chen@stonybrook.edu>
"""

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.string cimport string as std_string
from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp.utility cimport pair
cimport hilucsi4py as hilucsi

import os
import numpy as np
from .utils import (convert_to_crs, convert_to_crs_and_b, _as_index_array, _as_value_array)


__all__ = [
    'version',
    'is_warning',
    'enable_warning',
    'disable_warning',
    'VERBOSE_NONE',
    'VERBOSE_INFO',
    'VERBOSE_PRE',
    'VERBOSE_FAC',
    'VERBOSE_PRE_TIME',
    'REORDER_OFF',
    'REORDER_AUTO',
    'REORDER_AMD',
    'REORDER_RCM',
    'Options',
    'read_hilucsi',
    'write_hilucsi',
    'query_hilucsi_info',
    'HILUCSI',
    'KSP_Error',
    'KSP_InvalidArgumentsError',
    'KSP_MSolveError',
    'KSP_DivergedError',
    'KSP_StagnatedError',
    'FGMRES'
]

# utilities
def version():
    """Check the backend HILUCSI version

    The version number is also adapted to be the `hilucsi4py` version; the
    convension is ``global.major.minor``.
    """
    return hilucsi.version().decode('utf-8')


def is_warning():
    """Check if underlying HILUCSI enables warning"""
    return hilucsi.warn_flag(-1)


def enable_warning():
    """Enable warning for underlying HILUCSI routines"""
    hilucsi.warn_flag(1)


def disable_warning():
    """Disable warning messages from HILUCSI"""
    hilucsi.warn_flag(0)


# Options
# redefine the verbose options, not a good idea but okay for now
VERBOSE_NONE = hilucsi.VERBOSE_NONE
VERBOSE_INFO = hilucsi.VERBOSE_INFO
VERBOSE_PRE = hilucsi.VERBOSE_PRE
VERBOSE_FAC = hilucsi.VERBOSE_FAC
VERBOSE_PRE_TIME = hilucsi.VERBOSE_PRE_TIME

# reorderingoptions
REORDER_OFF = hilucsi.REORDER_OFF
REORDER_AUTO = hilucsi.REORDER_AUTO
REORDER_AMD = hilucsi.REORDER_AMD
REORDER_RCM = hilucsi.REORDER_RCM

# determine total number of parameters
def _get_opt_info():
    raw_info = hilucsi.opt_repr(hilucsi.get_default_options()).decode('utf-8')
    # split with newline
    info = list(filter(None, raw_info.split('\n')))
    return [x.split()[0].strip() for x in info]


_OPT_LIST = _get_opt_info()


cdef class Options:
    """Python interface of control parameters

    By default, each control parameter object is initialized with default
    values in the paper. In addition, modifying the parameters can be achieved
    by using key-value pairs, i.e. `__setitem__`. The keys are the names of
    those defined in original C/C++ ``struct``.

    Here is a complete list of parameter keys: ``tau_L``, ``tau_U``, ``tau_d``,
    ``tau_kappa``, ``alpha_L``, ``alpha_U``, ``rho``, ``c_d``, ``c_h``, ``N``,
    and ``verbose``. Please consult the original paper and/or the C++
    documentation for default information regarding these parameters.

    Examples
    --------

    >>> from hilucsi4py import *
    >>> opts = Options()  # default parameters
    >>> opts['verbose'] = VERBOSE_INFO | VERBOSE_FAC
    >>> opts.reset()  # reset to default parameters
    """
    cdef hilucsi.Options opts

    def __init__(self):
        # for enabling docstring purpose
        pass

    def __cinit__(self):
        self.opts = hilucsi.get_default_options()

    def reset(self):
        """This function will reset all options to their default values"""
        self.opts = hilucsi.get_default_options()

    def enable_verbose(self, int flag):
        """Enable a verbose flag

        Parameters
        ----------
        flag : int
            enable a log flag, defined with variables starting with ``VERBOSE``
        """
        hilucsi.enable_verbose(<int> flag, self.opts)

    @property
    def verbose(self):
        """str: get the verbose flag(s)"""
        return hilucsi.get_verbose(self.opts).decode('utf-8')

    def __str__(self):
        return hilucsi.opt_repr(self.opts).decode('utf-8')

    def __repr__(self):
        return self.__str__()

    def __setitem__(self, str opt_name, v):
        # convert to double
        cdef:
            double vv = v
            std_string nm = opt_name.encode('utf-8')
        if hilucsi.set_option_attr[double](nm, vv, self.opts):
            raise KeyError('unknown option name {}'.format(opt_name))

    def __getitem__(self, str opt_name):
        if opt_name not in _OPT_LIST:
            raise KeyError('unknown option name {}'.format(opt_name))
        cdef int idx = _OPT_LIST.index(opt_name)
        attr = list(filter(None, self.__str__().split('\n')))[idx]
        v = list(filter(None, attr.split()))[1]
        if opt_name in ('check', 'reorder', 'verbose'):
            return v
        if hilucsi.option_dtypes[idx]:
            return float(v)
        return int(v)


# I/O
def read_hilucsi(str filename, *, is_bin=None):
    """Read a HILUCSI file

    Parameters
    ----------
    filename : str
        file name
    is_bin : ``None`` or bool (optional)
        if ``None``, then will automatically detect

    Returns
    -------
    `tuple` of `nrows`, `ncols`, `m`, `indptr`, `indices`, `vals`

    See Also
    --------
    :func:`write_hilucsi` : write native formats
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError
    cdef:
        std_string fn = filename.encode('utf-8')
        bool isbin
        vector[int] indptr
        vector[int] indices
        vector[double] vals
        size_t nrows = 0
        size_t ncols = 0
        size_t m = 0

    def is_binary():
        textchars = \
            bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
        fin = open(filename, 'rb')
        flag = <bool> fin.read(1024).translate(None, textchars)
        fin.close()
        return flag

    if is_bin is None:
        isbin = is_binary()
    else:
        isbin = <bool> is_bin
    hilucsi.read_hilucsi(fn, nrows, ncols, m, indptr, indices, vals, isbin)
    return \
        _as_index_array(indptr), \
        _as_index_array(indices), \
        _as_value_array(vals), (nrows, ncols), m



def _write_hilucsi(str filename, int[::1] rowptr not None,
    int[::1] colind not None, double[::1] vals not None, int nrows, int ncols,
    int m, bool is_bin):
    cdef:
        std_string fn = filename.encode('utf-8')
        bool isbin = is_bin
    hilucsi.write_hilucsi(fn, nrows, ncols, &rowptr[0], &colind[0], &vals[0],
        m, isbin)


def write_hilucsi(str filename, *args, shape=None, m=0, is_bin=True):
    """Write data to HILUCSI file formats

    Parameters
    ----------
    filename : str
        file name
    *args : input matrix
        either three array of CRS or scipy sparse matrix
    shape : ``None`` or tuple
        if input is three array, then this must be given
    m : int (optional)
        leading symmetric block
    is_bin : bool (optional)
        if ``True`` (default), then assume binary file format

    See Also
    --------
    :func:`read_hilucsi` : read native formats
    """
    # essential checkings to avoid segfault
    cdef:
        size_t m0 = m
        bool isbin = is_bin
        size_t n
    rowptr, colind, vals = convert_to_crs(*args, shape=shape)
    assert len(rowptr), 'cannot write empty matrix'
    n = len(rowptr) - 1
    _write_hilucsi(filename, rowptr, colind, vals, n, n, m0, isbin)


def query_hilucsi_info(str filename, *, is_bin=None):
    """Read a HILUCSI file and only query its information

    Parameters
    ----------
    filename : str
        file name
    is_bin : ``None`` or bool (optional)
        if ``None``, then will automatically detect

    Returns
    -------
    `tuple` of `is_row`, `is_c`, `is_double`, `is_real`, `nrows`, `ncols`,
    `nnz`, and `m`
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError
    cdef:
        std_string fn = filename.encode('utf-8')
        bool isbin
        bool is_row
        bool is_c
        bool is_double
        bool is_real
        uint64_t nrows
        uint64_t ncols
        uint64_t nnz
        uint64_t m
    
    def is_binary():
        textchars = \
            bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
        fin = open(filename, 'rb')
        flag = <bool> fin.read(1024).translate(None, textchars)
        fin.close()
        return flag

    if is_bin is None:
        isbin = is_binary()
    else:
        isbin = <bool> is_bin
    hilucsi.query_hilucsi_info(fn, is_row, is_c, is_double, is_real, nrows,
        ncols, nnz, m, isbin)
    return is_row, is_c, is_double, is_real, nrows, ncols, nnz, m


cdef class HILUCSI:
    """Python HILUCSI object

    The interfaces remain the same as the original user object, i.e.
    `hilucsi::DefaultHILUCSI`. However, we significantly the parameters by
    hiding the needs of `hilucsi::CRS`, `hilucsi::CCS`, and `hilucsi::Array`.
    Therefore, the interface is very generic and easily adapted to any other
    Python modules without the hassle of complete object types.

    .. note::

        In addition, the only supported data types are ``int`` and ``double``.
    
    Examples
    --------

    >>> from scipy.sparse import random
    >>> from hilucsi4py import *
    >>> A = random(10, 10, 0.5)
    >>> M = HILUCSI()
    >>> M.factorize(A)
    """
    cdef shared_ptr[hilucsi.PyHILUCSI] M

    def __init__(self):
        # for docstring purpose
        pass

    def __cinit__(self):
        self.M.reset(new hilucsi.PyHILUCSI())

    def empty(self):
        """Check if or not the builder is empty"""
        return deref(self.M).empty()

    @property
    def levels(self):
        """int: number of levels"""
        return deref(self.M).levels()

    @property
    def nnz(self):
        """int: total number of nonzeros of all levels"""
        return deref(self.M).nnz()

    @property
    def nnz_EF(self):
        """int: total number of nonzeros in Es and Fs"""
        return deref(self.M).nnz_EF()

    @property
    def nnz_LDU(self):
        """int: total number of nonzeros in LDU fators"""
        return deref(self.M).nnz_LDU()

    @property
    def size(self):
        """int: system size"""
        return deref(self.M).nrows()

    def stats(self, int entry):
        """Get the statistics information

        Parameters
        ----------
        entry: int
            entry field
        """
        return deref(self.M).stats(entry)

    def _factorize(self, int[::1] rowptr not None, int[::1] colind not None,
        double[::1] vals not None, size_t n, size_t m, Options opts):
        cdef:
            size_t m0 = m
            Options my_opts = Options()
        if opts is not None:
            my_opts.opts = opts.opts
        deref(self.M).factorize(n, &rowptr[0], &colind[0], &vals[0], m0,
            my_opts.opts)

    def factorize(self, *args, shape=None, m=0, Options opts=None):
        """Compute/build the preconditioner

        Parameters
        ----------
        *args : input matrix
            either three array of CRS or scipy sparse matrix
        shape : ``None`` or tuple
            if input is three array, then this must be given
        m0 : int
            leading symmetric block
        opts : :py:class:`psmilu4py.Options` (optional)
            control parameters, if ``None``, then use the default values

        See Also
        --------
        :func:`solve`: solve for inv(HILUCSI)*x
        """
        cdef:
            size_t n
        rowptr, colind, vals = convert_to_crs(*args, shape=shape)
        assert len(rowptr), 'cannot deal with empty matrix'
        n = len(rowptr) - 1
        self._factorize(rowptr, colind, vals, n, m, opts)

    def _solve(self, double[::1] b not None, double[::1] x not None):
        cdef size_t n = len(b)
        assert n == len(x)
        deref(self.M).solve(n, &b[0], &x[0])

    def solve(self, b, x=None):
        r"""Core routine to use the preconditioner

        Essentailly, this routine is to perform
        :math:`\boldsymbol{x}=\boldsymbol{M}^{-1}\boldsymbol{b}`, where
        :math:`\boldsymbol{M}` is our MILU preconditioner.

        Parameters
        ----------
        b : array-like
            right-hand side parameters
        x : array-like (output) buffer (optional)
            solution vector
        """
        bb = _as_value_array(b)
        assert len(bb.shape) == 1
        if x is None:
            xx = np.empty_like(bb)
        else:
            xx = _as_value_array(x)
        assert xx.shape == bb.shape, 'inconsistent x and b'
        self._solve(bb, xx)
        return xx


class KSP_Error(RuntimeError):
    """Base class of KSP error exceptions"""
    pass


class KSP_InvalidArgumentsError(KSP_Error):
    """Invalid input argument error"""
    pass


class KSP_MSolveError(KSP_Error):
    """preconditioner solve error"""
    pass


class KSP_DivergedError(KSP_Error):
    """Diverged error, i.e. maximum iterations exceeded"""
    pass


class KSP_StagnatedError(KSP_Error):
    """Stagnated error, i.e. no improvement amount two iterations in a row"""
    pass


class KSP_BreakDownError(KSP_Error):
    """Solver breaks down error"""
    pass


def _handle_flag(int flag):
    """handle KSP returned flag value"""
    if flag != hilucsi.SUCCESS:
        if flag == hilucsi.INVALID_ARGS:
            raise KSP_InvalidArgumentsError
        if flag == hilucsi.M_SOLVE_ERROR:
            raise KSP_MSolveError
        if flag == hilucsi.DIVERGED:
            raise KSP_DivergedError
        if flag == hilucsi.STAGNATED:
            raise KSP_StagnatedError
        raise KSP_BreakDownError


cdef class FGMRES:
    r"""Flexible GMRES implementation with rhs preconditioner

    The FMGRES implementation has three modes (kernels): the first one is the
    ``traditional`` kernel by treating :math:`\boldsymbol{M}` as the rhs
    preconditioner; the second one is ``jacobi`` fashion, where
    :math:`\boldsymbol{M}` is treated as the "diagonal" term in Jacobi iteration
    combining with the input matrix :math:`\boldsymbol{A}`; finally, the last
    fashion is an extension to ``jacobi`` with Chebyshev acceleration, which
    requires estimations of the largest and smallest eigenvalues.

    Parameters
    ----------
    M : :class:`HILUCSI` or ``None``
        preconditioner
    rtol : float
        relative tolerance, default is 1e-6
    maxit : int
        maximum iterations, default is 500
    restart : int
        restart in GMRES, default is 30
    max_inners : int
        maximum inner iterations used in Jacobi style kernle, default is 4
    lamb1 : float or ``None``
        if given, then used as the largest eigenvalue estimation
    lamb2 : float or ``None``
        if given, then used as the smallest eigenvalue estimation

    Examples
    --------

    >>> from scipy.sparse import random
    >>> from hilucsi4py import *
    >>> import numpy as np
    >>> A = random(10,10,0.5)
    >>> M = HILUCSI()
    >>> M.factorize(A)
    >>> solver = FGMRES(M)
    >>> x = solver.solve(A, np.random.rand(10))
    """
    cdef shared_ptr[hilucsi.PyFGMRES] solver

    def __init__(self, M=None, rtol=1e-6, restart=30, maxit=500, max_inners=4,
                 **kw):
        pass

    def __cinit__(self, HILUCSI M=None, double rtol=1e-6, int restart=30,
                  int maxit=500, int max_inners=4, **kw):
        self.solver.reset(new hilucsi.PyFGMRES())
        if M is not None:
            deref(self.solver).set_M(M.M)
        deref(self.solver).rtol = rtol
        deref(self.solver).restart = restart
        deref(self.solver).maxit = maxit
        deref(self.solver).max_inners = max_inners
        deref(self.solver).check_pars()
        lamb1 = kw.pop('lamb1', None)
        if lamb1 is not None:
            deref(self.solver).lamb1 = lamb1
        lamb2 = kw.pop('lamb2', None)
        if lamb2 is not None:
            deref(self.solver).lamb2 = lamb2

    @property
    def rtol(self):
        """float: relative convergence tolerance (1e-6)"""
        return deref(self.solver).rtol

    @rtol.setter
    def rtol(self, double v):
        if v <= 0.0:
            raise ValueError('rtol must be positive')
        deref(self.solver).rtol = v

    @property
    def maxit(self):
        """int: maximum number of interations (500)"""
        return deref(self.solver).maxit

    @maxit.setter
    def maxit(self, int max_iters):
        if max_iters <= 0:
            raise ValueError('maxit must be positive integer')
        deref(self.solver).maxit = max_iters

    @property
    def restart(self):
        """int: restart for GMRES (30)"""
        return deref(self.solver).restart

    @restart.setter
    def restart(self, int rs):
        if rs <= 0:
            raise ValueError('restart must be positive integer')
        deref(self.solver).restart = rs

    @property
    def max_inners(self):
        """int: maximum inner iterations for Jacobi-like inner iterations (4)"""
        return deref(self.solver).max_inners

    @max_inners.setter
    def max_inners(self, int maxi):
        if maxi <= 0:
            raise ValueError('max_inners must be positive integer')
        deref(self.solver).max_inners = maxi

    @property
    def lamb1(self):
        """float: largest eigenvalue estimation"""
        return deref(self.solver).lamb1

    @lamb1.setter
    def lamb1(self, double v):
        deref(self.solver).lamb1 = v

    @property
    def lamb2(self):
        """float: smallest eigenvalue estimation"""
        return deref(self.solver).lamb2

    @lamb2.setter
    def lamb2(self, double v):
        deref(self.solver).lamb2 = v

    def set_M(self, HILUCSI M):
        deref(self.solver).set_M(M.M)

    @property
    def resids(self):
        """list: list of history residuals"""
        cdef:
            vector[double] res = vector[double](deref(self.solver).get_iters())
        deref(self.solver).get_resids(res.data())
        return res

    def _solve(self, int[::1] rowptr, int[::1] colind, double[::1] vals,
        double[::1] b, double[::1] x, int kernel, bool with_init_guess,
        bool trunc, bool verbose):
        cdef:
            size_t n = b.size
            bool wg = with_init_guess
            bool tr = trunc
            bool v = verbose
            pair[int, size_t] info
        info = deref(self.solver).solve(n, &rowptr[0], &colind[0], &vals[0],
            &b[0], &x[0], kernel, wg, tr, v)
        return info.first, info.second

    def solve(self, *args, shape=None, x=None, kernel='tradition',
        init_guess=False, trunc=False, verbose=True):
        """Sovle the rhs solution

        Parameters
        ----------
        *args : input matrix
            either three array of CRS or scipy sparse matrix, and rhs b
        shape : ``None`` or tuple (optional)
            if input is three array, then this must be given
        x : ``None`` or buffer of solution
            for efficiency purpose, one can provide the buffer
        kernel : str (optional)
            kernel for preconditioning, either 'tradition', 'jacobi', or
            'chebyshev-jacobi'
        init_guess : bool (optional)
            if ``False`` (default), then set initial state to be zeros
        trunc : bool (optional)
            if ``False`` (default), then do typical hard restart
        verbose : bool (optional)
            if `True`` (default), then enable verbose printing

        Returns
        -------
        tuple of solutions and iterations used.

        Raises
        ------
        KSP_InvalidArgumentsError
            invalid input arguments
        KSP_MSolveError
            preconditioenr solver error, see :func:`HILUCSI.solve`
        KSP_DivergedError
            iterations diverge due to exceeding :attr:`maxit`
        KSP_StagnatedError
            iterations stagnate
        KSP_BreakDownError
            solver breaks down
        """
        assert kernel in ('tradition', 'jacobi', 'chebyshev-jacobi')
        if init_guess and x is None:
            raise KSP_InvalidArgumentsError('init-guess missing x0')
        cdef:
            int kn
        if kernel == 'tradition':
            kn = hilucsi.PyFGMRES_TRADITION
        elif kernel == 'jacobi':
            kn = hilucsi.PyFGMRES_JACOBI
        else:
            kn = hilucsi.PyFGMRES_CHEBYSHEV_JACOBI
        rowptr, colind, vals, b = convert_to_crs_and_b(*args, shape=shape)
        if x is None:
            xx = np.empty_like(b)
        else:
            xx = _as_value_array(x)
        if xx.shape != b.shape:
            raise ValueError('inconsistent x and b')
        flag, iters = self._solve(rowptr, colind, vals, b, xx, kn, init_guess,
            trunc, verbose)
        _handle_flag(flag)
        return xx, iters
