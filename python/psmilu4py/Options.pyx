# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The PSMILU AUTHORS
# ----------------------------------------------------------------------------

# Authors:
#   Qiao,

# This is the implementation for Options

"""Python interface of `psmilu::Options`

The design of :class:`Options` follows the standard Python `dict` as well as
easy to support the C++ backend.

.. module:: psmilu4py.Options
.. moduleauthor:: Qiao Chen, <qiao.chen@stonybrook.edu>
"""

# from cython.operator cimport dereference as deref
from libcpp.string cimport string as std_string
cimport psmilu4py as psmilu


__all__ = [
    'VERBOSE_NONE',
    'VERBOSE_INFO',
    'VERBOSE_PRE',
    'VERBOSE_FAC',
    'VERBOSE_MEM',
    'REORDER_OFF',
    'REORDER_AUTO',
    'REORDER_AMD',
    'REORDER_RCM',
    'REORDER_KING',
    'REORDER_SLOAN',
    'Options',
]


# redefine the verbose options, not a good idea but okay for now
VERBOSE_NONE = 0
VERBOSE_INFO = 1
VERBOSE_PRE = VERBOSE_INFO << 1
VERBOSE_FAC = VERBOSE_PRE << 1
VERBOSE_MEM = VERBOSE_FAC << 1


# reorderingoptions
REORDER_OFF = 0
REORDER_AUTO = 1
REORDER_AMD = 2
REORDER_RCM = 3
REORDER_KING = 4
REORDER_SLOAN = 5


# determine total number of parameters
def _get_opt_info():
    raw_info = psmilu.opt_repr(psmilu.get_default_options()).decode('utf-8')
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

    >>> from psmilu4py import *
    >>> opts = Options()  # default parameters
    >>> opts['verbose'] = VERBOSE_INFO | VERBOSE_FAC
    >>> opts.reset()  # reset to default parameters
    """
    def __init__(self):
        # for enabling docstring purpose
        pass

    def __cinit__(self):
        self.opts = psmilu.get_default_options()

    def reset(self):
        """This function will reset all options to their default values"""
        self.opts = psmilu.get_default_options()

    def enable_verbose(self, int flag):
        """Enable a verbose flag"""
        psmilu.enable_verbose(<int> flag, self.opts)

    @property
    def verbose(self):
        """str: get the verbose flag(s)"""
        return psmilu.get_verbose(self.opts).decode('utf-8')

    def __str__(self):
        return psmilu.opt_repr(self.opts).decode('utf-8')

    def __repr__(self):
        return self.__str__()

    def __setitem__(self, str opt_name, v):
        # convert to double
        cdef:
            double vv = v
            std_string nm = opt_name.encode('utf-8')
        if psmilu.set_option_attr[double](nm, vv, self.opts):
            raise KeyError('unknown option name {}'.format(opt_name))

    def __getitem__(self, str opt_name):
        if opt_name not in _OPT_LIST:
            raise KeyError('unknown option name {}'.format(opt_name))
        cdef int idx = _OPT_LIST.index(opt_name)
        attr = list(filter(None, self.__str__().split('\n')))[idx]
        if psmilu.option_dtypes[idx]:
            return float(list(filter(None, attr.split()))[1])
        return int(list(filter(None, attr.split()))[1])
