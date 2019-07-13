# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The PSMILU AUTHORS
# ----------------------------------------------------------------------------

# Authors:
#   Qiao,

"""Some utility functions

.. module:: psmilu4py.utils
.. moduleauthor:: Qiao Chen, <qiao.chen@stonybrook.edu>
"""

cimport psmilu4py as psmilu


__all__ = [
    'version',
    'is_warning',
    'enable_warning',
    'disable_warning',
]


def version():
    """Check the backend PSMILU version

    The version number is also adapted to be the `psmilu4py` version; the
    convension is ``global.major.minor``.
    """
    return psmilu.version().decode('utf-8')


def is_warning():
    """Check if underlying PSMILU enables warning"""
    return psmilu.warn_flag(-1)


def enable_warning():
    """Enable warning for underlying PSMILU routines"""
    psmilu.warn_flag(1)


def disable_warning():
    """Disable warning messages from PSMILU"""
    psmilu.warn_flag(0)
