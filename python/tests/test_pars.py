# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The HILUCSI AUTHORS
# ----------------------------------------------------------------------------

from hilucsi4py import FGMRES


def test_pars():
    solver = FGMRES()
    print(solver)
    assert solver.rtol == 1e-6
    solver.maxit = 100
    assert solver.maxit == 100
