# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The HILUCSI AUTHORS
# ----------------------------------------------------------------------------

from hilucsi4py import *
from scipy.sparse import random
import numpy as np


def test_random():
    A = random(10, 10, 0.5)
    M = HILUCSI()
    M.factorize(A)
    b = A * np.ones(10)
    solver = FGMRES(M)
    x, _ = solver.solve(A, b)
    assert np.linalg.norm(x - 1) <= 1e-6
