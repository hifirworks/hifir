# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#                Copyright (C) 2019 The HILUCSI AUTHORS
# ----------------------------------------------------------------------------

from hilucsi4py import FQMRCGSTAB
from scipy.sparse import random
import numpy as np


def test_fqmrcgstab():
    A = random(10, 10, 0.5)
    solver = FQMRCGSTAB()
    solver.M.factorize(A)
    b = A * np.ones(10)
    x, _ = solver.solve(A, b)
    res = np.linalg.norm(x - 1) / np.linalg.norm(b)
    assert res <= 1e-6
    assert abs(res - solver.resids[-1]) <= 1e-14
