#!/usr/bin/env python3

import sys
import os
import numpy as np
from scipy.io import mmread
from psmilu4py.scipy import write_native_psmilu

try:
    case = sys.argv[1]
except IndexError:
    print('you need to specify the mm case base', file=sys.stderr)
    sys.exit(1)

sym = False
svl = True
for arg in sys.argv[1:]:
    if arg == '-s':
        sym = True
    elif arg == '-n':
        svl = False

A_file = case + '.mtx'
b_file = case + '_rhs1.mtx'
b_file2 = case + '_b.mtx'

A = mmread(A_file)
A = A.tocsr()
A.sort_indices()

have_rhs = True

try:
    b = mmread(b_file).reshape(-1)
except FileNotFoundError:
    try:
        b = mmread(b_file2).reshape(-1)
    except FileNotFoundError:
        print('no rhs found for {}, use A*1 instead'.format(case), file=sys.stderr)
        have_rhs = False
        b = A.dot(np.ones(A.shape[0]))

if svl:
    if have_rhs:
        from scipy.sparse.linalg import spsolve
        print('solving the system...')
        x = spsolve(A, b)
    else:
        x = np.ones(A.shape[0])

try:
    os.mkdir(case)
except OSError:
    pass

m = 0 if not sym else A.shape[0]
write_native_psmilu(case + '/A.psmilu', A, m=m)
np.savetxt(case + '/b.txt', b)
if svl:
    np.savetxt(case + '/ref_x.txt', x)
