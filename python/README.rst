Python3 Interface for PS-MILU
=============================

Welcome to the Python3 interface of PSMILU package--- *psmilu4py*. The Python
interface is implemented with Cython, and the Cython interface is also
available to use.

Dependencies
------------

*psmilu4py* requires *Cython* and *setuptools* during installation and
complilation. The must-have runtime package is *numpy*. If you plan to use the
*scipy* hooks, then you need to make sure *scipy* is available as well.

Roughly speaking, *psmilu4py* enables three core components in PSMILU:

1. building multilevel ILU preconditioner,
2. solving preconditiner system, and
3. configuring preconditiner parameters.

Installation
-------------

Notice that since PSMILU is template-based package, you need to specify
the include path (path to ``PSMILU.hpp``) as environment variable.
In addition, you need to configure the LAPACK/BLAS linking, the default
is ``-llapack`` ; you can also modify the library path to them.
Finally, HSL_MC64 can be automatically enabled by specifying the root
to the package.

Overall, the following envrionment variables can be configured

1. ``PSMILU_INCLUDE``, **must be specified!**
2. ``LAPACK_LIB``, default is ``-llapack``
3. ``LAPACK_LIB_ROOT``, default is empty
4. ``MC64_ROOT``, path contains the header/lib in ``include``/``lib``.

It's worth noting that ``PSMILU_INCLUDE`` is also needed if you plan to use
Cython interface of *psmilu4py* due to the need of recompilation.

Examples

.. code:: console

    # specify psmilu code generator
    env PSMILU_INCLUDE=/usr/local/include pip install . --user

    # user alternative blas, bulid in $PWD
    env LAPACK_LIB=-lopenblas LAPACK_LIB_ROOT=/opt/OpenBLAS/lib \
        pip3 install . --user

    # use mc64
    env MC64_ROOT=$HOME/hsl_mc64-2.3.1 pip3 install . --user

Contacts
--------

Qiao (Chiao) Chen, <qiao.chen@stonybrook.edu>
