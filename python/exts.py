# -*- coding: utf-8 -*-

import os
import glob
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

_psmilu4py_debug = os.environ.get('PSMILU4PY_DEBUG', None)
_psmilu4py_debug = _psmilu4py_debug is not None

PSMILU_INCLUDE = os.environ.get('PSMILU_INCLUDE', None)
if PSMILU_INCLUDE is None:
    raise RuntimeError('you must specify PSMILU_INCLUDE (path/to/PSMILU.hpp)')
incs = ['.', PSMILU_INCLUDE]
LAPACK_LIB = os.environ.get('LAPACK_LIB', '-llapack')
_lapack_libs = LAPACK_LIB.split(' ')
for i in range(len(_lapack_libs)):
    _l = _lapack_libs[i]
    _lapack_libs[i] = _l[2:]
libs = []
libs += _lapack_libs
MC64_ROOT = os.environ.get('MC64_ROOT', None)
lib_dirs = []
if MC64_ROOT is not None:
    libs += ['mc64', 'gfortran']
    lib_dirs += [os.path.join(MC64_ROOT, 'lib')]
    macros = [('PSMILU4PY_USE_MC64', '1')]
else:
    macros = [('PSMILU4PY_USE_MC64', '0')]
LAPACK_LIB_ROOT = os.environ.get('LAPACK_LIB_ROOT', None)
if LAPACK_LIB_ROOT is not None:
    lib_dirs += [LAPACK_LIB_ROOT]
    rpath = [LAPACK_LIB_ROOT]
else:
    rpath = None


class BuildExt(build_ext):
    def _remove_flag(self, flag):
        try:
            self.compiler.compiler_so.remove(flag)
        except (AttributeError, ValueError):
            pass

    def build_extensions(self):
        self._remove_flag('-Wstrict-prototypes')
        if _psmilu4py_debug:
            self._remove_flag('-DNDEBUG')

        cpl_type = self.compiler.compiler_type

        def test_switch(flag):
            import tempfile
            with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
                f.write('int main(int argc, char *argv[]){return 0;}')
                try:
                    self.compiler.compile([f.name], extra_postargs=[flag])
                except Exception:
                    return False
            return True

        opts = []
        if cpl_type == 'unix':
            assert test_switch('-std=c++11'), 'must have C++11 support'
            opts.append('-std=c++11')
            if test_switch('-rdynamic'):
                opts.append('-rdynamic')
            if test_switch('-O3') and '-O3' not in self.compiler.compiler_so:
                opts.append('-O3')
        for ext in self.extensions:
            ext.extra_compile_args = opts
        super().build_extensions()


_pyx = glob.glob(os.path.join('psmilu4py', '*.pyx'))
exts = []

for f in _pyx:
    _f = f.split('.')[0]
    mod = '.'.join(_f.split(os.sep))
    exts.append(
        Extension(mod, [f],
                  language='c++',
                  include_dirs=incs,
                  libraries=libs,
                  library_dirs=lib_dirs,
                  define_macros=macros,
                  runtime_library_dirs=rpath))

_opts = {'language_level': 3, 'embedsignature': True}
if not _psmilu4py_debug:
    _opts.update({'wraparound': False, 'boundscheck': False})
exts = cythonize(exts, compiler_directives=_opts)
