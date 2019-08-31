# -*- coding: utf-8 -*-

import os
import glob
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize


def is_debug():
    flag = os.environ.get('HILUCSI4PY_DEBUG', None)
    if flag is None:
        return False
    return flag.lower() not in ('0', 'no', 'off', 'false')


_hilucsi4py_debug = is_debug()

# configure include paths
_hilucsi_inc_path = os.environ.get('HILUCSI_INCLUDE_PATH', '')
if not _hilucsi_inc_path:
    incs = ['.']
elif _hilucsi_inc_path != os.getcwd() or _hilucsi_inc_path != '.':
    incs = ['.', _hilucsi_inc_path]

# configure libraries
_lapack_lib = os.environ.get('HILUCSI_LAPACK_LIB', '-llapack')
_lapack_libs = _lapack_lib.split(' ')
for i, _l in enumerate(_lapack_libs):
    _lapack_libs[i] = _l[2:]
libs = _lapack_libs

# configure library paths
lib_dirs = None
_lapack_path = os.environ.get('HILUCSI_LAPACK_LIB_PATH', '')
if _lapack_path:
    lib_dirs = [_lapack_path]
rpath = None if lib_dirs is None else lib_dirs


class BuildExt(build_ext):
    def _remove_flag(self, flag):
        try:
            self.compiler.compiler_so.remove(flag)
        except (AttributeError, ValueError):
            pass

    def build_extensions(self):
        self._remove_flag('-Wstrict-prototypes')
        if _hilucsi4py_debug:
            self._remove_flag('-DNDEBUG')

        cpl_type = self.compiler.compiler_type

        def test_switch(flag):
            import tempfile
            with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
                f.write('int main(int argc, char *argv[]){return 0;}')
                try:
                    self.compiler.compile([f.name], extra_postargs=[flag])
                except BaseException:
                    return False
            return True

        opts = []
        if cpl_type == 'unix':
            assert test_switch('-std=c++11'), 'must have C++11 support'
            if test_switch('-std=c++1z'):
                opts.append('-std=c++1z')
            else:
                opts.append('-std=c++11')
            if test_switch('-rdynamic'):
                opts.append('-rdynamic')
            if test_switch('-O3') and '-O3' not in self.compiler.compiler_so:
                opts.append('-O3')
        for ext in self.extensions:
            ext.extra_compile_args = opts
        super().build_extensions()


_pyx = glob.glob(os.path.join('hilucsi4py', '*.pyx'))
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
                  runtime_library_dirs=rpath))

_opts = {'language_level': 3, 'embedsignature': True}
if not _hilucsi4py_debug:
    _opts.update({'wraparound': False, 'boundscheck': False})
exts = cythonize(exts, compiler_directives=_opts)
