# -*- coding: utf-8 -*-

from setuptools import setup
from conf import BuildExt, exts

setup(
    ext_modules=exts,
    cmdclass={'build_ext': BuildExt},
)
