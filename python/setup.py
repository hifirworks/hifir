# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from exts import BuildExt, exts

# for now
version = '0.0.0'

setup(
    version=version,
    packages=find_packages(),
    ext_modules=exts,
    cmdclass={'build_ext': BuildExt},
)
