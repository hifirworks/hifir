# -*- coding: utf-8 -*-
###############################################################################
#                 This file is part of HILUCSI4PY project                     #
###############################################################################

from setuptools import setup
from conf import BuildExt, exts

setup(ext_modules=exts, cmdclass={"build_ext": BuildExt})
