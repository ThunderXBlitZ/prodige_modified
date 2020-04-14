"""
Does all sorts of dark magic in order to build/import c++ bfs
"""
import os
import os.path as osp
import random

import setuptools.sandbox
from multiprocessing import cpu_count
# from lib import cpp
import numpy as np
import torch

from lib import batch_dijkstra_algo

"""
package_abspath = osp.join(*osp.split(osp.abspath(__file__))[:-1])
if not os.path.exists(osp.join(package_abspath, "_bindings.so")):
    # try build _bfs.so
    workdir = os.getcwd()
    try:
        os.chdir(package_abspath)
        setuptools.sandbox.run_setup(
            osp.join(package_abspath, "setup.py"), ["clean", "build"]
        )
        os.system(
            "cp {}/build/lib*/*.so {}/_bindings.so".format(package_abspath, package_abspath)
        )
        assert os.path.exists(osp.join(package_abspath, "_bindings.so"))
    finally:
        os.chdir(workdir)

from . import _bindings
_bindings.set_seed(random.randint(0, 2 ** 16))
"""