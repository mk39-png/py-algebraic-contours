"""
Compile Cython functions.
Exists despite using Hatchling so that we can compile Cython while pyalgcon is installed via 
"pip install -e ."
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize(module_list=["src/pyalgcon/core/common_c.pyx",
                                       "src/pyalgcon/core/polynomial_function_c.pyx",
                                       "src/pyalgcon/contour_network/compute_closed_contours_c.pyx"],
                          compiler_directives={"language_level": "3"}),
    include_dirs=[np.get_include()]
)
