import sys
import glob
import os.path as op
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize


def requirements():
    """Returns the content of requirements.txt"""
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip()]


def extension_kwargs(f):
    """Disables new Numpy Cython API for dist_metrics and boruvka module.
    
    The dist metrics module requires inheritence with inline functions.
    This is not supported by the new API. Removing the inline attribute from these
    functions makes the code compile, but results in noticable slowdowns.
    Disabling the new API is the better choice.
    """
    import numpy as np
    
    if sys.platform == "win32":
        cpp_flags = ["/O2"]
    else:
        cpp_flags = ["-O2", "-DNDEBUG"]

    kwargs = {
        'include_dirs': [np.get_include()],
        'extra_compile_args': cpp_flags,
    }
    if 'dist_metrics' in f:
        return kwargs
    kwargs['language'] = 'c++'
    kwargs['define_macros'] = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    return kwargs

extensions = cythonize(
    [
        Extension(
            f.replace(".pyx", "").replace(op.sep, "."),
            [f],
            **extension_kwargs(f)
        )
        for f in glob.glob(op.join("flasc", "*.pyx"))
        if op.isfile(f)
    ],
    language_level=3,
)

setup(
    ext_modules=extensions, 
    install_requires=requirements()
)
