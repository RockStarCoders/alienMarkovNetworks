# cython makefile
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
#    ext_modules = [Extension("helloworld", ["helloworld.pyx"])]
    ext_modules = [Extension("cython_uflow", 
                             sources=["cython_uflow.pyx", "uflow.cpp", "graph.cpp", "maxflow.cpp"],
                             include_dirs=[numpy.get_include()],
                             language = "c++")],
)
