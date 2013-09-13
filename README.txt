INSTALL
-------

You will have to install some python packages, like numpy, sklearn, etc.

Some pip installed like pybrain.

Some you download, make sure they are on your python path:

  slic-python:  git clone https://github.com/amueller/slic-python.git

You will need the microsoft data set probably, from:

  http://research.microsoft.com/en-us/downloads/b94de342-60dc-45d0-830b-9f6eff91b301/default.aspx



BUILD
-----

It's python, you don't.  But you do have to build the C++ cython bindings:

   > cd maxflow
   > python setup.py build_ext --inplace

