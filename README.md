# ALPOpt

This is a python-based tool for gradient-based optimization of VMEC equilibria using adjoint methods. Details and examples are discussed in https://arxiv.org/abs/2012.10028. 

# Setup

Required python packages and variants I currently use: 
- mpi4py (3.0.3)
- numpy (1.19.1)
- scipy (1.5.2)
- nlopt (2.6.2)
- numba (0.51.2)
- scanf (1.5.2)
- f90nml (1.2)

Many objectives rely on the VMEC code. This can be installed from https://github.com/PrincetonUniversity/STELLOPT. 
- In order to run VMEC from the command line, the environment variable VMEC_PATH must be set to the absolute path to the executable.
- VMEC can alternatively be run from a python interface. This can be installed from https://github.com/ejpaul/simsopt. Once the interface is built, the directory simsopt.modules.vmec must be in your pythonpath. (The python interface to VMEC is currently maintained at https://gitlab.com/mattland/VMEC2000/-/tree/master)

The axis ripple objective relies on a variant of the ANIMEC code. This can be installed from https://github.com/ejpaul/ANIMEC_adjoint_ripple. 
- In order to run VMEC from the command line, the environment variable ANIMEC_PATH must be set to the absolute path to the executable.

# Examples

Several optimization examples are contained in the examples directory. 

# Testing

Each of the modules have unittesting in the test directory. These can be run with : python -m unittest test_\*.py
