"""
This script was used to perform the optimization in 
Paul, Landreman, Antonsen Section 3.3

This example does not require running the VMEC or ANIMEC codes.
"""
import sys
import os
import numpy as np

sys.path.append('../../')
from grad_optimizer import GradOptimizer
from vmec_optimization import VmecOptimization

vmecInputFilename = 'input.rotating_ellipse'
ntheta = 100 # Number of poloidal grid points 
nzeta = 100 # Number of toroidal grid points 
mmax_sensitivity = 3 # Maximum poloidal mode number
nmax_sensitivity = 3 # Maximum toroidal mode number
exp_weight = 0.01 # Exponential weight for proximity and radius objectives

vmecOptimizationObject = VmecOptimization(vmecInputFilename,
                         ntheta=ntheta,nzeta=nzeta,
                         mmax_sensitivity=mmax_sensitivity,
                         nmax_sensitivity=nmax_sensitivity)


def volumeObjective(boundary):
    volume = vmecOptimizationObject.input_objective(boundary=boundary,\
                                                which_objective='volume')
    return volume

def volumeObjectiveGrad(boundary):
    return vmecOptimizationObject.input_objective_grad(boundary=boundary,\
                                            which_objective='volume')

boundary = vmecOptimizationObject.boundary_opt
gradOptimizer = GradOptimizer(nparameters=len(boundary))
gradOptimizer.add_objective(volumeObjective,volumeObjectiveGrad,1)

[xopt,fopt,result] = gradOptimizer.optimize(boundary,package='scipy',\
                       method='BFGS',tol=1e-8,options={'disp':True,'gtol':1e-8})

np.savetxt('xopt.txt',xopt)
np.savetxt('fopt.txt',[fopt])
np.savetxt('result.txt',[result])
