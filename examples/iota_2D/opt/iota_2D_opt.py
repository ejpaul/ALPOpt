"""
This script was used to perform the optimization in 
Paul, Landreman, Antonsen Appendix A

This example uses the VMEC interface included in simsopt. Make sure that
    simsopt.modules.vmec is added to your path
"""
import sys
import os
import numpy as np
from mpi4py import MPI

sys.path.append('../../../')
from grad_optimizer import GradOptimizer
from call_vmec import CallVmecInterface
from vmec_optimization import VmecOptimization

vmecInputFilename = 'input.rotating_ellipse'
which_objective = 'iota_target'
delta = 10 # Amplitude of current perturbation 
ntheta = 200
nzeta = 200
name = 'optimization'
iota_target = 0.618034 # Target iota value

callVMECObject = CallVmecInterface(vmecInputFilename)
callVMEC_function = callVMECObject.call_vmec

# Optimize with respect to these two modes
xm_sensitivity = np.array([1])
xn_sensitivity = np.array([1])

if (MPI.COMM_WORLD.Get_rank()==0):
    verbose = True
else:
    verbose = False

vmecOptimizationObject = VmecOptimization(vmecInputFilename,
                             callVMEC_function=callVMEC_function,
                             delta_curr=delta,verbose=verbose,ntheta=ntheta,
                             nzeta=nzeta,xm_sensitivity=xm_sensitivity,
                             xn_sensitivity=xn_sensitivity)

boundary = vmecOptimizationObject.boundary_opt

# Weight function for iota_target objective
def weight(s):
    return 1

def iotaTargetObjective(boundary):
    iota_func = vmecOptimizationObject.vmec_objective(boundary=boundary,
                    which_objective=which_objective,weight_function=weight,
                                                  iota_target=iota_target)
    return 0.5*(iota_func - 0.06)**2

def iotaTargetObjectiveGrad(boundary):
    iota_func = vmecOptimizationObject.vmec_objective(boundary=boundary,
                    which_objective=which_objective,weight_function=weight,
                                                  iota_target=iota_target)
    iota_target_grad = vmecOptimizationObject.vmec_objective_grad(boundary=boundary,
                    which_objective=which_objective,weight_function=weight,
                                                  iota_target=iota_target)
    return (iota_func - 0.06)*iota_target_grad

gradOptimizer = GradOptimizer(nparameters=2)
gradOptimizer.add_objective(iotaTargetObjective,iotaTargetObjectiveGrad,1)

[xopt,fopt,result] = gradOptimizer.optimize(boundary,tol=1e-12,package='scipy',
                               method='BFGS',options={'disp':True,'gtol':1e-12})

# Save output
np.savetxt('xopt.txt',xopt) # Optimum parameters
np.savetxt('fopt.txt',[fopt]) # Optimum objective value
np.savetxt('result.txt',[result]) # Result of BFGS 