"""
This script was used to perform the optimization in 
Paul, Landreman, Antonsen Section 4.1

This example uses the VMEC interface included in simsopt. Make sure that
    simsopt.modules.vmec is added to your path
"""
import sys
from mpi4py import MPI
import numpy as np

sys.path.append('../../')
from grad_optimizer import GradOptimizer
from call_vmec import CallVmecInterface
from vmec_optimization import VmecOptimization

mmax_sensitivity=10 # maximum poloidal mode number to vary
nmax_sensitivity=10 # maximum toroidal mode number to vary
iota_target = 0.381966 # Target iota value 
exp_weight = 0.01 # Exponential weight for proximity and radius objectives

input_filename = 'input.rotating_ellipse'
callVMECObject = CallVmecInterface(input_filename)
callVMEC_function = callVMECObject.call_vmec

if (MPI.COMM_WORLD.Get_rank()==0):
    verbose = True
else:
    verbose = False

def weight_function(s):
    return 1

vmecOptimizationObject = VmecOptimization(input_filename,
                            callVMEC_function=callVMEC_function,
                            mmax_sensitivity=mmax_sensitivity,
                            nmax_sensitivity=nmax_sensitivity,verbose=verbose)

iotaObjective = lambda boundary : vmecOptimizationObject.vmec_objective(boundary=boundary,
                                    which_objective='iota_target',weight_function=weight_function,
                                    iota_target=iota_target)

iotaObjectiveGrad = lambda boundary : vmecOptimizationObject.vmec_objective_grad(boundary=boundary,
                                    which_objective='iota_target',weight_function=weight_function,
                                    iota_target=iota_target)

def proximityObjective(boundary):
    exp = vmecOptimizationObject.input_objective(boundary,
          which_objective='summed_proximity',exp_weight=exp_weight,
                                            min_curvature_radius=0.2)
    return exp

def proximityObjectiveGrad(boundary):
    expGrad = vmecOptimizationObject.input_objective_grad(boundary,
                which_objective='summed_proximity',exp_weight=exp_weight,
                                                min_curvature_radius=0.2)
    return expGrad
    
def radiusObjective(boundary):
    radius = vmecOptimizationObject.input_objective(boundary,
          which_objective='summed_radius',R_min=0.2,exp_weight=exp_weight)
    return radius

def radiusObjectiveGrad(boundary):
    radiusGrad = \
        vmecOptimizationObject.input_objective_grad(boundary,
                which_objective='summed_radius',R_min=0.2,exp_weight=exp_weight)
    return radiusGrad

boundary = np.copy(vmecOptimizationObject.boundary_opt)

gradOptimizer = GradOptimizer(nparameters=len(boundary))
gradOptimizer.add_objective(iotaObjective,iotaObjectiveGrad,1)
gradOptimizer.add_objective(proximityObjective,proximityObjectiveGrad,1e3)
gradOptimizer.add_objective(radiusObjective,radiusObjectiveGrad,1e3)

[xopt,fopt,result] = gradOptimizer.optimize(boundary,package='scipy',\
                       method='BFGS',tol=1e-8,options={'disp':True,'gtol':1e-8})

np.savetxt('xopt.txt',xopt)
np.savetxt('fopt.txt',[fopt])
np.savetxt('result.txt',[result])