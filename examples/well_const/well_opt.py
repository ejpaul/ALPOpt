"""
This script was used to perform the optimization in 
Paul, Landreman, Antonsen Section 4.2

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

mmax_sensitivity=10 # maximum poloidal mode number
nmax_sensitivity=10 # maximum toroidal mode number
delta_pres = 1e-1 # Amplitude of pressure perturbation
exp_weight = 0.01 # Exponential weight for proximity and radius objectives

# Weight functions for well calculation
weight_function1 = lambda s : np.exp(-(s-1)**2/0.1**2) - np.exp(-(s)**2/0.1**2)
weight_function2 = lambda s : np.exp(-(s-1)**2/0.1**2) + np.exp(-(s)**2/0.1**2)

input_filename = 'input.rotating_ellipse'
callVMECObject = CallVmecInterface(input_filename)
callVMEC_function = callVMECObject.call_vmec

if (MPI.COMM_WORLD.Get_rank()==0):
    verbose = True
else:
    verbose = False

vmecOptimizationObject = VmecOptimization(input_filename,
                                          callVMEC_function=callVMEC_function,
                                          mmax_sensitivity=mmax_sensitivity,
                                          nmax_sensitivity=nmax_sensitivity,
                                          verbose=verbose,delta_pres=delta_pres)

wellRatioObjective = lambda boundary: vmecOptimizationObject.vmec_objective(
                                 boundary=boundary,which_objective='well_ratio',
                                 weight_function1=weight_function1,
                                 weight_function2=weight_function2)

wellRatioObjectiveGrad = lambda boundary : vmecOptimizationObject.vmec_objective_grad(
                                boundary=boundary,which_objective='well_ratio',
                                 weight_function1=weight_function1,
                                 weight_function2=weight_function2)

def volumeObjective(boundary):
    vol = vmecOptimizationObject.input_objective(boundary,
          which_objective='volume')
    return 0.5*(vol-197.39208802178717)**2

def volumeObjectiveGrad(boundary):
    vol = vmecOptimizationObject.input_objective(boundary,
          which_objective='volume')
    volGrad = vmecOptimizationObject.input_objective_grad(boundary,
                which_objective='volume')
    return (vol-197.39208802178717)*volGrad    

def proximityObjective(boundary):
    exp = vmecOptimizationObject.input_objective(boundary,
          which_objective='summed_proximity',exp_weight=exp_weight,
                                                    min_curvature_radius=0.1)
    return exp

def proximityObjectiveGrad(boundary):
    expGrad = vmecOptimizationObject.input_objective_grad(boundary,
                which_objective='summed_proximity',exp_weight=exp_weight,
                                                    min_curvature_radius=0.1)
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

def curvatureObjective(boundary):
    exp = vmecOptimizationObject.input_objective(boundary,
          which_objective='curvature',max_curvature=7,exp_weight=1)
    return exp

def curvatureObjectiveGrad(boundary):
    expGrad = vmecOptimizationObject.input_objective_grad(boundary,
                which_objective='curvature',max_curvature=7,exp_weight=1)
    return expGrad

def curvatureIntegratedObjective(boundary):
    metric = vmecOptimizationObject.input_objective(boundary,
          which_objective='curvature_integrated',which_kappa=2)
    return metric

def curvatureIntegratedObjectiveGrad(boundary):
    metricGrad = vmecOptimizationObject.input_objective_grad(boundary,
                which_objective='curvature_integrated',which_kappa=2)
    return metricGrad

boundary = np.copy(vmecOptimizationObject.boundary_opt)

gradOptimizer = GradOptimizer(nparameters=len(boundary))
gradOptimizer.add_objective(wellRatioObjective,wellRatioObjectiveGrad,1)
gradOptimizer.add_objective(proximityObjective,proximityObjectiveGrad,1e-3)
gradOptimizer.add_objective(radiusObjective,radiusObjectiveGrad,1e2)
gradOptimizer.add_objective(curvatureObjective,curvatureObjectiveGrad,1e2)
gradOptimizer.add_objective(volumeObjective,volumeObjectiveGrad,1)
gradOptimizer.add_objective(curvatureIntegratedObjective,
                            curvatureIntegratedObjectiveGrad,1)

[xopt,fopt,result] = gradOptimizer.optimize(boundary,package='scipy',\
                       method='BFGS',tol=1e-8,options={'disp':True,'gtol':1e-8})

np.savetxt('xopt.txt',xopt)
np.savetxt('fopt.txt',[fopt])
np.savetxt('result.txt',[result])
