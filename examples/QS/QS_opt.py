"""
This script was used to perform the optimization in 
Paul, Landreman, Antonsen Section 4.3

Set ANIMEC_PATH environmnet variable in order to run this example 
This example uses the VMEC interface included in simsopt. Make sure that
    simsopt.modules.vmec is added to your path
"""
import sys
import numpy as np
import os

sys.path.append('../../')
from grad_optimizer import GradOptimizer
from call_vmec import CallVmecInterface, CallVmecCommandline
from vmec_optimization import VmecOptimization

input_filename = 'input.rotating_ellipse'
callVMECObject = CallVmecInterface(input_filename)
callVMEC_function = callVMECObject.call_vmec

path_to_executable_animec = os.environ.get('ANIMEC_PATH')
callANIMECObject = CallVmecCommandline(path_to_executable_animec)
callANIMEC_function = callANIMECObject.call_vmec

mmax_sensitivity=3 # Maximum poloidal mode number for optimization
nmax_sensitivity=3 # Maximum toroidal mode number for optimization

exp_weight = 0.1 # Exponential weight for proximity, radius, and curvature
delta_pres = 1e-1 # Scale of pressure perturbations for ANIMEC
min_curvature = 0.5 # Minimum curvature radius for proximity
weight_axis = lambda s : np.exp(-s**2/0.1**2) # Weight function for axis ripple
weight_iota = lambda s : 1
iota_target = [0.17004629, 0.16994813, 0.16990282, 0.16990766, 0.1699433,  0.16999502,
 0.17005301, 0.17011665, 0.17018178, 0.17024943, 0.1703172,  0.17038638,
 0.17045535, 0.17052527, 0.17059468, 0.17066466, 0.17073403, 0.17080374,
 0.17087277, 0.17094198, 0.17101046, 0.17107898, 0.17114669, 0.17121431,
 0.17128106, 0.17134757, 0.1714131,  0.17147823, 0.1715423,  0.17160587,
 0.17166833, 0.17173018, 0.17179085, 0.17185077, 0.17190942, 0.1719672,
 0.1720236,  0.17207902, 0.17213295, 0.17218575, 0.17223695, 0.17228689,
 0.17233512, 0.17238194, 0.17242693, 0.17247036, 0.17251184, 0.17255162,
 0.17258932, 0.17262516, 0.1726588,  0.17269039, 0.17271963, 0.17274666,
 0.17277118, 0.1727933,  0.17281272, 0.17282956, 0.17284352, 0.17285467,
 0.17286275, 0.17286779, 0.17286953, 0.17286797, 0.17286286, 0.17285419,
 0.17284169, 0.17282532, 0.17280483, 0.17278017, 0.17275102, 0.17271731,
 0.17267875, 0.17263524, 0.17258645, 0.17253224, 0.17247226, 0.17240636,
 0.17233413, 0.17225539, 0.17216969, 0.17207682, 0.17197624, 0.17186769,
 0.17175059, 0.17162462, 0.17148912, 0.17134367, 0.17118753, 0.17102011,
 0.17084053, 0.17064814, 0.17044194, 0.17022089, 0.16998375, 0.16972962,
 0.16945687, 0.16916409] # Initial rotational transform profile 

vmecOptimizationObject = VmecOptimization(input_filename,callVMEC_function=callVMEC_function,
                                          callANIMEC_function=callANIMEC_function,
                                          mmax_sensitivity=mmax_sensitivity,
                                          nmax_sensitivity=nmax_sensitivity,delta_pres=delta_pres)

rippleObjective = lambda boundary : vmecOptimizationObject.vmec_objective(boundary=boundary,which_objective='axis_ripple',
                                         weight_function=weight_axis)

rippleObjectiveGrad = lambda boundary : vmecOptimizationObject.vmec_objective_grad(boundary=boundary,which_objective='axis_ripple',
                                        weight_function=weight_axis)

iotaObjective = lambda boundary : vmecOptimizationObject.vmec_objective(boundary=boundary,which_objective='iota_target',
                                         weight_function=weight_iota,iota_target=iota_target)

iotaObjectiveGrad = lambda boundary : vmecOptimizationObject.vmec_objective_grad(boundary=boundary,which_objective='iota_target',
                                         weight_function=weight_iota,iota_target=iota_target)

def proximityObjective(boundary):
    exp = vmecOptimizationObject.input_objective(boundary,
          which_objective='summed_proximity',exp_weight=exp_weight,
                                            min_curvature_radius=min_curvature)
    return exp

def proximityObjectiveGrad(boundary):
    expGrad = \
        vmecOptimizationObject.input_objective_grad(boundary,
            which_objective='summed_proximity',exp_weight=exp_weight,
                                            min_curvature_radius=min_curvature)
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
          which_objective='curvature',max_curvature=10,exp_weight=1)
    return exp

def curvatureObjectiveGrad(boundary):
    expGrad = \
        vmecOptimizationObject.input_objective_grad(boundary,
                which_objective='curvature',max_curvature=10,exp_weight=1)
    return expGrad

def curvatureIntegratedObjective(boundary):
    metric = vmecOptimizationObject.input_objective(boundary,
          which_objective='curvature_integrated',which_kappa=2)
    return metric

def curvatureIntegratedObjectiveGrad(boundary):
    metricGrad = \
        vmecOptimizationObject.input_objective_grad(boundary,\
                which_objective='curvature_integrated',which_kappa=2)
    return metricGrad

boundary = np.copy(vmecOptimizationObject.boundary_opt)

gradOptimizer = GradOptimizer(nparameters=len(boundary))
gradOptimizer.add_objective(rippleObjective,rippleObjectiveGrad,1)
gradOptimizer.add_objective(iotaObjective,iotaObjectiveGrad,1)
gradOptimizer.add_objective(proximityObjective,proximityObjectiveGrad,1e-1)
gradOptimizer.add_objective(radiusObjective,radiusObjectiveGrad,1e3)
gradOptimizer.add_objective(curvatureObjective,curvatureObjectiveGrad,5e-2)
gradOptimizer.add_objective(curvatureIntegratedObjective,\
                            curvatureIntegratedObjectiveGrad,1e-3)

[xopt,fopt,result] = gradOptimizer.optimize(boundary,package='scipy',\
                       method='BFGS',tol=1e-8,options={'disp':True,'gtol':1e-8})

np.savetxt('xopt.txt',xopt)
np.savetxt('fopt.txt',[fopt])
np.savetxt('result.txt',[result])