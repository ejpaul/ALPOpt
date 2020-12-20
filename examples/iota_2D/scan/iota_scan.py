"""
This script was used to perform the optimization in 
Paul, Landreman, Antonsen Appendix A

This example uses the VMEC interface included in simsopt. Make sure that
    simsopt.modules.vmec is added to your path
"""
import sys
import numpy as np

sys.path.append('../../../')
from vmec_optimization import VmecOptimization
from call_vmec import CallVmecInterface

input_filename = 'input.rotating_ellipse'
callVMECObject = CallVmecInterface(input_filename)
callVMEC_function = callVMECObject.call_vmec

iota_target = 0.618034 # target iota value

def weight_function(s):
    return 1

xm_sensitivity = np.array([1])
xn_sensitivity = np.array([1])

optimizationObject = VmecOptimization(input_filename,
                        callVMEC_function=callVMEC_function,
                        xm_sensitivity=xm_sensitivity,
                        xn_sensitivity=xn_sensitivity)

iotaObjective = lambda boundary : optimizationObject.vmec_objective(boundary=boundary,
                                                    which_objective='iota_target',
                                                    weight_function=weight_function,
                                                    iota_target=iota_target)

iotaObjectiveGrad = lambda boundary : optimizationObject.evaluate_vmec_objective_grad(boundary=boundary,which_objective='iota_target',
                                                    weight_function=weight_function,
                                                    iota_target=iota_target)

boundary = optimizationObject.boundary_opt

rbc = np.linspace(0.4,0.7,20)
zbs = np.linspace(-0.7,-0.4,20)
[rbc_mesh,zbs_mesh] = np.meshgrid(rbc,zbs)

iota_mesh = np.zeros(np.shape(rbc_mesh))
for im in range(len(rbc_mesh[:,0])):
    for il in range(len(rbc_mesh[0,:])):
        iota_objective = iotaObjective([rbc_mesh[im,il],zbs_mesh[im,il]])
        iota_mesh[im,il] = iota_objective
np.savetxt('iota_mesh.txt',iota_mesh)
