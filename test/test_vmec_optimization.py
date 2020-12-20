import unittest
import sys
import numpy as np
import numbers
import os
import shutil

sys.path.append('../')

from vmec_input import VmecInput
from vmec_optimization import VmecOptimization, axis_weight
from finite_differences import finite_difference_derivative_random, \
    finite_difference_derivative

from call_vmec import CallVmecBatch, CallVmecCommandline, CallVmecInterface

class Test(unittest.TestCase):
    
    input_filename = 'input.rotating_ellipse'
    input_filename_intersecting = 'input.intersecting'
    input_filename_highres = 'input.rotating_ellipse_highres'
    path_to_executable_animec = os.environ.get('ANIMEC_PATH')
    path_to_executable = os.environ.get('VMEC_PATH')    
    callVmecCommandline = CallVmecCommandline(path_to_executable,nproc=1)
    callVmecFunction = callVmecCommandline.call_vmec
    callAnimecCommandline = CallVmecCommandline(path_to_executable_animec,
                                                nproc=1)
    callAnimecFunction = callVmecCommandline.call_vmec
    directory = os.getcwd()
    
    def set_up(self):
        """
        """
        if self.path_to_executable is None:
            raise RuntimeError('''Unable to run vmec_optimization tests because
                               VMEC_PATH is undefined''')
        try:
            os.mkdir('optimization_test')
        except FileExistsError:
            pass
        except: 
            raise RuntimeError('''Error encountered in making optimization_test
                                  directory.''')
        shutil.copy(self.input_filename,'optimization_test')
        shutil.copy(self.input_filename_intersecting,'optimization_test')
        shutil.copy(self.input_filename_highres,'optimization_test')

        os.chdir('optimization_test')
        self.vmecOptimization = VmecOptimization(self.input_filename,
              self.callVmecFunction,callANIMEC_function=self.callAnimecFunction,
                            mmax_sensitivity=1,nmax_sensitivity=0,verbose=False)
        
    def tear_down(self):
        """
        """
        os.chdir(self.directory)
        if (os.path.isfile('optimization_test')):
            shutil.rmtree('optimization_test',ignore_errors=True)

#     def setUp(self):
#         """
#         """
#         if self.path_to_executable is None:
#             raise RuntimeError('''Unable to run vmec_optimization tests because
#                                VMEC_PATH is undefined''')
#         try:
#             os.mkdir('optimization_test')
#         except FileExistsError:
#             pass
#         except: 
#             raise RuntimeError('''Error encountered in making optimization_test
#                                   directory.''')
#         shutil.copy(self.input_filename,'optimization_test')
#         os.chdir('optimization_test')
#         self.vmecOptimization = VmecOptimization(self.input_filename,
#               self.callVmecFunction,callANIMEC_function=self.callAnimecFunction,
#                             mmax_sensitivity=1,nmax_sensitivity=0,verbose=False)
    
#     def tearDown(self):
#         """
#         """
#         os.chdir(self.directory)
#         if (os.path.isfile('optimization_test')):
#             shutil.rmtree('optimization_test',ignore_errors=True)
    
    def test_fail(self):
        """
        Return to test directory and remove vmec_test if test failed
        """
        os.chdir(self.directory)
        if (os.path.isfile('optimization_test')):
            shutil.rmtree('optimization_test',ignore_errors=True)
    
    def test_evaluate_vmec(self):
        """
        Call with boundary specified and update=True. Check that boundary is 
            updated. 
        Call with boundary specified and update=False. Check that boundary is 
            not updated. 
        Call with updated pressure profile. Check that VmecOutput instance
            has desired profile.
        Call with updated current profile. Check that VmecOutput instance
            has desired current.
        """
        self.set_up()
        boundary = self.vmecOptimization.boundary_opt
        boundary[0] = 1.1*boundary[0]
        exit_code, vmecOutput_new = \
            self.vmecOptimization.evaluate_vmec(boundary=boundary)
        self.assertEqual(exit_code,0) 
        self.assertTrue(np.array_equal(boundary,
                                       self.vmecOptimization.boundary_opt))
        self.assertEqual(boundary[0],vmecOutput_new.rmnc[-1,0])
        
        boundary_new = np.copy(boundary)
        boundary_new[1] = 1.1*boundary_new[1]
        exit_code, vmecOutput_new = \
            self.vmecOptimization.evaluate_vmec(boundary=boundary_new,
                                                update=False)
        self.assertEqual(exit_code,0) 
        self.assertTrue(np.array_equal(boundary,
                                       self.vmecOptimization.boundary_opt))
        self.assertEqual(boundary_new[0],vmecOutput_new.rmnc[-1,0])

        pres = vmecOutput_new.pres+1
        exit_code, vmecOutput_new = \
            self.vmecOptimization.evaluate_vmec(pres=pres)
        self.assertEqual(exit_code,0) 
        self.assertTrue(np.allclose(pres,vmecOutput_new.pres))
        
        It = vmecOutput_new.current()+1
        exit_code, vmecOutput_new = \
            self.vmecOptimization.evaluate_vmec(It=It)
        self.assertEqual(exit_code,0) 
        self.assertTrue(np.allclose(It,vmecOutput_new.current()))
        self.tear_down()
        
    def test_evaluate_animec(self):
        """
        Evaluate with new boundary and check that VmecOutput instance has 
            correct boundary harmonics
        """
        if self.path_to_executable_animec is None:
            raise RuntimeError('''Unable to run vmec_optimization tests 
                           involving ANIMEC because ANIMEC_PATH is undefined''')
            
        self.set_up()
        boundary = self.vmecOptimization.boundary_opt
        exit_code, vmecOutput_new = \
            self.vmecOptimization.evaluate_animec(boundary=boundary)
        self.assertEqual(exit_code,0) 
        self.assertTrue(np.array_equal(boundary,
                                       self.vmecOptimization.boundary_opt))
        self.assertEqual(boundary[0],vmecOutput_new.rmnc[-1,0])
        self.tear_down()
 
    def test_update_boundary_opt(self):
        """
        Check that ValueError raised if incorrect boundary_opt length is passed.
        Check that first element is updated in self.boundary_opt
        """
        self.set_up()
        boundary_opt_new = np.hstack((self.vmecOptimization.boundary_opt,
                                     self.vmecOptimization.boundary_opt))
        self.assertRaises(ValueError,self.vmecOptimization.update_boundary_opt,
                         boundary_opt_new)
        boundary_opt_new = np.copy(self.vmecOptimization.boundary_opt)
        boundary_opt_new[0] = 1
        self.vmecOptimization.update_boundary_opt(boundary_opt_new)
        self.assertEqual(1,self.vmecOptimization.boundary_opt[0])
        self.tear_down()
        
    def test_input_objective(self):
        """
        Check that ValueError is raised if incorrect which_objective is passed
        Check that ValueError is raised if boundary is incorrect length
        Check that volume matches that computed from output file
        Check that inputObject is updated if update=True
        Check that inputObject is not updated if update=True
        """
        self.set_up()
        self.assertRaises(ValueError,
                          self.vmecOptimization.input_objective,
                          which_objective='volumee')
        boundary = np.copy(self.vmecOptimization.boundary_opt)
        boundary_new = np.hstack((boundary,boundary))
        self.assertRaises(ValueError,
                         self.vmecOptimization.input_objective,
                         boundary=boundary_new)
        volume1 = self.vmecOptimization.input_objective(
                which_objective='volume')
        volume2 = self.vmecOptimization.vmec_objective(
                which_objective='volume')        
        self.assertAlmostEqual(volume1,volume2)
        
        boundary_new = np.copy(self.vmecOptimization.boundary_opt)
        boundary_new[0] = 1.1*boundary_new[0]
        volume1 = self.vmecOptimization.input_objective(
                which_objective='volume',boundary=boundary_new)
        self.assertAlmostEqual(self.vmecOptimization.boundary_opt[0],
                              boundary_new[0])
        boundary_old = np.copy(self.vmecOptimization.boundary_opt)
        boundary_new = np.copy(self.vmecOptimization.boundary_opt)
        boundary_new[0] = 1.1*boundary_new[0]
        volume1 = self.vmecOptimization.input_objective(
                which_objective='volume',boundary=boundary_new,update=False)
        self.assertAlmostEqual(self.vmecOptimization.boundary_opt[0],
                              boundary_old[0])
        self.tear_down()

    def test_vmec_objective(self):
        """
        Check that ValueError is raised if incorrect which_objective is passed
        Check that ValueError is raised if boundary is incorrect length
        Check that outputObject is updated if update=True
        Check that outputObject is not updated if update=False
        """
        self.set_up()
        self.assertRaises(ValueError,
                          self.vmecOptimization.vmec_objective,
                          which_objective='volumee')
        boundary = np.copy(self.vmecOptimization.boundary_opt)
        boundary_new = np.hstack((boundary,boundary))
        self.assertRaises(ValueError,
                         self.vmecOptimization.vmec_objective,
                         boundary=boundary_new)

        boundary_new = np.copy(self.vmecOptimization.boundary_opt)
        boundary_new[0] = 1.1*boundary_new[0]
        volume1 = self.vmecOptimization.vmec_objective(
                which_objective='volume',boundary=boundary_new)
        self.assertAlmostEqual(self.vmecOptimization.boundary_opt[0],
                              boundary_new[0])
        boundary_old = np.copy(self.vmecOptimization.boundary_opt)
        boundary_new = np.copy(self.vmecOptimization.boundary_opt)
        boundary_new[0] = 1.1*boundary_new[0]
        volume1 = self.vmecOptimization.vmec_objective(
                which_objective='volume',boundary=boundary_new,update=False)
        self.assertAlmostEqual(self.vmecOptimization.boundary_opt[0],
                              boundary_old[0])
        self.tear_down()
        
    def test_vmec_shape_gradient(self):
        """
        Check that ValueError is raised if incorrect which_objective is passed
        """        
        self.set_up()
        self.assertRaises(ValueError,
                          self.vmecOptimization.vmec_shape_gradient,
                          which_objective='volumee')
        self.tear_down()
        
    def test_input_objective_grad(self):
        """
        Check that ValueError is raised if incorrect which_objective is passed
        Check that ValueError is raised if boundary is incorrect length
        Perform finite-difference testing for each objective
        """
        self.set_up()
        self.assertRaises(ValueError,
                          self.vmecOptimization.input_objective_grad,
                          which_objective='volumee')
        
        boundary = np.copy(self.vmecOptimization.boundary_opt)
        boundary_new = np.hstack((boundary,boundary))
        self.assertRaises(ValueError,
                         self.vmecOptimization.input_objective_grad,
                         boundary=boundary_new)
        # Jacobian
        dfdomega = self.vmecOptimization.input_objective_grad(
                            boundary=boundary,which_objective='jacobian')
        boundary = self.vmecOptimization.boundary_opt
        dfdomega_fd = finite_difference_derivative(boundary, lambda boundary :
                        self.vmecOptimization.input_objective(boundary=boundary,
                            which_objective='jacobian'),epsilon=1e-4,
                                method='centered')

        self.assertTrue(np.allclose(dfdomega,dfdomega_fd,atol=1e-4))
        # Radius
        dfdomega = self.vmecOptimization.input_objective_grad(
                            boundary=boundary,which_objective='radius')
        boundary = self.vmecOptimization.boundary_opt
        dfdomega_fd = finite_difference_derivative(boundary, lambda boundary :
                        self.vmecOptimization.input_objective(boundary=boundary,
                            which_objective='radius'),epsilon=1e-4,
                                method='centered')
        self.assertTrue(np.allclose(dfdomega,dfdomega_fd,atol=1e-5))
        # normalized_jacobian
        dfdomega = self.vmecOptimization.input_objective_grad(
                            boundary=boundary,which_objective=
                            'normalized_jacobian')
        boundary = self.vmecOptimization.boundary_opt
        dfdomega_fd = finite_difference_derivative(boundary, lambda boundary :
                        self.vmecOptimization.input_objective(boundary=boundary,
                            which_objective='normalized_jacobian'),epsilon=1e-4,
                                method='centered')
        self.assertTrue(np.allclose(dfdomega,dfdomega_fd,atol=1e-5))
        # area
        dfdomega = self.vmecOptimization.input_objective_grad(
                            boundary=boundary,which_objective='area')
        boundary = self.vmecOptimization.boundary_opt
        dfdomega_fd = finite_difference_derivative(boundary, lambda boundary :
                        self.vmecOptimization.input_objective(boundary=boundary,
                            which_objective='area'),epsilon=1e-4,
                                method='centered')
        self.assertTrue(np.allclose(dfdomega,dfdomega_fd,atol=1e-5))
        # volume
        dfdomega = self.vmecOptimization.input_objective_grad(
                            boundary=boundary,which_objective='volume')
        boundary = self.vmecOptimization.boundary_opt
        dfdomega_fd = finite_difference_derivative(boundary, lambda boundary :
                        self.vmecOptimization.input_objective(boundary=boundary,
                            which_objective='volume'),epsilon=1e-4,
                                method='centered')
        self.assertTrue(np.allclose(dfdomega,dfdomega_fd,atol=1e-5))
        # summed_proximity
        dfdomega = self.vmecOptimization.input_objective_grad(
                            boundary=boundary,which_objective='summed_proximity')
        boundary = self.vmecOptimization.boundary_opt
        dfdomega_fd = finite_difference_derivative(boundary, lambda boundary :
                        self.vmecOptimization.input_objective(boundary=boundary,
                            which_objective='summed_proximity'),epsilon=1e-4,
                                method='centered')
        self.assertTrue(np.allclose(dfdomega,dfdomega_fd,atol=1e-5))
        # summed_radius
        dfdomega = self.vmecOptimization.input_objective_grad(
                            boundary=boundary,which_objective='summed_radius')
        boundary = self.vmecOptimization.boundary_opt
        dfdomega_fd = finite_difference_derivative(boundary, lambda boundary :
                        self.vmecOptimization.input_objective(boundary=boundary,
                            which_objective='summed_radius'),epsilon=1e-4,
                                method='centered')
        self.assertTrue(np.allclose(dfdomega,dfdomega_fd,atol=1e-5))
        # curvature
        dfdomega = self.vmecOptimization.input_objective_grad(
                            boundary=boundary,which_objective='curvature')
        boundary = self.vmecOptimization.boundary_opt
        dfdomega_fd = finite_difference_derivative(boundary, lambda boundary :
                        self.vmecOptimization.input_objective(boundary=boundary,
                            which_objective='curvature'),epsilon=1e-4,
                                method='centered')
        self.assertTrue(np.allclose(dfdomega,dfdomega_fd,atol=1e-5))
        # curvature_integrated
        dfdomega = self.vmecOptimization.input_objective_grad(
                            boundary=boundary,which_objective='curvature_integrated')
        boundary = self.vmecOptimization.boundary_opt
        dfdomega_fd = finite_difference_derivative(boundary, lambda boundary :
                        self.vmecOptimization.input_objective(boundary=boundary,
                            which_objective='curvature_integrated'),epsilon=1e-4,
                                method='centered')
        self.assertTrue(np.allclose(dfdomega,dfdomega_fd,atol=1e-5))
        self.tear_down()

    def test_vmec_objective_grad(self):
        """
        Check that ValueError is raised if incorrect which_objective is passed
        Check that ValueError is raised if boundary is incorrect length
        Check that RuntimeError is raised if weight functions not prescribed
        Perform finite-difference testing for each objective
        """       
        self.set_up()
        self.assertRaises(ValueError,
                          self.vmecOptimization.vmec_objective_grad,
                          which_objective='volumee')
        
        boundary = np.copy(self.vmecOptimization.boundary_opt)
        boundary_new = np.hstack((boundary,boundary))
        self.assertRaises(ValueError,
                         self.vmecOptimization.vmec_objective_grad,
                         boundary=boundary_new)
        self.assertRaises(RuntimeError,
                          self.vmecOptimization.vmec_objective_grad,
                          which_objective='iota')        
        self.assertRaises(RuntimeError,
                          self.vmecOptimization.vmec_objective_grad,
                          which_objective='iota_prime')       
        self.assertRaises(RuntimeError,
                          self.vmecOptimization.vmec_objective_grad,
                          which_objective='iota_target') 
        self.assertRaises(RuntimeError,
                          self.vmecOptimization.vmec_objective_grad,
                          which_objective='well_ratio') 
        self.assertRaises(RuntimeError,
                          self.vmecOptimization.vmec_objective_grad,
                          which_objective='well') 
        self.assertRaises(RuntimeError,
                          self.vmecOptimization.vmec_objective_grad,
                          which_objective='axis_ripple')         
        # Jacobian
        dfdomega = self.vmecOptimization.vmec_objective_grad(
                            boundary=boundary,which_objective='jacobian')
        boundary = self.vmecOptimization.boundary_opt
        dfdomega_fd = finite_difference_derivative(boundary, lambda boundary :
                        self.vmecOptimization.vmec_objective(boundary=boundary,
                            which_objective='jacobian'),epsilon=1e-4,
                                method='centered')

        self.assertTrue(np.allclose(dfdomega,dfdomega_fd,atol=1e-5))
        # Radius
        dfdomega = self.vmecOptimization.vmec_objective_grad(
                            boundary=boundary,which_objective='radius')
        boundary = self.vmecOptimization.boundary_opt
        dfdomega_fd = finite_difference_derivative(boundary, lambda boundary :
                        self.vmecOptimization.vmec_objective(boundary=boundary,
                            which_objective='radius'),epsilon=1e-4,
                                method='centered')
        self.assertTrue(np.allclose(dfdomega,dfdomega_fd,atol=1e-5))
        # normalized_jacobian
        dfdomega = self.vmecOptimization.vmec_objective_grad(
                            boundary=boundary,which_objective=
                            'normalized_jacobian')
        boundary = self.vmecOptimization.boundary_opt
        dfdomega_fd = finite_difference_derivative(boundary, lambda boundary :
                        self.vmecOptimization.vmec_objective(boundary=boundary,
                            which_objective='normalized_jacobian'),epsilon=1e-4,
                                method='centered')
        self.assertTrue(np.allclose(dfdomega,dfdomega_fd,atol=1e-5))
        # iota
        inputObject = VmecInput('input.rotating_ellipse_highres')
        boundary = np.copy(self.vmecOptimization.boundary_opt)
        boundary[0] = 1.1*boundary[0]
        self.vmecOptimization.inputObject = inputObject
        self.vmecOptimization.vmecInputFilename = 'input.rotating_ellipse_highres'
        self.vmecOptimization.delta_curr = 10
        dfdomega = self.vmecOptimization.vmec_objective_grad(
                        boundary=boundary,which_objective='iota',
                        weight_function = axis_weight)
        dfdomega_fd = finite_difference_derivative(boundary, lambda boundary :
                        self.vmecOptimization.vmec_objective(boundary=boundary,
                        which_objective='iota', weight_function = axis_weight),
                        epsilon=1e-3, method='centered')
        self.assertTrue(np.allclose(dfdomega,dfdomega_fd,atol=1e-2))
        
        # To do : finish FD testing
        self.tear_down()
        
    def test_test_jacobian(self):
        """
        Test that False is returned if rmnc/zmns set to 0
        """
        self.set_up()
        inputObject = self.vmecOptimization.vmecInputObject
        rbc = np.copy(inputObject.rbc)
        zbs = np.copy(inputObject.zbs)
        inputObject.rbc = 0*inputObject.rbc
        inputObject.zbs = 0*inputObject.zbs
        orientable = self.vmecOptimization.test_jacobian(inputObject)
        self.assertFalse(orientable)
        # Reset boundary
        inputObject.rbc = rbc
        inputObject.zbs = zbs
        self.tear_down()
        
    def test_call_vmec(self):
        """
        Check that error is returned if self-intersecting boundary is provided
        Check that error is returned if jacobian ill-conditioned
        Check that no error is returned with sample input file
        """
        self.set_up()
        inputObject = self.vmecOptimization.vmecInputObject
        inputObject_new = VmecInput(self.input_filename_intersecting)
        error_code = self.vmecOptimization.call_vmec(inputObject_new)
        self.assertEqual(error_code,16)
        
        # To do : add test for ill-conditioned Jacobian
#         rbc = np.copy(inputObject.rbc)
#         zbs = np.copy(inputObject.zbs)
#         inputObject.rbc[1::] = 1e-4*inputObject.rbc[1::]
#         inputObject.zbs = inputObject.zbs
#         error_code = self.vmecOptimization.call_vmec(inputObject)
#         self.assertEqual(error_code,16)
        # Reset boundary
#         inputObject.rbc = rbc
#         inputObject.zbs = zbs
        
        inputObject = VmecInput(self.input_filename)
        error_code = self.vmecOptimization.call_vmec(inputObject)
        self.assertEqual(error_code,0)
        self.tear_down()
        
    def test_parameter_derivatives(self):
        """
        Check that ValueError is raised if shape_gradient is of incorrect shape
        """
        self.set_up()
        shape_gradient = np.zeros((self.vmecOptimization.nzeta+1,
                                 self.vmecOptimization.ntheta))       
        self.assertRaises(ValueError,
                          self.vmecOptimization.vmec_shape_gradient,
                          shape_gradient,self.vmecOptimization.vmecOutputObject)
        self.tear_down()

    


        

       



        
        
        


        

        
        
        
        


        





       


       

        
        
        
        
        

       

       




        
        
