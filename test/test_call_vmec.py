import unittest
import sys
import numpy as np
import numbers
import scipy.optimize
import os
from simsopt.modules.vmec.input import VmecInput
import shutil

sys.path.append('../')

from call_vmec import CallVmecBatch, CallVmecCommandline, CallVmecInterface

class Test(unittest.TestCase):
    
    input_filename = 'input.rotating_ellipse'
    batch = 'batch'
    directory = os.getcwd()
    
    path_to_executable = os.environ.get('VMEC_PATH')
    inputObject = VmecInput(input_filename)
    nproc = 1
    
    callVmecBatch = CallVmecBatch(input_filename,batch)
    callVmecCommandline = CallVmecCommandline(path_to_executable,nproc=1)
    
    def test_fail(self):
        """
        Return to test directory and remove vmec_test if test failed
        """
        os.chdir(self.directory)
        if (os.path.isfile('vmec_test')):
            shutil.rmtree('vmec_test',ignore_errors=True)
    
    def setup(self,directory):
        """
        Setup for test by creating new directory and copying files
        """
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass
        except: 
            raise RuntimeError('Error encountered in making '+directory+
                               ' directory.')
        shutil.copy(self.input_filename,directory)
        shutil.copy(self.batch,directory)
        os.chdir(directory)
        
    def reset(self,directory):
        """
        Teardown after test by returning to original directory and removing
        new directory
        """    
        os.chdir('..')
        shutil.rmtree(directory,ignore_errors=True)

    def test_call_vmec_batch(self):
        """
        Test batch call from same directory with sample input_filename
        Test batch call from same directory with modifed input_filename
        Test batch call from new directory with same input_filename
        Test batch call from new directory with modified input_filename
        """          
        self.setup('vmec_test')
        exit_code = self.callVmecBatch.call_vmec(self.inputObject)
        self.assertEqual(exit_code,0)
        
        new_input_filename = self.input_filename+'_modified'
        shutil.copy(self.input_filename, new_input_filename)
        self.inputObject.input_filename = new_input_filename
        exit_code = self.callVmecBatch.call_vmec(self.inputObject)
        self.assertEqual(exit_code,0)
        
        self.setup('new')
        exit_code = self.callVmecBatch.call_vmec(self.inputObject)
        self.assertEqual(exit_code,0) 
        
        shutil.copy(self.input_filename, new_input_filename)
        self.inputObject.input_filename = new_input_filename
        exit_code = self.callVmecBatch.call_vmec(self.inputObject)

        self.reset('new')
        self.reset('vmec_test')

    def test_call_vmec_commandline(self):
        """
        Test call from same directory with sample input_filename
        Test mpi call from same directory with sample input_filename
        Test call from same directory with modifed input_filename
        Test call from new directory with same input_filename
        Test call from new directory with modified input_filename
        """
        if self.path_to_executable is None:
            raise RuntimeError('''Unable to run CallVmecCommandline tests because
                               VMEC_PATH is undefined''')
            
        self.setup('vmec_test')
        exit_code = self.callVmecCommandline.call_vmec(self.inputObject)
        self.assertEqual(exit_code,0)
        
        self.callVmecCommandline.nproc = 2
        exit_code = self.callVmecCommandline.call_vmec(self.inputObject)
        self.assertEqual(exit_code,0)      
        self.callVmecCommandline.nproc = 1
        
        new_input_filename = self.input_filename+'_modified'
        shutil.copy(self.input_filename, new_input_filename)
        self.inputObject.input_filename = new_input_filename
        exit_code = self.callVmecCommandline.call_vmec(self.inputObject)
        self.assertEqual(exit_code,0)
        
        self.setup('new')
        exit_code = self.callVmecCommandline.call_vmec(self.inputObject)
        self.assertEqual(exit_code,0) 
        
        shutil.copy(self.input_filename, new_input_filename)
        self.inputObject.input_filename = new_input_filename
        exit_code = self.callVmecCommandline.call_vmec(self.inputObject)

        self.reset('new')
        self.reset('vmec_test')
        
    def test_call_vmec_interface(self):
        """
        Test call from same directory with sample input_filename
        Test call from same directory with modifed input_filename
        Test call from new directory with same input_filename
        Test call from new directory with modified input_filename
        """
        try:
            from simsopt.modules.vmec.core import VMEC
            from mpi4py import MPI
        except:
            raise RuntimeError('Unable to load modules needed for interface.')
        callVmecInterface = CallVmecInterface(input_filename=self.input_filename)
            
        self.setup('vmec_test')
        exit_code = callVmecInterface.call_vmec(self.inputObject)
        self.assertEqual(exit_code,0)
        
        new_input_filename = self.input_filename+'_modified'
        shutil.copy(self.input_filename, new_input_filename)
        self.inputObject.input_filename = new_input_filename
        exit_code = callVmecInterface.call_vmec(self.inputObject)
        self.assertEqual(exit_code,0)
        
        self.setup('new')
        exit_code = callVmecInterface.call_vmec(self.inputObject)
        self.assertEqual(exit_code,0) 
        
        shutil.copy(self.input_filename, new_input_filename)
        self.inputObject.input_filename = new_input_filename
        exit_code = callVmecInterface.call_vmec(self.inputObject)

        self.reset('new')
        self.reset('vmec_test')