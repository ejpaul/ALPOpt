import subprocess
import sys
import os
from vmec_class import VMEC
from mpi4py import MPI
import numpy as np
# import vmec

run_modes =  {'all': 63,
              'input': 35, # STELLOPT uses 35; V3FIT uses 7
              'output': 8, 
              'main': 45}

class callVMEC_batch:
  def __init__(self,sample_input_filename,sample_batch):
    self.sample_input_filename = sample_input_filename
    self.sample_batch = sample_batch
    self.directory = os.getcwd()
    
  def callVMEC(self,inputObject):
    self.update_batch(inputObject.input_filename)
    completedProcess = subprocess.run(['sbatch','-W',self.sample_batch],capture_output=True)
    if (completedProcess.returncode != 0):
      print('Error in submitting batch job! Code = '+str(completedProcess.returncode))
      sys.exit(1)
    
  # update sample_batch with new input_filename
  def update_batch(self,input_filename,inputObject):
    f1 = open(self.directory+'/'+self.sample_batch,'r')
    f2 = open(self.sample_batch,'w')
    contains = False
    for line in f1:
      if (line.find(self.sample_input_filename) > -1):
        contains = True
      f2.write(line.replace(self.sample_input_filename,input_filename))
    f1.close()
    f2.close() 
    if (contains == False):
      print('Error! update_batch did not find self.sample_input_filename in self.sample_batch')
      sys.exit(1)
  
class callVMEC_commandline:
  def __init__(self,path_to_executable):
    self.path_to_executable = path_to_executable
    
  def callVMEC(self,inputObject):
    input_filename = inputObject.input_filename
    print('Calling VMEC from '+os.getcwd())
    with open('out.txt','w+') as fout:
      with open('err.txt','w+') as ferr:
        exit_code=subprocess.call([self.path_to_executable,input_filename],\
                            stdout=fout,stderr=ferr)
        fout.seek(0)
        output=fout.read()

        ferr.seek(0) 
        errors = ferr.read()
  
    exit_code = subprocess.call([self.path_to_executable,input_filename],\
                               stdout=subprocess.PIPE)
    if (exit_code != 0):
      print('VMEC returned with an error (exit_code = '+str(exit_code)+')')
      sys.exit(1)
  
class callVMEC_interface:
  def __init__(self,input_filename='',verbose=False):
    self.input_filename = input_filename
    self.verbose = verbose
    self.comm = MPI.COMM_WORLD
    self.vmecObject = VMEC(input_file=input_filename,verbose=self.verbose,comm=self.comm.py2f())
    
  def callVMEC(self,inputObject):
    input_filename = inputObject.input_filename
    
    # Set desired input data
    self.vmecObject.indata.raxis[0:len(inputObject.raxis)] = np.copy(inputObject.raxis)
    self.vmecObject.indata.zaxis[0:len(inputObject.zaxis)] = np.copy(inputObject.zaxis)
    if inputObject.curtor is not None:
      self.vmecObject.indata.curtor = inputObject.curtor
    if inputObject.ac_aux_f is not None:
      self.vmecObject.indata.ac_aux_f[0:len(inputObject.ac_aux_f)] = np.copy(inputObject.ac_aux_f)
    if inputObject.ac_aux_s is not None:
      self.vmecObject.indata.ac_aux_s[0:len(inputObject.ac_aux_s)] = np.copy(inputObject.ac_aux_s)
    if inputObject.pcurr_type is not None:
      self.vmecObject.indata.pcurr_type = inputObject.pcurr_type
    if inputObject.am_aux_f is not None:
      self.vmecObject.indata.am_aux_f[0:len(inputObject.am_aux_f)] = np.copy(inputObject.am_aux_f)
    if inputObject.am_aux_s is not None:
      self.vmecObject.indata.am_aux_s[0:len(inputObject.am_aux_s)] = np.copy(inputObject.am_aux_s)
    if inputObject.pmass_type is not None:
      self.vmecObject.indata.pmass_type = inputObject.pmass_type
    self.vmecObject.indata.mpol = inputObject.mpol
    self.vmecObject.indata.ntor = inputObject.ntor
    self.vmecObject.indata.nfp = inputObject.nfp
    rbc = np.zeros(np.shape(self.vmecObject.indata.rbc))
    zbs = np.zeros(np.shape(self.vmecObject.indata.zbs))
    for imn in range(inputObject.mnmax):
      if (inputObject.rbc[imn]!=0):
        rbc[101+int(inputObject.xn[imn]),int(inputObject.xm[imn])] = inputObject.rbc[imn]
      if (inputObject.zbs[imn]!=0):
        zbs[101+int(inputObject.xn[imn]),int(inputObject.xm[imn])] = inputObject.zbs[imn]
    self.vmecObject.indata.rbc = np.copy(rbc)
    self.vmecObject.indata.zbs = np.copy(zbs)
    self.vmecObject.reinit()
    
    rank = self.comm.Get_rank()
    self.vmecObject.run(iseq=rank,input_file=input_filename)
    return self.vmecObject.ictrl[1]
    
    

    
