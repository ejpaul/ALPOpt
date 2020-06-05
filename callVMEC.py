import subprocess
import sys
import os
from vmec_class import VMEC

class callVMEC_batch:
  def __init__(self,sample_input_filename,sample_batch):
    self.sample_input_filename = sample_input_filename
    self.sample_batch = sample_batch
    self.directory = os.getcwd()
    
  def callVMEC(self,input_filename):
    self.update_batch(input_filename)
    completedProcess = subprocess.run(['sbatch','-W',self.sample_batch],capture_output=True)
    if (completedProcess.returncode != 0):
      print('Error in submitting batch job! Code = '+str(completedProcess.returncode))
      sys.exit(1)
    
  # update sample_batch with new input_filename
  def update_batch(self,input_filename):
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
    
  def callVMEC(self,input_filename):
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
  def __init__(self,input_filename='',verbose=False,comm=0):
    self.input_filename = input_filename
    self.verbose = verbose
    self.comm = comm
    self.vmecObject = VMEC(input_file=input_filename,verbose=verbose,comm=comm)
    
  def callVMEC(self,input_filename):
    if (input_filename is None):
      inputFilename = self.input_filename
    print('Calling VMEC from ' + os.getcwd())
    self.vmecObject.run(input_file=input_filename)
    exit_code = self.vmecObject.ictrl[1]
    return exit_code
    
    

    
