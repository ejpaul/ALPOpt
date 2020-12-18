import subprocess
import sys
import os
import numpy as np
from scipy.io import netcdf
try:
    from vmec_class import VMEC
    from mpi4py import MPI
except:
    raise RuntimeError('Unable to load modules needed for interface.')

class CallVmecBatch:
    """
    This class is used to call VMEC by submitting a job with a batch script
    """
    def __init__(self,sample_input_filename,sample_batch):
        """
        Args: 
            sample_input_filename : sample VMEC input filename
            sample_batch : sample batch file. Filename will be modified when
                submitted
        """
        self.sample_input_filename = sample_input_filename
        self.sample_batch = sample_batch
        self.directory = os.getcwd()
    
    def call_vmec(self,inputObject):
        """
        Calls VMEC by submitting batch script with subprocess call with -W 
            (waits until job is completed before continuing execution)
        
        Args:
            inputObject : instance of VmecInput
        Returns:
            exit_code : returncode from subprocess call. 0 indicates success.
        """
        batch = self.update_batch(inputObject.input_filename)
        completedProcess = subprocess.run(['sbatch','-W',batch],capture_output=True)
        exit_code = completedProcess.returncode
        return exit_code
    
    def update_batch(self,input_filename):
        """
        Update sample_batch file with new input_filename
        
        Args:
            input_filename (string) : name of VMEC file
        """
        if (os.getcwd()==self.directory):
            if (input_filename==self.sample_input_filename):
                return self.sample_batch
            else:
                batch = self.sample_batch+'_modified'
        else:
            batch = self.sample_batch
        f1 = open(self.directory+'/'+self.sample_batch,'r')
        f2 = open(batch,'w')
        contains = False
        for line in f1:
            if (line.find(self.sample_input_filename) > -1):
                contains = True
            f2.write(line.replace(self.sample_input_filename,input_filename))
        f1.close()
        f2.close() 
        if (contains == False):
            raise RuntimeError('''Error! update_batch did not find  
                            self.sample_input_filename in self.sample_batch''')
        return batch
  
class CallVmecCommandline:
    """
    This class is used to call VMEC using a subprocess call. The call can be made
        with mpiexec if requested. 
    """
    def __init__(self,path_to_executable,nproc=1):
        """
        Args: 
            path_to_executable (string) : absolute or relative to path to 
                VMEC executable
            nproc (integer) : number of processes for execution
        """
        self.path_to_executable = path_to_executable
        self.nproc = nproc
    
    def call_vmec(self,inputObject):
        """
        Call VMEC using specified input object
        
        Args: 
            inputObject : instance of VmecInput
            
        Returns: 
            exit_code : code returned by subprocess call. Nonzero indicates an
                error. 
        """
        input_filename = inputObject.input_filename
        with open('out.txt','w+') as fout:
            with open('err.txt','w+') as ferr:
                if (self.nproc==1):
                    exit_code=subprocess.call([self.path_to_executable,input_filename],\
                                stdout=fout,stderr=ferr)
                else: 
                    mpicmd = 'mpiexec -n '+str(self.nproc)+' '+self.path_to_executable \
                    +' '+input_filename
                    proc=subprocess.Popen(mpicmd,\
                                stdout=fout,stderr=ferr,shell=True)
                    # Wait for process to complete
                    proc.wait()
                    res = proc.communicate()
                    exit_code = proc.returncode
                    
        return exit_code
  
class CallVmecInterface:
    """
    This class is used to call VMEC using the python wrapper provided in 
        simsopt (https://github.com/hiddenSymmetries/simsopt)
    """
    def __init__(self,input_filename='',verbose=False):
        """
        Args: 
            input_filename : VMEC input filename for initialization
            verbose (boolean) : verbosity of output 
        """
        self.input_filename = input_filename
        self.verbose = verbose
        self.vmecObject = VMEC(input_file=input_filename,verbose=self.verbose,\
                               comm=MPI.COMM_WORLD.py2f())
    
    def call_vmec(self,inputObject):
        """
        Call VMEC using specified input object

        Args: 
            inputObject : instance of VmecInput
            
        Returns: 
            status : code returned by VMEC call. Nonzero indicates an
                error. 
        """
        input_filename = inputObject.input_filename

        rank = MPI.COMM_WORLD.Get_rank()
        # Set desired input data
        raxis_cc = np.zeros(102)
        zaxis_cs = np.zeros(102)
        raxis_cc[0:len([inputObject.raxis])] = np.copy(inputObject.raxis)
        zaxis_cs[0:len([inputObject.zaxis])] = np.copy(inputObject.zaxis)
        self.vmecObject.indata.raxis_cc = np.copy(raxis_cc)
        self.vmecObject.indata.zaxis_cs = np.copy(zaxis_cs)
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
                rbc[101+int(inputObject.xn[imn]),int(inputObject.xm[imn])] \
                    = inputObject.rbc[imn]
            if (inputObject.zbs[imn]!=0):
                zbs[101+int(inputObject.xn[imn]),int(inputObject.xm[imn])] \
                    = inputObject.zbs[imn]
        self.vmecObject.indata.rbc = np.copy(rbc)
        self.vmecObject.indata.zbs = np.copy(zbs)
        self.vmecObject.reinit()
	
        if rank == 0:
            verbose = self.verbose
        else:
            verbose = False
        self.vmecObject.run(iseq=rank,input_file=input_filename)
        MPI.COMM_WORLD.Barrier()
        self.vmecObject.load()
        if (rank == 0):
            status = self.vmecObject.ictrl[1]
        else:
            status = 0
        status = MPI.COMM_WORLD.bcast(status,root=0)
        
        # Now reset the profiles
        if inputObject.ac_aux_f is not None:
            self.vmecObject.indata.ac_aux_f[0:len(inputObject.ac_aux_f)] = 0
        if inputObject.am_aux_f is not None:
            self.vmecObject.indata.am_aux_f[0:len(inputObject.am_aux_f)] = 0
            
        # status = 11 also indicates success
        if (status == 11):
            status = 0
        return status
    

    
