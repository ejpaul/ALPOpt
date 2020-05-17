import numpy as np
import os
from vmecOutput import *
import errno
import subprocess
import sys
import shutil
from FortranNamelistTools import *
from scipy import interpolate
import f90nml

class vmecOptimization:
  def __init__(self,vmecInputFilename,mmax_sensitivity,
             nmax_sensitivity,path_to_vmec_executable,name):
    self.mmax_sensitivity = mmax_sensitivity
    self.nmax_sensitivity = nmax_sensitivity
    [mnmax_sensitivity,xm_sensitivity,xn_sensitivity] = \
      init_modes(mmax_sensitivity,nmax_sensitivity)
    self.mnmax_sensitivity = mnmax_sensitivity
    self.xm_sensitivity = xm_sensitivity
    self.xn_sensitivity = xn_sensitivity
    self.path_to_vmec_executable = path_to_vmec_executable
    self.counter = 0 # Counts number of function evals (used for naming directories)
    self.name = name # Name used for directories
    self.vmecInputFilename = vmecInputFilename
    self.directory = os.getcwd()
    
    # Read items from Fortran namelist
    nml = f90nml.read(vmecInputFilename)
    nml = nml.get("indata")
    nfp = nml.get("nfp")
    mpol_input = nml.get("mpol")
    if (mpol_input<2):
      print('Error! mpol must be > 1.')
      sys.exit(1)
      
    ntor_input = nml.get("ntor")
    # max(xm) = mpol-1 in VMEC
    [mnmax_input,xm_input,xn_input] = init_modes(mpol_input-1,ntor_input)
    # Returns rbc and zbs on grid of input modes 
    [rbc_input,zbs_input] = self.read_boundary_input(xn_input,xm_input)
    
    # mpol, ntor for calculaitons must be large enough for sensitivity required
    self.mpol = max(mmax_sensitivity+1,mpol_input)
    self.ntor = max(nmax_sensitivity,ntor_input)
    [self.mnmax,self.xm,self.xn] = init_modes(self.mpol-1,self.ntor)
    [mnmax_sensitivity,xm_sensitivity,xn_sensitivity] = init_modes(mmax_sensitivity,nmax_sensitivity)

    rmnc = np.zeros(self.mnmax)
    zmns = np.zeros(self.mnmax)
    rmnc_opt = np.zeros(mnmax_sensitivity)
    zmns_opt = np.zeros(mnmax_sensitivity)
    for imn in range(mnmax_input):
      cond = np.logical_and((self.xm == xm_input[imn]),\
          (self.xn == xn_input[imn]))
      if (any(cond)):
        rmnc[cond] = rbc_input[imn]
        zmns[cond] = zbs_input[imn]
      cond = np.logical_and((xm_sensitivity == xm_input[imn]),\
          (xn_sensitivity == xn_input[imn]))
      if (any(cond)):
        rmnc_opt[cond] = rbc_input[imn]
        zmns_opt[cond] = zbs_input[imn]
    self.boundary = np.vstack((rmnc,zmns))   
    self.boundary_opt = np.vstack((rmnc_opt,zmns_opt))
    
    self.vmecOutputObject = self.evaluate_vmec() # Current boundary evaluation
    
  # If update = True, new equilibrium evaluation saved in self.vmecOutputObject and boundary is assigned to self.boundary
  def evaluate_vmec(self,boundary=None,It=None,update=True):
    if (os.getcwd()!=self.directory):
      print("Evaluate_vmec called from incorrect directory. Changing to "+self.directory)
      os.chdir(self.directory)
    if ((boundary is not None) and (It is not None)):
      print("evaluate_vmec called with perturbed boundary and current profile. This is not supported.")
      sys.exit(1)
    if (boundary is not None):
      # Update equilibrium count
      self.counter += 1
      # Update boundary
      self.update_boundary_opt(boundary)
      directory_name = self.name+"_"+str(self.counter)
    elif (It is not None):
      directory_name = "delta_curr_"+str(self.counter)
    elif (self.counter == 0):
      directory_name = self.name+"_"+str(self.counter)
    else:
      print('Error! evaluate_vmec called when equilibrium does not need to be evaluated.')
      sys.exit(1)
    # Make directory for VMEC evaluations
    try:
      os.mkdir(directory_name)
    except OSError as exc:
      if exc.errno != errno.EEXIST:
        print('Warning. Directory '+directory_name+' already exists. Contents may be overwritten.')
        raise
      pass

    # Edit input filename with boundary 
    f = open(self.vmecInputFilename,'r')
    os.chdir(directory_name)
    input_file = "input."+directory_name
    f2 = open(input_file,'w')
    input_file_lines = f.readlines()
    for line in input_file_lines:
        if (line.strip()=="/"):
          break
        write_condition = True
        if (boundary is not None):
          write_condition = write_condition and (not namelistLineContains(line,"rbc") and \
              not namelistLineContains(line,"zbs") and \
              not namelistLineContains(line,"mpol") and \
              not namelistLineContains(line,"ntor"))
        if (It is not None):
          write_condition = write_condition and (not namelistLineContains(line,"ac_aux_s") and \
              not namelistLineContains(line,"ac_aux_f") and \
              not namelistLineContains(line,"curtor") and \
              not namelistLineContains(line,"pcurr_type"))
        if (write_condition):
          f2.write(line)
    if (boundary is not None):
      f2.write("mpol = " + str(self.mpol) + "\n")
      f2.write("ntor = " + str(self.ntor) + "\n")
      # Write RBC
      for imn in range(self.mnmax):
        f2.write('RBC('+str(int(self.xn[imn]))+","\
                 +str(int(self.xm[imn]))+") = "\
                 +str(self.boundary[0,imn]))
        f2.write("\n")
      # Write ZBS
      for imn in range(self.mnmax):
        f2.write('ZBS('+str(int(self.xn[imn]))+","\
                 +str(int(self.xm[imn]))+") = "\
                 +str(self.boundary[1,imn]))
        f2.write("\n")
    if (It is not None):
      curtor = 1.5*It[-1] - 0.5*It[-2]
      s_full = np.linspace(0,1,len(It)+1)
      ds = s_full[1]-s_full[0]
      s_half = s_full-0.5*ds
      s_half = np.delete(s_half,0)
      f2.write("curtor = " + str(curtor) + "\n")
      f2.write("ac_aux_f = ")
      f2.writelines(map(lambda x:str(x)+" ",It))
      f2.write("\n")
      f2.write("ac_aux_s = ")
      f2.writelines(map(lambda x:str(x)+" ",s_half))
      f2.write("\n")
      f2.write("pcurr_type = 'line_segment_I'" + "\n")
    f2.write("/ \n")
    f.close()
    f2.close()

    # Call VMEC with revised input file
    exit_code = call_vmec(self.path_to_vmec_executable,input_file)
    # Read from new equilibrium
    outputFileName = "wout_"+input_file[6::]+".nc"

    vmecOutput_new = readVmecOutput(outputFileName)
    if (boundary is not None and update):
      self.vmecOutputObject = vmecOutput_new

    os.chdir("..")
    return vmecOutput_new
    
  # Reads in rbc and zbs from input file
  def read_boundary_input(self,xn,xm):
    nml = f90nml.read(self.vmecInputFilename)
    nml = nml["indata"]
    rbc_array = np.array(nml["rbc"]).T
    zbs_array = np.array(nml["zbs"]).T

    rbc_array[rbc_array==None] = 0
    zbs_array[zbs_array==None] = 0
    [dim1_rbc,dim2_rbc] = np.shape(rbc_array)
    [dim1_zbs,dim2_zbs] = np.shape(zbs_array)
    
    [nmin_rbc,mmin_rbc,nmax_rbc,mmax_rbc] = min_max_indices_2d('rbc',self.vmecInputFilename)
    nmax_rbc = max(nmax_rbc,-nmin_rbc)
    [nmin_zbs,mmin_zbs,nmax_zbs,mmax_zbs] = min_max_indices_2d('zbs',self.vmecInputFilename)
    nmax_zbs = max(nmax_zbs,-nmin_zbs)

    mnmax = len(xm)
    rbc = np.zeros(mnmax)
    zbs = np.zeros(mnmax)

    for imn in range(mnmax):
        if ((xn[imn]-nmin_rbc) < dim1_rbc and (xm[imn]-mmin_rbc) < dim2_rbc \
           and (xn[imn]-nmin_rbc)> -1 and (xm[imn]-mmin_rbc)> -1):
          rbc[imn] = rbc_array[int(xn[imn]-nmin_rbc),int(xm[imn]-mmin_rbc)]
        if ((xn[imn]-nmin_zbs) < dim1_zbs and (xm[imn]-mmin_zbs) < dim2_zbs \
           and (xn[imn]-nmin_rbc)> -1 and (xm[imn]-mmin_rbc)> -1):
          zbs[imn] = zbs_array[int(xn[imn]-nmin_zbs),int(xm[imn]-mmin_zbs)]
    
    return rbc, zbs
    
  # Update boundary_opt and boundary with new boundary coefficients
  def update_boundary_opt(self,boundary_opt_new):
    rmnc_opt_new = boundary_opt_new[0,:]
    zmns_opt_new = boundary_opt_new[1,:]
    self.boundary_opt = boundary_opt_new

    rmnc_new = self.boundary[0,:]
    zmns_new = self.boundary[1,:]
    for imn in range(self.mnmax_sensitivity):
      cond = np.logical_and((self.xm == self.xm_sensitivity[imn]),\
        (self.xn == self.xn_sensitivity[imn]))
      if (any(cond)):
        rmnc_new[cond] = rmnc_opt_new[imn]
        zmns_new[cond] = zmns_opt_new[imn]
    self.boundary = np.vstack((rmnc_new,zmns_new))
    return
  
  # Call VMEC with boundary specified by boundaryObjective to evaluate which_objective
  def evaluate_vmec_objective(self,boundary=None,which_objective=None,weight_function=None,update=True):
    if (which_objective=='iota'):
      print("Evaluating iota objective.")
    else:
      print("Error! evaluate_vmec called with incorrect value of which_objective")
      sys.exit(1)
    if (np.shape(boundary)!=np.shape(self.boundary_opt)):
      print("Error! evaluate_vmec called with incorrect boundary shape.")
      sys.exit(1)
    # Save old boundary if update=False
    if (update==False):
      boundary_old = np.copy(self.boundary_opt)
    # Evaluate new equilibrium if necessary
    if (boundary is not None and (boundary!=self.boundary_opt).any()):
      vmecOutputObject = self.evaluate_vmec(boundary=boundary,update=update)
    else:
      vmecOutputObject = self.vmecOutputObject
    # Reset old boundary 
    if (update==False):
      self.update_boundary_opt(boundary_old)
    
    if (which_objective == 'iota'):
      return vmecOutputObject.evaluate_iota_objective(weight_function)
    if (which_objective == 'well'):
      return vmecOutputObject.evaluate_well_objective(weight_function)

  # If boundary is specified, new equilibrium will be specified
  def vmec_shape_gradient(self,which_objective,weight_function,delta,boundary=None,update=True):
    if (which_objective=='iota'):
      print("Evaluating iota objective shape gradient.")
    else:
      print("Error! vmec_shape_gradient called with incorrect value of"+which_objective)
      sys.exit(1)
      # Evaluate base equilibrium if necessary
    if (boundary is not None and (boundary!=self.boundary_opt).all()):
      vmecOutputObject = self.evaluate_vmec(boundary=boundary,update=update)
    else:
      vmecOutputObject = self.vmecOutputObject
    [Bx, By, Bz, theta_arclength] = vmecOutputObject.B_on_arclength_grid()

    if (which_objective == 'iota'):
      It_half = vmecOutputObject.compute_current()
      It_new = It_half + delta*weight_function(self.vmecOutputObject.s_half)

      vmecOutput_delta = self.evaluate_vmec(It=It_new)

      [Bx_delta, By_delta, Bz_delta, theta_arclength_delta] = vmecOutput_delta.B_on_arclength_grid()
      for izeta in range(self.vmecOutputObject.nzeta):
          f = interpolate.InterpolatedUnivariateSpline(theta_arclength_delta[izeta,:],Bx_delta[izeta,:])
          Bx_delta[izeta,:] = f(theta_arclength[izeta,:])
          f = interpolate.InterpolatedUnivariateSpline(theta_arclength_delta[izeta,:],By_delta[izeta,:])
          By_delta[izeta,:] = f(theta_arclength[izeta,:])
          f = interpolate.InterpolatedUnivariateSpline(theta_arclength_delta[izeta,:],Bz_delta[izeta,:])
          Bz_delta[izeta,:] = f(theta_arclength[izeta,:])

      deltaB_dot_B = (Bx_delta-Bx)*Bx/delta + (By_delta-By)*By/delta + (Bz_delta-Bz)*Bz/delta

      shape_gradient = deltaB_dot_B/(2*np.pi*self.vmecOutputObject.mu0)

    return shape_gradient
  
  # Call VMEC with boundary specified by boundaryObjective to evaluate which_objective and compute gradient
  # If boundary is not specified, boundary will not be updated
  def evaluate_vmec_objective_grad(self,which_objective,weight_function,delta,boundary=None,update=True):
    if (which_objective=='iota'):
      print("Evaluating iota objective gradient.")
    else:
      print("Error! evaluate_vmec_objective_grad called with incorrect value of"+which_objective)
      sys.exit(1)

    # Evaluate base equilibrium if necessary
    if (boundary is not None):
      vmecOutputObject = self.evaluate_vmec(boundary=boundary,update=update)
    else:
      vmecOutputObject = self.vmecOutputObject

    # Compute objective
    if (which_objective == 'iota'):
      objective = vmecOutputObject.evaluate_iota_objective(weight_function)

    # Compute gradient
    shape_gradient = self.vmec_shape_gradient(which_objective,weight_function,delta,update=update)

    [dfdrmnc,dfdzmns,xm_sensitivity,xn_sensitivity] = \
      parameter_derivatives(shape_gradient,vmecOutputObject,self.mmax_sensitivity,\
                                self.nmax_sensitivity)
    gradient = np.vstack((dfdrmnc,dfdzmns))

    return objective, gradient
  
def call_vmec(path_to_vmec_executable,input_file):
  print(os.getcwd())
  with open('out.txt','w+') as fout:
    with open('err.txt','w+') as ferr:
      exit_code=subprocess.call([path_to_vmec_executable,input_file],\
                          stdout=fout,stderr=ferr)
      # reset file to read from it
      fout.seek(0)
      # save output (if any) in variable
      output=fout.read()

      # reset file to read from it
      ferr.seek(0) 
      # save errors (if any) in variable
      errors = ferr.read()
  
  exit_code = subprocess.call([path_to_vmec_executable,input_file],\
                               stdout=subprocess.PIPE)
  if (exit_code != 0):
    print('VMEC returned with an error (exit_code = '+str(exit_code)+')')
    sys.exit(1)
  return
  
# Note that xn is not multiplied by nfp
def init_modes(mmax,nmax):
  mnmax = (nmax+1) + (2*nmax+1)*mmax
  xm = np.zeros(mnmax)
  xn = np.zeros(mnmax)

  # m = 0 modes
  index = 0
  for jn in range(nmax+1):
    xm[index] = 0
    xn[index] = jn
    index += 1
  
  # m /= 0 modes
  for jm in range(1,mmax+1):
    for jn in range(-nmax,nmax+1):
      xm[index] = jm
      xn[index] = jn
      index += 1
  
  return mnmax, xm, xn
 
# Shape gradient is passed in on [nzeta,ntheta] grid
# Returns derivatives with respect to rmnc and zmns
def parameter_derivatives(shape_gradient,readVmecOutputObject,mmax_sensitivity,nmax_sensitivity):
  
  [Nx,Ny,Nz] = readVmecOutputObject.compute_N()    

  [mnmax_sensitivity, xm_sensitivity, xn_sensitivity] = init_modes(mmax_sensitivity,nmax_sensitivity)

  dfdrmnc = np.zeros(mnmax_sensitivity)
  dfdzmns = np.zeros(mnmax_sensitivity)
  for imn in range(mnmax_sensitivity):
    angle = xm_sensitivity[imn]*readVmecOutputObject.thetas_2d \
      - readVmecOutputObject.nfp*xn_sensitivity[imn]*readVmecOutputObject.zetas_2d
    dfdrmnc[imn] = np.sum(np.sum(np.cos(angle)*(Nx*np.cos(readVmecOutputObject.zetas_2d) \
      + Ny*np.sin(readVmecOutputObject.zetas_2d))*shape_gradient))
    dfdzmns[imn] = np.sum(np.sum(np.sin(angle)*Nz*shape_gradient))
  dfdrmnc = dfdrmnc*readVmecOutputObject.dtheta*readVmecOutputObject.dzeta
  dfdzmns = dfdzmns*readVmecOutputObject.dtheta*readVmecOutputObject.dzeta
  return dfdrmnc, dfdzmns, xm_sensitivity, xn_sensitivity
  
# Approximates finite difference derivative with an N-point stencil with step size epsilon
def finite_difference_derivative(x,function,args=None,epsilon=1e-2,method='forward'):
  if (np.any(method==['forward','centered'])):
    print('Error! Incorrect method passed to finite_difference_derivative.')
    sys.exit(1)
  dfdx = np.zeros(np.shape(x))
  for i in range(np.size(x)):
    if (method=='centered'):
      x_r = np.copy(x)
      x_r[i] = x_r[i]+epsilon
      function_r = function(x_r,*args)
      x_l = np.copy(x)
      x_l[i] = x_l[i]-epsilon
      function_l = function(x_l,*args)
      dfdx[i] = (function_r-function_l)/(2*epsilon)
    if (method=='forward'):
      x_r = np.copy(x)
      x_l = np.copy(x)
      x_r[i] = x_r[i]+epsilon
      function_l = function(x_r,*args)
      function_r = function(x_l,*args)
      dfdx[i] = (function_r-function_l)/(epsilon)
      
  return dfdx