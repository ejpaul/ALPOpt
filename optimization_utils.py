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
from scipy import linalg

"""
Required packages: nlopt, f90nml, scanf
"""

def axis_weight(s):
  return np.exp(-(s-0.1)**2/0.05**2)

class vmecOptimization:
  """
  A class used for optimization of a VMEC equilibrium
  
  ...
  Attributes
  ---------
  mmax_sensitivity (int) : maximum poloidal mode number of boundary to optimize 
  nmax_sensitivity (int) : maximum toroidal mode number of boundary to optimize
  mnmax_sensitivity (int) : total number of boundary modes to optimize
  xm_sensitivity (int array) : poloidal mode numbers of boundary to optimize
  xn_sensitivity (int array) : toroidal mode numbers of boundary to optimize
  callVMEC_function (function) : takes vmecInputFilename (str) as sole input argument and
    returns runs VMEC with given Fortran namelist. See callVMEC.py for examples.
  counter (int) : Used for naming of directories for labeling VMEC function evaluations. 
    Counter is incremented when a new boundary is evaluated. 
  name (str) : Name used as prefix for directory names of new VMEC evaluations.
  vmecInputFilename (str) : Fortran namelist for initial equilibrium evaluation. This
    namelist will be modified as the boundary is modified. 
  directory (str) : Home directory for optimization. 
  mpol (int) : maximum poloidal mode number for VMEC calculations
  ntor (int) : maximum toroidal mode number for VMEC calculations
  mnmax (int) :  total number of modes for VMEC calculations
  xm (int array) : poloidal mode numbers for VMEC calculations
  xn (int array) : toroidal mode numbers for VMEc calculations
  boundary (float array) : current set of boundary harmonics
  boundary_opt (float array) : current set of boundary harmonics which are being optimized 
    (subset of boundary)
  vmecOutputObject (vmecOutput) : instance of object corresponding to most recent equilibrium
    evaluation
  delta (float) : defines size of current perturbation for adjoint shape gradient calculation
  
  """
  def __init__(self,vmecInputFilename,mmax_sensitivity,
             nmax_sensitivity,callVMEC_function,name,delta_curr=10,delta_pres=10,
             woutFilename=None):
    self.mmax_sensitivity = mmax_sensitivity
    self.nmax_sensitivity = nmax_sensitivity
    [mnmax_sensitivity,xm_sensitivity,xn_sensitivity] = \
      init_modes(mmax_sensitivity,nmax_sensitivity)
    self.mnmax_sensitivity = mnmax_sensitivity
    self.xm_sensitivity = xm_sensitivity
    self.xn_sensitivity = xn_sensitivity
    self.callVMEC_function = callVMEC_function
    self.counter = 0 # Counts number of function evals (used for naming directories)
    self.name = name # Name used for directories
    self.vmecInputFilename = vmecInputFilename
    self.directory = os.getcwd()
    self.delta_pres = delta_pres
    self.delta_curr = delta_curr
    
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
    self.mmax = max(mmax_sensitivity,mpol_input-1)
    self.nmax = max(nmax_sensitivity,ntor_input)
    [self.mnmax,self.xm,self.xn] = init_modes(self.mmax,self.nmax)
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
    self.boundary = np.hstack((rmnc,zmns))   
    self.boundary_opt = np.hstack((rmnc_opt,zmns_opt[1::]))
    
    if (woutFilename is None):
      [error_code,self.vmecOutputObject] = self.evaluate_vmec() # Current boundary evaluation
      if (error_code != 0):
        print('Unable to evaluate base VMEC equilibrium in vmecOptimization constructor.')
    else:
      self.vmecOutputObject = readVmecOutput(woutFilename)
        
    
  def evaluate_vmec(self,boundary=None,It=None,pres=None,update=True):
    """
    Evaluates VMEC equilibrium with specified boundary or toroidal current profile update
    
    boundary and It should not be specified simultaneously. 
    If neither are specified, then an exception is raised.
    
    Parameters
    ----------
    boundary (float array) : values of boundary harmonics for VMEC evaluations
    It (float array) : value of toroidal current on half grid VMEC mesh for prescription of profile 
      with "ac_aux_f" Fortran namelist variable
    update (bool) : if True, self.vmecOutputObject is updated and boundary is assigned to self.boundary_opt
    
    Returns
    ----------
    vmecOutput_new (vmecOutput) : object corresponding to output of VMEC evaluations
    
    """
    if (os.getcwd()!=self.directory):
      print("Evaluate_vmec called from incorrect directory. Changing to "+self.directory)
      os.chdir(self.directory)
    if (np.count_nonzero([It,pres,boundary] is not None)>1):
      print("evaluate_vmec called with more than one type of perturbation \
      (boundary, It, pres). This behavior is not supported.")
      sys.exit(1)
    if (boundary is not None):
      # Update equilibrium count
      self.counter += 1
      # Update boundary
      self.update_boundary_opt(boundary)
      directory_name = self.name+"_"+str(self.counter)
    elif (It is not None):
      directory_name = "delta_curr_"+str(self.counter)
    elif (pres is not None):
      directory_name = "delta_pres_"+str(self.counter)
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
        if (pres is not None):
          write_condition = write_condition and (not namelistLineContains(line,"am_aux_s") and \
              not namelistLineContains(line,"am_aux_f") and \
              not namelistLineContains(line,"pmass_type"))
        if (write_condition):
          f2.write(line)
    if (boundary is not None):
      f2.write("mpol = " + str(self.mmax+1) + "\n")
      f2.write("ntor = " + str(self.nmax) + "\n")
      # Write RBC
      for imn in range(self.mnmax):
        if (self.boundary[imn]!=0):
          f2.write('RBC('+str(int(self.xn[imn]))+","\
                   +str(int(self.xm[imn]))+") = "\
                   +str(self.boundary[imn]))
          f2.write("\n")
      # Write ZBS
      for imn in range(self.mnmax):
        if (self.boundary[self.mnmax+imn]!=0):
          f2.write('ZBS('+str(int(self.xn[imn]))+","\
                   +str(int(self.xm[imn]))+") = "\
                   +str(self.boundary[self.mnmax+imn]))
          f2.write("\n")
    if (It is not None):
      curtor = 1.5*It[-1] - 0.5*It[-2]
      s_half = self.vmecOutputObject.s_half
      f2.write("curtor = " + str(curtor) + "\n")
      f2.write("ac_aux_f = ")
      f2.writelines(map(lambda x:str(x)+" ",It))
      f2.write("\n")
      f2.write("ac_aux_s = ")
      f2.writelines(map(lambda x:str(x)+" ",s_half))
      f2.write("\n")
      f2.write("pcurr_type = 'line_segment_I'" + "\n")
    if (pres is not None):
      s_half = self.vmecOutputObject.s_half
      f2.write("am_aux_f = ")
      f2.writelines(map(lambda x:str(x)+" ",pres))
      f2.write("\n")
      f2.write("am_aux_s = ")
      f2.writelines(map(lambda x:str(x)+" ",s_half))
      f2.write("\n")
      f2.write("pmass_type = 'line_segment'" + "\n")
      
    f2.write("/ \n")
    f.close()
    f2.close()

    # Call VMEC with revised input file
    exit_code = self.call_vmec(input_file)
      
    if (exit_code == 0):
      # Read from new equilibrium
      outputFileName = "wout_"+input_file[6::]+".nc"

      vmecOutput_new = readVmecOutput(outputFileName)
      if (boundary is not None and update):
        self.vmecOutputObject = vmecOutput_new
    else:
      vmecOutput_new = None

    os.chdir("..")
    return exit_code, vmecOutput_new
    
  def read_boundary_input(self,xn,xm):
    """
    Read in boundary harmonics (rbc, zbs) from Fortran namelist
    
    Parameters
    ----------
    xm (int array) : poloidal mode numbers to read
    xn (int array) : toroidal mode numbers to read
    
    Returns
    ----------
    rbc (float array) : boundary harmonics for radial coordinate (same size as xm and xn)
    zbs (float array) : boundary harmonics for height coordinate (same size as xm and xn)
    
    """
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
    
  def update_boundary_opt(self,boundary_opt_new):
    """
    Updates boundary harmonics at new evaluation point 
    
    Parameters
    ----------
    boundary_opt_new (float array) : new boundary to evaluate (same size as self.boundary_opt)
        
    """
    rmnc_opt_new = boundary_opt_new[0:self.mnmax_sensitivity]
    zmns_opt_new = boundary_opt_new[self.mnmax_sensitivity::]
    # m = 0, n = 0 mode excluded in boundary_opt
    zmns_opt_new = np.append([0],zmns_opt_new)
    self.boundary_opt = np.copy(boundary_opt_new)

    rmnc_new = self.boundary[0:self.mnmax]
    zmns_new = self.boundary[self.mnmax::]
    for imn in range(self.mnmax_sensitivity):
      cond = np.logical_and((self.xm == self.xm_sensitivity[imn]),\
        (self.xn == self.xn_sensitivity[imn]))
      if (any(cond)):
        rmnc_new[cond] = rmnc_opt_new[imn]
        zmns_new[cond] = zmns_opt_new[imn]
    self.boundary = np.hstack((rmnc_new,zmns_new))
    return
  
  # Call VMEC with boundary specified by boundaryObjective to evaluate which_objective
  def evaluate_vmec_objective(self,boundary=None,which_objective='iota',weight_function=axis_weight,update=True):
    """
    Evaluates vmec objective function with prescribed boundary
    
    Parameters
    ----------
    boundary (float array) : new boundary to evaluate. If None, objective will be computed from self.vmecOutputObject
        
    Returns
    ----------
    objective function (float) : value of objective function defined through which_objective and weight_function.
      If VMEC completes with an error, the value of the objective function is set to 1e12.
    """
    if (which_objective=='iota'):
      print("Evaluating iota objective.")
    elif (which_objective=='well'):
      print("Evaluating well objective.")
    elif (which_objective=='volume'):
      print("Evaluating volume objective.")
    elif (which_objective=='area'):
      print("Evaluating area objective.")
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
    error_code = 0
    if (boundary is not None and (boundary!=self.boundary_opt).any()):
      [error_code,vmecOutputObject] = self.evaluate_vmec(boundary=boundary,update=update)
      if (error_code != 0):
        print('VMEC returned with an error when evaluating new boundary in evaluate_vmec_objective. \
              Objective function will be set to 1e12')
    else:
      vmecOutputObject = self.vmecOutputObject
    # Reset old boundary 
    if (update==False):
      self.update_boundary_opt(boundary_old)
    
    if (which_objective == 'iota'):
      objective_function = vmecOutputObject.evaluate_iota_objective(weight_function)
    elif (which_objective == 'well'):
      objective_function = vmecOutputObject.evaluate_well_objective(weight_function)
    elif (which_objective == 'volume'):
      objective_function = vmecOutputObject.volume
    elif (which_objective == 'area'):
      objective_function = vmecOutputObject.area(vmecOutputObject.ns-1)
      
    if (error_code == 0):
      return objective_function
    else:
      return 1e12

  def vmec_shape_gradient(self,boundary=None,which_objective='iota',weight_function=axis_weight,update=True):
    """
    Evaluates shape gradient for objective function 
    
    Parameters
    ----------
    boundary (float array) : new boundary to evaluate. If None, objective will be computed from self.vmecOutputObject
        
    Returns
    ----------
    objective function (float) : value of objective function defined through which_objective and weight_function.
      If VMEC completes with an error, the value of the objective function is set to 1e12.
    """
    if (which_objective=='iota'):
      print("Evaluating iota objective shape gradient.")
      delta = self.delta_curr
    elif (which_objective=='well'):
      print("Evaluating well objective shape gradient.")
      delta = self.delta_pres
    elif (which_objective=='volume'):
      print("Evaluating volume shape gradient.")
      return 1
    elif (which_objective=='area'):
      print("Evaluating area shape gradient.")
    else:
      print("Error! vmec_shape_gradient called with incorrect value of"+which_objective)
      sys.exit(1)
      
    # Evaluate base equilibrium if necessary
    if (boundary is not None and (boundary!=self.boundary_opt).all()):
      [error_code, vmecOutputObject] = self.evaluate_vmec(boundary=boundary,update=update)
      if (error_code != 0):
        print('Unable to evaluate base VMEC equilibrium in vmec_shape_gradient.')
        sys.exit(1)
    else:
      vmecOutputObject = self.vmecOutputObject
    [Bx, By, Bz, theta_arclength] = vmecOutputObject.B_on_arclength_grid()

    if (which_objective == 'iota'):
      It_half = vmecOutputObject.compute_current()
      It_new = It_half + delta*weight_function(self.vmecOutputObject.s_half)

      [error_code, vmecOutput_delta] = self.evaluate_vmec(It=It_new)
      if (error_code != 0):
        print('Unable to evaluate VMEC equilibrium with current perturbation in vmec_shaep_gradient.')
        sys.exit(1)

      [Bx_delta, By_delta, Bz_delta, theta_arclength_delta] = vmecOutput_delta.B_on_arclength_grid()
      for izeta in range(self.vmecOutputObject.nzeta):
        f = interpolate.InterpolatedUnivariateSpline(theta_arclength_delta[izeta,:],Bx_delta[izeta,:])
        Bx_delta[izeta,:] = f(theta_arclength[izeta,:])
        f = interpolate.InterpolatedUnivariateSpline(theta_arclength_delta[izeta,:],By_delta[izeta,:])
        By_delta[izeta,:] = f(theta_arclength[izeta,:])
        f = interpolate.InterpolatedUnivariateSpline(theta_arclength_delta[izeta,:],Bz_delta[izeta,:])
        Bz_delta[izeta,:] = f(theta_arclength[izeta,:])

      deltaB_dot_B = ((Bx_delta-Bx)*Bx + (By_delta-By)*By + (Bz_delta-Bz)*Bz)/delta

      shape_gradient = deltaB_dot_B/(2*np.pi*self.vmecOutputObject.mu0)
    elif (which_objective == 'well'):
      pres = vmecOutputObject.pres
      pres_new = pres + delta*weight_function(self.vmecOutputObject.s_half)
      
      [error_code, vmecOutput_delta] = self.evaluate_vmec(pres=pres_new)
      if (error_code != 0):
        print('Unable to evaluate VMEC equilibrium with pressure perturbation in vmec_shaep_gradient.')
        sys.exit(1)  
        
      [Bx_delta, By_delta, Bz_delta, theta_arclength_delta] = vmecOutput_delta.B_on_arclength_grid()
      for izeta in range(self.vmecOutputObject.nzeta):
        f = interpolate.InterpolatedUnivariateSpline(theta_arclength_delta[izeta,:],Bx_delta[izeta,:])
        Bx_delta[izeta,:] = f(theta_arclength[izeta,:])
        f = interpolate.InterpolatedUnivariateSpline(theta_arclength_delta[izeta,:],By_delta[izeta,:])
        By_delta[izeta,:] = f(theta_arclength[izeta,:])
        f = interpolate.InterpolatedUnivariateSpline(theta_arclength_delta[izeta,:],Bz_delta[izeta,:])
        Bz_delta[izeta,:] = f(theta_arclength[izeta,:])

      deltaB_dot_B = ((Bx_delta-Bx)*Bx + (By_delta-By)*By + (Bz_delta-Bz)*Bz)/delta
      shape_gradient = deltaB_dot_B/(self.vmecOutputObject.mu0) + weight_function(1)
    elif (which_objective == 'area'):
      shape_gradient = vmecOutputObject.mean_curvature(vmecOutputObject.ns-1)
      
    return shape_gradient
  
  # Call VMEC with boundary specified by boundaryObjective to evaluate which_objective and compute gradient
  # If boundary is not specified, boundary will not be updated
  def evaluate_vmec_objective_grad(self,boundary=None,which_objective='iota',weight_function=axis_weight,update=True):
    if (which_objective=='iota'):
      print("Evaluating iota objective gradient.")
    elif (which_objective=='well'):
      print("Evaluating well objective gradient.")
    elif (which_objective=='volume'):
      print("Evaluating volume objective gradient.")
    elif (which_objective=='area'):
      print("Evaluating area objective gradient.")
    else:
      print("Error! evaluate_vmec_objective_grad called with incorrect value of"+str(which_objective))
      sys.exit(1)

    # Evaluate base equilibrium if necessary
    if (boundary is not None and (boundary!=self.boundary_opt).any()):
      [error_code, vmecOutputObject] = self.evaluate_vmec(boundary=boundary,update=update)
      if (error_code != 0):
        print('Unable to evaluate base VMEC equilibrium in vmec_shape_gradient.')
        sys.exit(1)
    else:
      vmecOutputObject = self.vmecOutputObject

    # Compute gradient
    shape_gradient = self.vmec_shape_gradient(which_objective=which_objective,weight_function=weight_function,update=update)

    [dfdrmnc,dfdzmns,xm_sensitivity,xn_sensitivity] = \
      parameter_derivatives(shape_gradient,vmecOutputObject,self.mmax_sensitivity,\
                                self.nmax_sensitivity)
    gradient = np.hstack((dfdrmnc,dfdzmns[1::]))

    return gradient
  
  def call_vmec(self,input_file):
    self.callVMEC_function(input_file)
    # Check for VMEC errors
    wout_filename = "wout_"+input_file[6::]+".nc"
    f = netcdf.netcdf_file(wout_filename,'r',mmap=False)
    error_code = f.variables["ier_flag"][()]
    if (error_code != 0):
      print('VMEC completed with error code '+str(error_code))
    return error_code
  
  def optimization_plot(self,which_objective,weight):
    os.chdir(self.directory)
    # Get all subdirectories
    dir_list = next(os.walk('.'))[1]
    # Filter those beginning with self.name
    for dir in dir_list:
      if (len(dir)<len(self.name)+1):
        dir_list.remove(dir)
      elif (dir[0:len(self.name)+1]!=self.name+'_'):
        dir_list.remove(dir)
      elif (not dir[len(self.name)+1].isdigit()):
        dir_list.remove(dir)
    objective = np.zeros(len(dir_list))
    for i in range(len(dir_list)):
      os.chdir(dir_list[i])
      wout_filename = 'wout_'+dir_list[i]+'.nc'
      if (os.path.exists(wout_filename)):
        f = netcdf.netcdf_file(wout_filename,'r',mmap=False)
        error_code = f.variables["ier_flag"][()]
        if (error_code != 0):
          print('VMEC completed with error code '+str(error_code)+' in '+dir+". Moving on to next directory. \
            Objective function will be set to zero.")
        else:
          readVmecObject = readVmecOutput('wout_'+dir_list[i]+'.nc')
          if (which_objective=='iota'):
            objective[i] = readVmecObject.evaluate_iota_objective(weight)
          elif (which_objective=='well'):
            objective[i] = readVmecObject.evaluate_well_objective(weight)
          elif (which_objective=='volume'):
            objective[i] = readVmecObject.volume
          elif (which_objective=='area'):
            objective[i] = readVmecObject.area
          else:
            print("Incorrect objective specified in optimization_plot!")
            sys.exit(1)
      else:
        print("wout_filename not found in "+ dir+". Moving on to next directory Objective function will be set to zero.")
      os.chdir('..')
    return objective
  
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

    