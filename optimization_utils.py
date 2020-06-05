import numpy as np
import os
from vmecOutput import *
from vmecInput import *
import errno
import subprocess
import sys
import shutil
from scipy import interpolate
from scipy import linalg
from scipy import optimize
from matplotlib import pyplot as plt
import scipy
import copy

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
             woutFilename=None,ntheta=100,nzeta=100):
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
    self.ntheta = ntheta
    self.nzeta = nzeta
    self.jacobian_threshold = 1e-4
    
    self.vmecInputObject = readVmecInput(vmecInputFilename,ntheta,nzeta)
    self.nfp = self.vmecInputObject.nfp
    mpol_input = self.vmecInputObject.mpol
    ntor_input = self.vmecInputObject.ntor
    mnmax_input = self.vmecInputObject.mnmax
    rbc_input = self.vmecInputObject.rbc
    zbs_input = self.vmecInputObject.zbs
    xm_input = self.vmecInputObject.xm
    xn_input = self.vmecInputObject.xn
    self.vmec_evaluated = False
    
    if (mpol_input<2):
      print('Error! mpol must be > 1.')
      sys.exit(1)
        
    # mpol, ntor for calculaitons must be large enough for sensitivity required
    self.mmax = max(mmax_sensitivity,mpol_input-1)
    self.nmax = max(nmax_sensitivity,ntor_input)
    [self.mnmax,self.xm,self.xn] = init_modes(self.mmax,self.nmax)
    [mnmax_sensitivity,xm_sensitivity,xn_sensitivity] = init_modes(mmax_sensitivity,nmax_sensitivity)

    # Update input object with required mpol and ntor
    self.vmecInputObject.update_modes(self.mmax+1,self.nmax)
    
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
    self.norm_normal = None
        
    self.vmecOutputObject = None
    if (woutFilename is None):
      [error_code,self.vmecOutputObject] = self.evaluate_vmec() # Current boundary evaluation
      if (error_code != 0):
        print('Unable to evaluate base VMEC equilibrium in vmecOptimization constructor.')
    else:
      self.vmecOutputObject = readVmecOutput(woutFilename,self.ntheta,self.nzeta)  
      
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
    
    input_file = "input."+directory_name
    os.chdir(directory_name)

    inputObject_new = copy.deepcopy(self.vmecInputObject)
    inputObject_new.input_filename = input_file
    inputObject_new.ntor = self.nmax
    inputObject_new.mpol = self.mmax+1
    # Edit input filename with boundary 
    if (It is not None):
      curtor = 1.5*It[-1] - 0.5*It[-2]
      s_half = self.vmecOutputObject.s_half
      if (self.vmecOutputObject.ns>101):
        s_spline = np.linspace(0,1,102)
        ds_spline = s_spline[1]-s_spline[0]
        s_spline_half = s_spline - 0.5*ds_spline
        s_spline_half = np.delete(s_spline_half,0)
        It_spline = interpolate.InterpolatedUnivariateSpline(s_half,It)
        It = It_spline(s_spline_half)
        s_half = s_spline_half
      inputObject_new.curtor = curtor
      inputObject_new.ac_aux_f = list(It)
      inputObject_new.ac_aux_s = list(s_half)
      inputObject_new.pcurr_type = "line_segment_I"
    if (pres is not None):
      s_half = self.vmecOutputObject.s_half
      if (self.vmecOutputObject.ns>101):
        s_spline = np.linspace(0,1,102)
        ds_spline = s_spline[1]-s_spline[0]
        s_spline_half = s_spline - 0.5*ds_spline
        s_spline_half = np.delete(s_spline_half,0)
        pres_spline = interpolate.InterpolatedUnivariateSpline(s_half,pres)
        pres = pres_spline(s_spline_half)
        s_half = s_spline_half
      inputObject_new.am_aux_f = list(pres)
      inputObject_new.am_aux_s = list(s_half)
      inputObject_new.pmass_type = "line_segment"
    if (boundary is not None):
      inputObject_new.rbc = self.boundary[0:self.mnmax]
      inputObject_new.zbs = self.boundary[self.mnmax::]
    
    # Call VMEC with revised input file
    exit_code = self.call_vmec(inputObject_new)
      
    if (exit_code == 0):
      # Read from new equilibrium
      outputFileName = "wout_"+input_file[6::]+".nc"
      vmecOutput_new = readVmecOutput(outputFileName,self.ntheta,self.nzeta)
      if (boundary is not None and update):
        self.vmecOutputObject = vmecOutput_new
        self.vmec_evaluated = True
    else:
      vmecOutput_new = None

    os.chdir("..")
    return exit_code, vmecOutput_new
    
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
  
  def evaluate_input_objective(self,boundary=None,which_objective='volume',update=True):
    if (which_objective=='volume'):
      print("Evaluating volume objective.")
    elif (which_objective=='area'):
      print("Evaluating area objective.")
    elif (which_objective=='jacobian'):
      print("Evaluating jacobian objective.")
    elif (which_objective=='radius'):
      print("Evaluating radius objective.")
    elif (which_objective=='normalized_jacobian'):
      print("Evaluating normalized jacobian objective.")
    else:
      print("incorrect value of which_objective in evaluate_input_objective.")
      sys.exit(1)
    if (boundary is None):
      boundary = self.boundary_opt
    if (np.shape(boundary)!=np.shape(self.boundary_opt)):
      print("Error! evaluate_vmec called with incorrect boundary shape.")
      sys.exit(1)
      
    if ((boundary != self.boundary_opt).any()):
      # Save old boundary if update=False
      if (update==False):
        boundary_old = np.copy(self.boundary_opt)
      # Update equilibrium count
      self.counter += 1
      # Update boundary
      self.update_boundary_opt(boundary)
      # VMEC has not been called with this boundary
      self.vmec_evaluated = False
      directory_name = self.name+"_"+str(self.counter)
      print("Creating new input object in "+directory_name)
      # Make directory for VMEC evaluations
      try:
        os.mkdir(directory_name)
      except OSError as exc:
        if exc.errno != errno.EEXIST:
          print('Warning. Directory '+directory_name+' already exists. Contents may be overwritten.')
          raise
        pass
      input_file = "input."+directory_name
      # Copy input object and edit with new boundary
      vmecInputObject = copy.deepcopy(self.vmecInputObject)
      vmecInputObject.input_filename = input_file
      vmecInputObject.rbc = np.copy(self.boundary[0:self.mnmax])
      vmecInputObject.zbs = np.copy(self.boundary[self.mnmax::])
      os.chdir(directory_name)
      vmecInputObject.print_namelist()
      os.chdir('..')
      # Reset old boundary 
      if (update==False):
        self.update_boundary_opt(boundary_old)
      else:
        self.vmecInputObject = vmecInputObject
    else:
      vmecInputObject = copy.deepcopy(self.vmecInputObject)

    if (which_objective == 'volume'):
      objective_function = vmecInputObject.volume()
    elif (which_objective == 'area'):
      objective_function = vmecInputObject.area()
    elif (which_objective == 'jacobian'):
      objective_function = vmecInputObject.jacobian()
    elif (which_objective == 'radius'):
      [X,Y,Z,R] = vmecInputObject.position()
      objective_function = R
    elif (which_objective == 'normalized_jacobian'):
      objective_function = vmecInputObject.normalized_jacobian()
    return objective_function
  
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
    elif (which_objective=='jacobian'):
      print("Evaluating jacobian objective.")
    elif (which_objective=='radius'):
      print("Evaluating radius objective.")
    elif (which_objective=='normalized_jacobian'):
      print("Evaluating normalized jacobian objective.")
    else:
      print("Error! evaluate_vmec called with incorrect value of which_objective")
      sys.exit(1)
    if (boundary is None):
      boundary = self.boundary_opt
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
        print('VMEC returned with an error when evaluating new boundary in evaluate_vmec_objective.')
        print('Objective function will be set to 1e12')
    else:
      vmecOutputObject = self.vmecOutputObject
    # Reset old boundary 
    if (update==False):
      self.update_boundary_opt(boundary_old)
    
    if (error_code == 0):
      if (which_objective == 'iota'):
        objective_function = vmecOutputObject.evaluate_iota_objective(weight_function)
      elif (which_objective == 'well'):
        objective_function = vmecOutputObject.evaluate_well_objective(weight_function)
      elif (which_objective == 'volume'):
        objective_function = vmecOutputObject.volume
      elif (which_objective == 'area'):
        objective_function = vmecOutputObject.area(vmecOutputObject.ns-1)
      elif (which_objective == 'jacobian'):
        objective_function = vmecOutputObject.jacobian(-1)
      elif (which_objective == 'radius'):
        [X,Y,Z,R] = vmecOutputObject.compute_position(-1)
        objective_function = R
      elif (which_objective == 'normalized_jacobian'):
        objective_function = vmecOutputObject.normalized_jacobian(-1)
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
      return np.ones(np.shape(self.vmecOutputObject.zetas_2d))
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

    if (which_objective == 'iota'):
      [Bx, By, Bz, theta_arclength] = vmecOutputObject.B_on_arclength_grid()
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
      [Bx, By, Bz, theta_arclength] = vmecOutputObject.B_on_arclength_grid()
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
  
  def evaluate_input_objective_grad(self,boundary=None,which_objective='volume',update=True):
    if (which_objective=='volume'):
      print("Evaluating volume objective gradient.")
    elif (which_objective=='area'):
      print("Evaluating area objective gradient.")
    elif (which_objective=='jacobian'):
      print("Evaluating jacobian objective gradient.")
    elif (which_objective=='radius'):
      print("Evaluating radius objective gradient.")
    elif (which_objective=='normalized_jacobian'):
      print("Evaluating normalized_jacobian objective gradient.")
    else:
      print("Error! evaluate_input_objective_grad called with incorrect value of which_objective.")
      sys.exit(1)  
      
    # Get new input object if necessary
    if (boundary is not None and (boundary!=self.boundary_opt).any()):
      if (np.shape(boundary)!=np.shape(self.boundary_opt)):
        print("Error! evaluate_vmec called with incorrect boundary shape.")
        sys.exit(1)
      # Save old boundary if update=False
      if (update==False):
        boundary_old = np.copy(self.boundary_opt)
      # Update equilibrium count
      self.counter += 1
      # Update boundary
      self.update_boundary_opt(boundary)
      # VMEC has not been called with this boundary
      self.vmec_evaluated = False
      directory_name = self.name+"_"+str(self.counter)
      # Make directory for VMEC evaluations
      try:
        os.mkdir(directory_name)
      except OSError as exc:
        if exc.errno != errno.EEXIST:
          print('Warning. Directory '+directory_name+' already exists. Contents may be overwritten.')
          raise
        pass
      input_file = "input."+directory_name
      # Copy input object and edit with new boundary
      vmecInputObject = copy.deepcopy(self.vmecInputObject)
      vmecInputObject.input_filename = input_file
      vmecInputObject.rbc = self.boundary[0:self.mnmax]
      vmecInputObject.zbs = self.boundary[self.mnmax::]
      if (update==False):
        self.update_boundary_opt(boundary_old)
    else:
      vmecInputObject = self.vmecInputObject
      
    if (which_objective == 'jacobian'):
      [dfdrmnc,dfdzmns] = vmecInputObject.jacobian_derivatives(self.xm_sensitivity,self.xn_sensitivity)
    elif (which_objective == 'radius'):
      [dfdrmnc,dfdzmns] = vmecInputObject.radius_derivatives(self.xm_sensitivity,self.xn_sensitivity)
    elif (which_objective == 'normalized_jacobian'):
      [dfdrmnc,dfdzmns] = vmecInputObject.normalized_jacobian_derivatives(self.xm_sensitivity,self.xn_sensitivity)
    elif (which_objective == 'area'):
      [dfdrmnc,dfdzmns] = vmecInputObject.area_derivatives(self.xm_sensitivity,self.xn_sensitivity)
    elif (which_objective == 'volume'):
      [dfdrmnc,dfdzmns] = vmecInputObject.volume_derivatives(self.xm_sensitivity,self.xn_sensitivity)

    if (dfdrmnc.ndim==1):
      gradient = np.hstack((dfdrmnc,dfdzmns[1::]))
    elif (dfdrmnc.ndim==3):
      gradient = np.vstack((dfdrmnc,dfdzmns[1::,:,:]))
    else:
      print("Incorrect shape of derivatives encountered in evaluate_input_objective_grad.")
      sys.exit(0)
      
    return np.squeeze(gradient)
  
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
    elif (which_objective=='jacobian'):
      print("Evaluating jacobian objective gradient.")
    elif (which_objective=='radius'):
      print("Evaluating radius objective gradient.")
    elif (which_objective=='normalized_jacobian'):
      print("Evaluating normalized_jacobian gradient.")
    else:
      print("Error! evaluate_vmec_objective_grad called with incorrect value of which_objective.")
      sys.exit(1)

    # Evaluate base equilibrium if necessary
    if (boundary is not None and (boundary!=self.boundary_opt).any()):
      [error_code, vmecOutputObject] = self.evaluate_vmec(boundary=boundary,update=update)
      if (error_code != 0):
        print('Unable to evaluate base VMEC equilibrium in vmec_shape_gradient.')
        sys.exit(1)
    else:
      vmecOutputObject = self.vmecOutputObject

    if (which_objective == 'jacobian'):
      [dfdrmnc,dfdzmns] = vmecOutputObject.jacobian_derivatives(self.xm_sensitivity,self.xn_sensitivity)
    elif (which_objective == 'radius'):
      [dfdrmnc,dfdzmns] = vmecOutputObject.radius_derivatives(self.xm_sensitivity,self.xn_sensitivity)
    elif (which_objective == 'normalized_jacobian'):
      [dfdrmnc,dfdzmns] = vmecOutputObject.normalized_jacobian_derivatives(self.xm_sensitivity,self.xn_sensitivity)
    else:
      # Compute gradient
      shape_gradient = self.vmec_shape_gradient(which_objective=which_objective,weight_function=weight_function,update=update)
      [dfdrmnc,dfdzmns,xm_sensitivity,xn_sensitivity] = \
        parameter_derivatives(shape_gradient,vmecOutputObject,self.mmax_sensitivity,\
                                self.nmax_sensitivity)
    if (dfdrmnc.ndim==1):
      gradient = np.hstack((dfdrmnc,dfdzmns[1::]))
    elif (dfdrmnc.ndim==3):
      gradient = np.vstack((dfdrmnc,dfdzmns[1::,:,:]))
    else:
      print("Incorrect shape of derivatives encountered in evaluate_vmec_objective_grad.")
      sys.exit(0)
      
    return np.squeeze(gradient)
  
  def test_jacobian(self,vmecInputObject=None):
    if (vmecInputObject is None):
      vmecInputObject = self.vmecInputObject
    jac_grid = vmecInputObject.jacobian()
    # Normalize by average Jacobian
    norm = np.sum(jac_grid)/np.size(jac_grid)
    jacobian_diagnostic = np.min(jac_grid)/norm
    if jacobian_diagnostic < self.jacobian_threshold:
      return False
    else:
      return True
    
  def test_boundary(self,vmecInputObject=None):
    if (vmecInputObject is None):
      vmecInputObject = self.vmecInputObject
    [X,Y,Z,R] = vmecInputObject.position()
    # Iterate over cross-sections
    for izeta in range(self.nzeta):
      if (self_intersect(R[izeta,:],Z[izeta,:])):
        return False
    return True
  
  def call_vmec(self,vmecInputObject):
    if (vmecInputObject is None):
      vmecInputObject = copy.deepcopy(vmecInputObject)
    orientable = self.test_jacobian(vmecInputObject)

    if (not orientable):
      print("Requested boundary has ill-conditioned Jacobian. Surface is likely self-intersecting. I will not call"\
        "VMEC")
      return 15
    else:
      inSurface = vmecInputObject.test_axis()
      if (not inSurface):
        print('Initial magnetic axis does not lie within requested boundary! Trying a modified axis shape.')
        vmecInputObject.modify_axis()
                
    # VMEC error codes (from vmec_params.f) :
    # norm_term_flag=0
    # bad_jacobian_flag=1
    # more_iter_flag=2
    # jac75_flag=4
    # input_error_flag=5
    # phiedge_error_flag=7
    # ns_error_flag=8
    # misc_error_flag=9
    # successful_term_flag=11
    # bsub_bad_js1_flag=12
    # r01_bad_value_flag=13
    # arz_bad_value_flag=14
    vmecInputObject.print_namelist()
    self.callVMEC_function(vmecInputObject.input_filename)
    # Check for VMEC errors
    wout_filename = "wout_"+vmecInputObject.input_filename[6::]+".nc"
    f = netcdf.netcdf_file(wout_filename,'r',mmap=False)
    error_code = f.variables["ier_flag"][()]
    # Check if tolerances were met
    ftolv = f.variables["ftolv"][()]
    fsqr = f.variables["fsqr"][()]
    fsqz = f.variables["fsqz"][()]
    fsql = f.variables["fsql"][()]
    if (fsqr>ftolv or fsqz>ftolv or fsql>ftolv):
      error_code = 2
    
    if (error_code != 0):
      print('VMEC completed with error code '+str(error_code))
    
    return error_code
  
  def optimization_plot(self,which_objective,weight=axis_weight,objective_type='vmec'):
    os.chdir(self.directory)
    # Get all subdirectories
    dir_list_all = next(os.walk('.'))[1]
    # Filter those beginning with self.name
    objective_number = []
    dir_list = []
    for dir in dir_list_all:
      if ((len(dir)>=len(self.name)+1) and 
          (dir[0:len(self.name)+1]==self.name+'_') and 
          (dir[len(self.name)+1].isdigit())):
        dir_list.append(dir)
        objective_number.append(int(dir[len(self.name)+1::]))
    # Sort by evaluation number
    sort_indices = np.argsort(objective_number)
    dir_list = np.array(dir_list)[sort_indices]
    objective = np.zeros(len(dir_list))
    for i in range(len(dir_list)):
      os.chdir(dir_list[i])
      if (objective_type == 'input'):
        input_filename = 'input.'+dir_list[i]
        readInputObject = readVmecInput(input_filename,self.ntheta,self.nzeta)
        if (which_objective=='volume'):
          objective[i] = readInputObject.volume()
        elif (which_objective=='area'):
          objective[i] = readInputObject.area()
        elif (which_objective=='jacobian'):
          jacobian = readInputObject.jacobian()
          objective[i] = np.min(jacobian)
        elif (which_objective=='normalized_jacobian'):
          normalized_jacobian = readInputObject.normalized_jacobian()
          objective[i] = np.min(normalized_jacobian)
        elif (which_objective=='radius'):
          [X,Y,Z,R] = readInputObject.position()
          objective[i] = np.min(R)
      elif (objective_type == 'vmec'):
        wout_filename = 'wout_'+dir_list[i]+'.nc'
        if (os.path.exists(wout_filename)):
          f = netcdf.netcdf_file(wout_filename,'r',mmap=False)
          error_code = f.variables["ier_flag"][()]
          if (error_code != 0):
            print('VMEC completed with error code '+str(error_code)+' in '+dir+". Moving on to next directory. \
              Objective function will be set to zero.")
          else:
            readVmecObject = readVmecOutput('wout_'+dir_list[i]+'.nc',self.ntheta,self.nzeta)
            if (which_objective=='iota'):
              objective[i] = readVmecObject.evaluate_iota_objective(weight)
            elif (which_objective=='well'):
              objective[i] = readVmecObject.evaluate_well_objective(weight)
            elif (which_objective=='volume'):
              objective[i] = readVmecObject.volume
            elif (which_objective=='area'):
              objective[i] = readVmecObject.area
            elif (which_objective=='radius'):
              [X,Y,Z,R] = readVmecObject.compute_position()
              objective[i] = np.min(R)
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
  [Nx,Ny,Nz] = readVmecOutputObject.compute_N(-1)    

  [mnmax_sensitivity, xm_sensitivity, xn_sensitivity] = init_modes(mmax_sensitivity,nmax_sensitivity)

  dfdrmnc = np.zeros(mnmax_sensitivity)
  dfdzmns = np.zeros(mnmax_sensitivity)
  for imn in range(mnmax_sensitivity):
    angle = xm_sensitivity[imn]*readVmecOutputObject.thetas_2d \
      - readVmecOutputObject.nfp*xn_sensitivity[imn]*readVmecOutputObject.zetas_2d
    dfdrmnc[imn] = np.sum(np.sum(np.cos(angle)*(Nx*np.cos(readVmecOutputObject.zetas_2d) \
      + Ny*np.sin(readVmecOutputObject.zetas_2d))*shape_gradient))
    dfdzmns[imn] = np.sum(np.sum(np.sin(angle)*Nz*shape_gradient))
  dfdrmnc = -dfdrmnc*readVmecOutputObject.dtheta*readVmecOutputObject.dzeta*readVmecOutputObject.nfp
  dfdzmns = -dfdzmns*readVmecOutputObject.dtheta*readVmecOutputObject.dzeta*readVmecOutputObject.nfp
  return dfdrmnc, dfdzmns, xm_sensitivity, xn_sensitivity
  
# Approximates finite difference derivative with an N-point stencil with step size epsilon
def finite_difference_derivative(x,function,args=None,epsilon=1e-2,method='forward',dim1=1,dim2=1):
  if (np.any(method==['forward','centered'])):
    print('Error! Incorrect method passed to finite_difference_derivative.')
    sys.exit(1)
  
  if (dim1 == 1 and dim2 == 1):
    dfdx = np.zeros((len(x)))
    ndim = 1
  elif (dim1 != 1 and dim2 == 1):
    dfdx = np.zeros((len(x),dim1))
    ndim = 2
  else:
    dfdx = np.zeros((len(x),dim1,dim2))
    ndim = 3
  for i in range(np.size(x)):
    if (method=='centered'):
      x_r = np.copy(x)
      x_r[i] = x_r[i]+epsilon
      if (args is not None):
        function_r = function(x_r,*args).copy()
      else:
        function_r = function(x_r).copy()
      x_l = np.copy(x)
      x_l[i] = x_l[i]-epsilon
      if (args is not None):
        function_l = function(x_l,*args).copy()
      else:
        function_l = function(x_l).copy()
      if (ndim == 3):
        dfdx[i,:,:] = (function_r-function_l)/(2*epsilon)
      elif (ndim == 2):
        dfdx[i,:] = (function_r-function_l)/(2*epsilon)
      elif (ndim == 1):
        dfdx[i] = (function_r-function_l)/(2*epsilon)
    if (method=='forward'):
      x_r = np.copy(x)
      x_l = np.copy(x)
      x_r[i] = x_r[i]+epsilon
      if (args is not None):
        function_l = function(x_r,*args)
        function_r = function(x_l,*args)
      else:
        function_l = function(x_r)
        function_r = function(x_l)        
      if (ndim == 3):
        dfdx[i,:,:] = (function_r-function_l)/(epsilon)
      elif (ndim == 2):
        dfdx[i,:] = (function_r-function_l)/(epsilon)
      elif (ndim == 1):
        dfdx[i] = (function_r-function_l)/(epsilon)
  return dfdx

def plot_surface_axis(vmecInputFilename,izeta=0,angle1=0,angle2=np.pi):
  ntheta = 500
  nzeta = 501
  thetas = np.linspace(0,2*np.pi,ntheta+1)
  zetas = np.linspace(0,2*np.pi,nzeta+1)
  thetas = np.delete(thetas,-1)
  zetas = np.delete(zetas,-1)
  [thetas_2d,zetas_2d] = np.meshgrid(thetas,zetas)
  
  nml = f90nml.read(vmecInputFilename)
  nml = nml.get("indata")
  nfp = nml.get("nfp")
  raxis = nml.get("raxis")
  zaxis = nml.get("zaxis")
  mpol_input = nml.get("mpol")
  ntor_input = nml.get("ntor")
  # max(xm) = mpol-1 in VMEC
  [mnmax,xm,xn] = init_modes(mpol_input-1,ntor_input)
  # Returns rbc and zbs on grid of input modes 
  rbc_array = np.array(nml["rbc"]).T
  zbs_array = np.array(nml["zbs"]).T

  rbc_array[rbc_array==None] = 0
  zbs_array[zbs_array==None] = 0
  [dim1_rbc,dim2_rbc] = np.shape(rbc_array)
  [dim1_zbs,dim2_zbs] = np.shape(zbs_array)

  [nmin_rbc,mmin_rbc,nmax_rbc,mmax_rbc] = min_max_indices_2d('rbc',vmecInputFilename)
  nmax_rbc = max(nmax_rbc,-nmin_rbc)
  [nmin_zbs,mmin_zbs,nmax_zbs,mmax_zbs] = min_max_indices_2d('zbs',vmecInputFilename)
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

  R = np.zeros(np.shape(thetas_2d))
  Z = np.zeros(np.shape(thetas_2d))
  Raxis = np.zeros(np.shape(zetas))
  Zaxis = np.zeros(np.shape(zetas))
  for im in range(mnmax):
    angle = float(xm[im])*thetas_2d - float(nfp*xn[im])*zetas_2d
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    R = R + rbc[im]*cos_angle
    Z = Z + zbs[im]*sin_angle
  for im in range(len(raxis)):
    angle = -float(im*nfp)*zetas
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)
    Raxis = Raxis + raxis[im]*cos_angle
    Zaxis = Zaxis + zaxis[im]*sin_angle  

  plt.figure()
  plt.plot(R[izeta,:],Z[izeta,:])
  plt.plot(Raxis[izeta],Zaxis[izeta],marker='.')
  plt.show()
  
# Check if segment defined by (p1, p2) intersections segment defined by (p2, p4)
def segment_intersect(p1,p2,p3,p4):
  
  assert(isinstance(p1,(list,np.ndarray)))
  assert(len(p1)==2 and np.array(p1).ndim==1)
  assert(isinstance(p2,(list,np.ndarray)))
  assert(len(p2)==2 and np.array(p2).ndim==1)
  assert(isinstance(p3,(list,np.ndarray)))
  assert(len(p3)==2 and np.array(p3).ndim==1)
  assert(isinstance(p4,(list,np.ndarray)))
  assert(len(p4)==2 and np.array(p4).ndim==1)
  
  l1 = np.array(p3) - np.array(p1)
  l2 = np.array(p2) - np.array(p1)
  l3 = np.array(p4) - np.array(p1)
  
  # Check for intersection of bounding box 
  b1 = [min(p1[0],p2[0]),min(p1[1],p2[1])]
  b2 = [max(p1[0],p2[0]),max(p1[1],p2[1])]
  b3 = [min(p3[0],p4[0]),min(p3[1],p4[1])]
  b4 = [max(p3[0],p4[0]),max(p3[1],p4[1])]
  
  boundingBoxIntersect = (b1[0] <= b4[0]) and (b2[0] >= b3[0]) and (b1[1] <= b4[1]) and (b2[1] >= b3[1])
  return ((l1[0]*l2[1]-l1[1]*l2[0])*(l3[0]*l2[1]-l3[1]*l2[0]) < 0) and boundingBoxIntersect

def self_intersect(x,y):
  assert(isinstance(x,(list,np.ndarray)))
  assert(isinstance(y,(list,np.ndarray)))
  assert(x.ndim==1 and y.ndim==1 and len(x)==len(y))
  
  # Make sure there are no repeated points
  points = (np.array([x,y]).T).tolist()
  assert (not any(points.count(z) > 1 for z in points))
  # Repeat last point
  points.append(points[0])
  
  npoints = len(points)
  for i in range(npoints-1):
    p1 = points[i]
    p2 = points[i+1]
    # Compare with all line segments that do not contain x[i] or x[i+1]
    for j in range(i+2,npoints-1):
      p3 = points[j]
      p4 = points[j+1]
      # Ignore if any points are in common with p1
      if (p1==p3 or p2==p3 or p1==p4 or p2==p4):
        continue
      if (segment_intersect(p1,p2,p3,p4)):
        return True
  return False
      
      
