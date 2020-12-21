import numpy as np
import os
from vmec_output import VmecOutput
from vmec_input import VmecInput
import errno
from scipy import interpolate
import copy
from scipy.io import netcdf
from mpi4py import MPI
from surface_utils import init_modes

def axis_weight(s):
    return np.exp(-(s-0.1)**2/0.05**2)

class VmecOptimization:
    """
    A class used for optimization of a VMEC equilibrium

    ...
    Attributes
    ---------
    vmecInputFilename (str) : Fortran namelist for initial equilibrium
        evaluation. This namelist will be modified as the boundary is modified.
    callVMEC_function (function) : takes vmecInputFilename (str) as sole input
        argument and returns runs VMEC with given Fortran namelist. Returns 
        error code (nonzero = error). See call_vmec.py for examples. 
    name (str) : Name used as prefix for directory names of new VMEC
        evaluations.
    callANIMEC_function (function) : takes vmecInputFilename (str) as sole input
        argument and returns runs ANIMEC with given Fortran namelist. Returns 
        error code (nonzero = error). See call_vmec.py for examples.     
    mmax_sensitivity (int) : maximum poloidal mode number of boundary to
        optimize
    nmax_sensitivity (int) : maximum toroidal mode number of boundary to
        optimize
    delta_curr (float) : defines size of current perturbation for adjoint shape
        gradient calculation
    delta_pres (float) : defines size of pressure perturbation for adjoint shape
        gradient calculation
    woutFilename (nc filename) : VMEC outputfile for initialization of
        optimization
    ntheta (int) : number of poloidal grid points 
    nzeta (int) : number of toroidal grid points per period
    verbose (boolean) : if True, more output is printed
    xn_sensitivity (int array) : toroidal mode numbers for optimization
    xm_sensitivity (int array) : poloidal mode numbers for optimizationm. Must
        be same size as xn_sensitivity. 
    """
    def __init__(self,vmecInputFilename,callVMEC_function=None,name='optimization',
             callANIMEC_function=None,mmax_sensitivity=0,nmax_sensitivity=0,
             delta_curr=10,delta_pres=10,woutFilename=None,ntheta=100,nzeta=100,
             verbose=True,xn_sensitivity=None,xm_sensitivity=None):
        if (xn_sensitivity is None or xm_sensitivity is None):
            [mnmax_sensitivity,xm_sensitivity,xn_sensitivity] = \
                init_modes(mmax_sensitivity,nmax_sensitivity)
            self.mmax_sensitivity = mmax_sensitivity
            self.nmax_sensitivity = nmax_sensitivity
        else:
            assert(len(xn_sensitivity)==len(xm_sensitivity))
            self.mmax_sensitivity = np.max(xm_sensitivity)
            self.nmax_sensitivity = np.max(xn_sensitivity)
            mnmax_sensitivity = len(xn_sensitivity)

        self.mnmax_sensitivity = mnmax_sensitivity
        self.xm_sensitivity = xm_sensitivity
        self.xn_sensitivity = xn_sensitivity
        self.callVMEC_function = callVMEC_function
        self.callANIMEC_function = callANIMEC_function
        self.counter = 0 # Counts number of function evals (used for
            # naming directories)
        self.name = name # Name used for directories
        self.vmecInputFilename = vmecInputFilename
        self.directory = os.getcwd()
        self.delta_pres = delta_pres
        self.delta_curr = delta_curr
        self.ntheta = ntheta
        self.nzeta = nzeta
        self.jacobian_threshold = 1e-4
        self.verbose = verbose
        self.vmec_objectives = ['iota','iota_prime','iota_target','well',
                                'well_ratio','volume','area','jacobian',
                                'radius','normalized_jacobian','modB_vol',
                                'axis_ripple']
        self.input_objectives = ['volume','area','jacobian','radius',
                                 'normalized_jacobian','summed_proximity',
                                 'summed_radius','curvature',
                                 'curvature_integrated']

        self.vmecInputObject = VmecInput(vmecInputFilename,ntheta,nzeta,
                                        verbose=verbose)
        self.nfp = self.vmecInputObject.nfp
        mpol_input = self.vmecInputObject.mpol
        ntor_input = self.vmecInputObject.ntor
        mnmax_input = self.vmecInputObject.mnmax
        rbc_input = self.vmecInputObject.rbc
        zbs_input = self.vmecInputObject.zbs
        xm_input = self.vmecInputObject.xm
        xn_input = self.vmecInputObject.xn

        if (mpol_input<2):
            raise ValueError('Error! mpol must be > 1.')

        # mpol, ntor for calculaitons must be large enough for sensitivity
            #required
        self.mmax = max(self.mmax_sensitivity,mpol_input-1)
        self.nmax = max(self.nmax_sensitivity,ntor_input)
        [self.mnmax,self.xm,self.xn] = init_modes(self.mmax,self.nmax)

        # Update input object with required mpol and ntor
        self.vmecInputObject.update_modes(self.mmax+1,self.nmax)

        rmnc = np.zeros(self.mnmax)
        zmns = np.zeros(self.mnmax)
        rmnc_opt = np.zeros(self.mnmax_sensitivity)
        zmns_opt = np.zeros(self.mnmax_sensitivity)
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
        # Remove 0,0 mode from zmns
        cond = np.logical_not(np.logical_and(xm_sensitivity == 0,
                                             xn_sensitivity==0))
        zmns_opt = zmns_opt[cond]
        self.boundary_opt = np.hstack((rmnc_opt,zmns_opt))
        self.norm_normal = None

        self.vmecOutputObject = None

        if (woutFilename is not None):
            self.vmecOutputObject = VmecOutput(woutFilename,self.ntheta,
                                                   self.nzeta)

    def evaluate_vmec(self,boundary=None,It=None,pres=None,update=True):
        """
        Evaluates VMEC equilibrium with specified boundary, pressure, or
            toroidal current profile update

        boundary, It, and pres should not be specified simultaneously.
        If neither are specified, then an exception is raised.

        Parameters
        ----------
        boundary (float array) : values of boundary harmonics for VMEC
            evaluations
        It (float array) : value of toroidal current on half grid VMEC mesh for
            prescription of profile with "ac_aux_f" Fortran namelist variable
        pres (float array) : value of pressure on half grid VMEC mesh for
            prescription of profile with "am_aux_f" Fortran namelist variable
        update (bool) : if True, self.vmecOutputObject is updated and boundary
            is assigned to self.boundary_opt

        Returns
        ----------
        exit_code (integer) : code returned by VMEC evaluation. non-zero
            indicates error.
        vmecOutput_new (VmecOutput) : object corresponding to output 
            of VMEC evaluations
        """
        self.counter += 1
        if (boundary is not None and update):
            # Update boundary
            self.update_boundary_opt(boundary)
        rank = MPI.COMM_WORLD.Get_rank()
        if (os.getcwd()!=self.directory):
            if (self.verbose):
                print('''evaluate_vmec called from incorrect directory.
                      Changing to '''+self.directory)
            os.chdir(self.directory)
        if (np.count_nonzero([It,pres,boundary] is not None)>1):
            raise RuntimeError('''evaluate_vmec called with more than one type of
                  perturbation (boundary, It, pres). This behavior is not
                  supported.''')
        if (rank == 0):
            if (boundary is not None):
            # Update boundary
                directory_name = self.name+"_"+str(self.counter)
            elif (It is not None):
                directory_name = "delta_curr_"+str(self.counter)
            elif (pres is not None):
                directory_name = "delta_pres_"+str(self.counter)
            else:
                raise RuntimeError('''Error! evaluate_vmec called when 
                  equilibrium does not need to be evaluated.''')
            # Make directory for VMEC evaluations
            try:
                os.mkdir(directory_name)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    if (self.verbose):
                        print('Warning. Directory '+directory_name+\
                          ' already exists. Contents may be overwritten.')
                    raise
                pass

            input_file = "input."+directory_name

            inputObject_new = copy.deepcopy(self.vmecInputObject)
            inputObject_new.input_filename = input_file
            inputObject_new.ntor = self.nmax
            inputObject_new.mpol = self.mmax+1
            # Edit input filename with boundary
            if (It is not None):
                curtor = 1.5*It[-1] - 0.5*It[-2]
                s_half = self.vmecOutputObject.s_half
                if (self.vmecOutputObject.ns>101):
                    s_spline = np.linspace(0,1,101)
                    ds_spline = s_spline[1]-s_spline[0]
                    s_spline_half = s_spline - 0.5*ds_spline
                    s_spline_half = np.delete(s_spline_half,0)
                    It_spline = \
                        interpolate.InterpolatedUnivariateSpline(s_half,It)
                    It = It_spline(s_spline_half)
                    s_half = s_spline_half
                inputObject_new.curtor = curtor
                inputObject_new.ac_aux_f = list(It)
                inputObject_new.ac_aux_s = list(s_half)
                inputObject_new.pcurr_type = "line_segment_I"
            if (pres is not None):
                s_half = self.vmecOutputObject.s_half
                if (self.vmecOutputObject.ns>101):
                    s_spline = np.linspace(0,1,101)
                    ds_spline = s_spline[1]-s_spline[0]
                    s_spline_half = s_spline - 0.5*ds_spline
                    s_spline_half = np.delete(s_spline_half,0)
                    pres_spline = \
                        interpolate.InterpolatedUnivariateSpline(s_half,pres)
                    pres = pres_spline(s_spline_half)
                    s_half = s_spline_half
                inputObject_new.am_aux_f = list(pres)
                inputObject_new.am_aux_s = list(s_half)
                inputObject_new.pmass_type = "cubic_spline"
            if (boundary is not None):
                inputObject_new.rbc = self.boundary[0:self.mnmax]
                inputObject_new.zbs = self.boundary[self.mnmax::]
        else:
            inputObject_new = None
            directory_name = None
            input_file = None

        # Call VMEC with revised input file
        inputObject_new = MPI.COMM_WORLD.bcast(inputObject_new,root=0)
        directory_name = MPI.COMM_WORLD.bcast(directory_name,root=0)
        input_file = MPI.COMM_WORLD.bcast(input_file,root=0)

        os.chdir(directory_name)
        exit_code = self.call_vmec(inputObject_new)

        if (exit_code == 0):
            # Read from new equilibrium
            outputFileName = "wout_"+input_file[6::]+".nc"
            if (rank == 0):
                vmecOutput_new = \
                    VmecOutput(outputFileName,self.ntheta,self.nzeta)
            else:
                vmecOutput_new = None
            vmecOutput_new = MPI.COMM_WORLD.bcast(vmecOutput_new,root=0)
            if (boundary is not None and update):
                self.vmecOutputObject = vmecOutput_new
                self.vmecInputObject = inputObject_new
        else:
            vmecOutput_new = None

        os.chdir("..")
        
        return exit_code, vmecOutput_new

    def evaluate_animec(self,boundary=None,pres=None,It=None,bcrit=1):
        """
        Evaluates ANIVMEC equilibrium with specified boundary pressure
            profile update

        boundary and pres should not be specified simultaneously.
        If neither are specified, then an exception is raised.

        Parameters
        ----------
        boundary (float array) : values of boundary harmonics for VMEC
            evaluations
        pres (float array) : value of hot pressure on half grid VMEC mesh for
            prescription of profile with "ah_aux_f" Fortran namelist variable
        It (float array) : value of toroidal current on half grid VMEC mesh
            for prescription of profile with "ac_aux_f" Fortran namelist 
            variable.
        bcrit (float) : defines the BCRIT parameter in ANIMEC namelist. Sets the
            scale of the perturbation of the pressure tensor. 

        Returns
        ----------
        exit_code (integer) : code returned by VMEC evaluation. non-zero
            indicates error.
        vmecOutput_new (VmecOutput) : object corresponding to output of ANIMEC
            evaluations
        """
        self.counter += 1
        rank = MPI.COMM_WORLD.Get_rank()
        if (rank == 0):
            if (os.getcwd()!=self.directory):
                if (self.verbose):
                    print('''evaluate_animec called from incorrect directory. 
                          Changing to '''+self.directory)
                os.chdir(self.directory)
            if (np.count_nonzero([pres,boundary] is not None)>1):
                raise RuntimeError('''evaluate_animec called with more than one 
                  type of perturbation (boundary, pres). This behavior is not
                  supported.''')
            if (boundary is not None):
            # Update boundary
                self.update_boundary_opt(boundary)
                directory_name = self.name+"_"+str(self.counter)
            elif (pres is not None):
                directory_name = "delta_pres_"+str(self.counter)
            elif (self.counter == 0):
                directory_name = self.name+"_"+str(self.counter)
            else:
                raise RuntimeError('''Error! evaluate_animec called when
                  equilibrium does not need to be evaluated.''')
            # Make directory for ANIMEC evaluations
            try:
                os.mkdir(directory_name)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    if (self.verbose):
                        print('Warning. Directory '+directory_name+\
                          ' already exists. Contents may be overwritten.')
                    raise
                pass

            input_file = "input."+directory_name

            inputObject_new = copy.deepcopy(self.vmecInputObject)
            inputObject_new.input_filename = input_file
            inputObject_new.ntor = self.nmax
            inputObject_new.mpol = self.mmax+1
            # Edit input filename with boundary
            if (pres is not None):
                s_half = self.vmecOutputObject.s_half
                if (self.vmecOutputObject.ns>101):
                    s_spline = np.linspace(0,1,101)
                    ds_spline = s_spline[1]-s_spline[0]
                    s_spline_half = s_spline - 0.5*ds_spline
                    s_spline_half = np.delete(s_spline_half,0)
                    pres_spline = \
                        interpolate.InterpolatedUnivariateSpline(s_half,pres)
                    pres = pres_spline(s_spline_half)
                    s_half = s_spline_half
                inputObject_new.animec = True
                inputObject_new.ah_aux_f = list(pres)
                inputObject_new.ah_aux_s = list(s_half)
                inputObject_new.photp_type = "line_segment"
                inputObject_new.bcrit = bcrit
            if (It is not None):
                curtor = 1.5*It[-1] - 0.5*It[-2]
                s_half = self.vmecOutputObject.s_half
                if (self.vmecOutputObject.ns>101):
                    s_spline = np.linspace(0,1,101)
                    ds_spline = s_spline[1]-s_spline[0]
                    s_spline_half = s_spline - 0.5*ds_spline
                    s_spline_half = np.delete(s_spline_half,0)
                    It_spline = \
                        interpolate.InterpolatedUnivariateSpline(s_half,It)
                    It = It_spline(s_spline_half)
                    s_half = s_spline_half
                inputObject_new.curtor = curtor
                inputObject_new.ac_aux_f = list(It)
                inputObject_new.ac_aux_s = list(s_half)
                inputObject_new.pcurr_type = "line_segment_I"
            if (boundary is not None):
                inputObject_new.rbc = self.boundary[0:self.mnmax]
                inputObject_new.zbs = self.boundary[self.mnmax::]
        else:
            inputObject_new = None
            directory_name = None
            input_file = None

        # Call VMEC with revised input file
        inputObject_new = MPI.COMM_WORLD.bcast(inputObject_new,root=0)
        directory_name = MPI.COMM_WORLD.bcast(directory_name,root=0)
        input_file = MPI.COMM_WORLD.bcast(input_file,root=0)

        os.chdir(directory_name)
        exit_code = self.call_vmec(inputObject_new,animec=True)

        if (exit_code == 0):
            # Read from new equilibrium
            outputFileName = "wout_"+input_file[6::]+".nc"
            vmecOutput_new = \
                VmecOutput(outputFileName,self.ntheta,self.nzeta)
        else:
            vmecOutput_new = None

        os.chdir("..")
        return exit_code, vmecOutput_new

    def update_boundary_opt(self,boundary_opt_new):
        """
        Updates boundary harmonics at new evaluation point

        Parameters
        ----------
        boundary_opt_new (float array) : new boundary to evaluate
            (same size as self.boundary_opt)
        """
        if (len(boundary_opt_new)!=len(self.boundary_opt)):
            raise ValueError('Incorrect size of boundary_opt_new.')
        rmnc_opt_new = boundary_opt_new[0:self.mnmax_sensitivity]
        zmns_opt_new = boundary_opt_new[self.mnmax_sensitivity::]
        # m = 0, n = 0 mode excluded in boundary_opt
        if (np.min(self.xm_sensitivity)==0 and 
                np.min(np.abs(self.xn_sensitivity)) == 0):
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

    def input_objective(self,boundary=None,which_objective='volume',\
                                 update=True,which_kappa=1,**kwargs):
        """
        Evaluates objective from VMEC input namelist. 
        
        Parameters
        ----------
        boundary (np array) : coefficients defining boundary in the format of 
            self.boundary_opt. If None, objective
            will be computed from self.vmecOutputObject
        which_objective (string) : string defining which input objective. Must
            be in self.input_objectives
        update (boolean) : If True, self.vmecInputObject will be updated.
        which_kappa (int) : (1 or 2) Only relevant for 'curvature_integrated'.
            Indicates which principal curvature for evaluation.
        **kwargs (dict) : additional keyword arguments to pass to objective 
            function. See input.py for details. 
      
        Returns
        ----------
        objective_function (float) : value of objective function

        """
        if (which_objective not in self.input_objectives):
            raise ValueError('''incorrect value of which_objective in
                  input_objective.''')
        elif self.verbose:
            print("Evaluating ",which_objective," objective.")
        if (boundary is None):
            boundary = self.boundary_opt
        if (np.shape(boundary)!=np.shape(self.boundary_opt)):
            raise ValueError('''Error! input_objective called with 
                incorrect boundary shape.''')

        if ((np.copy(boundary) != self.boundary_opt).any() 
            or self.vmecInputObject is None):
            # Save old boundary if update=False
            if (update==False):
                boundary_old = np.copy(self.boundary_opt)
            # Update equilibrium count
            self.counter += 1
            # Update boundary
            self.update_boundary_opt(boundary)
            directory_name = self.name+"_"+str(self.counter)
            if (self.verbose):
                print("Creating new input object in "+directory_name)
            # Make directory for VMEC evaluations
            try:
                os.mkdir(directory_name)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    if (self.verbose):
                        print('Warning. Directory '+directory_name+\
                          ' already exists. Contents may be overwritten.')
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
            [x,y,z,R] = vmecInputObject.position()
            objective_function = R
        elif (which_objective == 'normalized_jacobian'):
            objective_function = vmecInputObject.normalized_jacobian()
        elif (which_objective == 'summed_proximity'):
            objective_function = vmecInputObject.summed_proximity(**kwargs)
        elif (which_objective == 'summed_radius'):
            objective_function = \
                vmecInputObject.summed_radius_constraint(**kwargs)
        elif (which_objective == 'curvature'):
            objective_function = \
                vmecInputObject.surface_curvature_metric(**kwargs)
        elif (which_objective == 'curvature_integrated'):
            [metric1,metric2] = \
                vmecInputObject.surface_curvature_metric_integrated(**kwargs)
            if (which_kappa == 1):
                objective_function = metric1
            elif (which_kappa == 2):
                objective_function = metric2
            else:
                raise ValueError('Incorrect value of which_kappa')
        return objective_function

    def vmec_objective(self,boundary=None,which_objective='iota',
                                update=True,**kwargs):
        """
        Evaluates vmec objective function with prescribed boundary

        Parameters
        ----------
        boundary (np array) : coefficients defining boundary in the format of 
            self.boundary_opt. If None, objective
            will be computed from self.vmecOutputObject
        which_objective (string) : string defining which vmec objective. Must
            be in self.vmec_objectives
        update (boolean) : If True, self.vmecOutputObject will be updated.
        **kwargs (dict) : additional keyword arguments to pass to objective 
            function. See output.py for details. 

        Returns
        ----------
        objective function (float) : value of objective function defined through
            which_objective and weight_function. If VMEC completes with an
            error, the value of the objective function is set to 1e12.
        """
        if (which_objective not in self.vmec_objectives):
            raise ValueError('''Error! vmec_objectives called with incorrect value 
                  of which_objective''')
        elif self.verbose:
            print("Evaluating ",which_objective," objective.")
        if (boundary is None):
            boundary = self.boundary_opt
        if (np.shape(boundary)!=np.shape(self.boundary_opt)):
            raise ValueError('''Error! vmec_objectives called with incorrect
                             boundary shape.''')
        # Save old boundary if update=False
        if (update==False):
            boundary_old = np.copy(self.boundary_opt)
        # Evaluate new equilibrium if necessary
        error_code = 0
        if ((boundary is not None and 
             (np.copy(boundary)!=self.boundary_opt).any()) 
                or self.vmecOutputObject is None):
            [error_code,vmecOutputObject] = self.evaluate_vmec(\
                                                boundary=boundary,update=update)
            if (error_code != 0 and self.verbose):
                print('''VMEC returned with an error when evaluating new
                    boundary in vmec_objective.''')
                print('Objective function will be set to 1e12')
        else:
            vmecOutputObject = self.vmecOutputObject
        
        # Reset old boundary
        if (update==False):
            self.update_boundary_opt(boundary_old)

        if (error_code == 0):
            if (which_objective == 'iota'):
                objective_function = \
                    vmecOutputObject.iota_objective(**kwargs)
            elif (which_objective == 'iota_target'):
                objective_function = \
                    vmecOutputObject.iota_target_objective(**kwargs)
            elif (which_objective == 'iota_prime'):
                objective_function = \
                    vmecOutputObject.iota_prime_objective(**kwargs)
            elif (which_objective == 'well'):
                objective_function = \
                    vmecOutputObject.well_objective(**kwargs)
            elif (which_objective == 'well_ratio'):
                objective_function = \
                    vmecOutputObject.well_ratio_objective(**kwargs)
            elif (which_objective == 'volume'):
                objective_function = vmecOutputObject.volume
            elif (which_objective == 'area'):
                objective_function = \
                    vmecOutputObject.area(vmecOutputObject.ns-1)
            elif (which_objective == 'jacobian'):
                objective_function = vmecOutputObject.jacobian(-1)
            elif (which_objective == 'radius'):
                [x,y,z,R] = vmecOutputObject.position(-1)
                objective_function = R
            elif (which_objective == 'normalized_jacobian'):
                objective_function = vmecOutputObject.normalized_jacobian(-1)
            elif (which_objective == 'modB'):
                objective_function = vmecOutputObject.modB_objective()
            elif (which_objective == 'modB_vol'):
                objective_function = \
                    vmecOutputObject.modB_objective_volume()
            elif (which_objective == 'axis_ripple'):
                objective_function = \
                    vmecOutputObject.ripple_objective(**kwargs)
            return objective_function
        else:
            return 1e12

    def vmec_shape_gradient(self,boundary=None,vmecOutputObject=None,
                                which_objective='iota',update=True,
                                weight_function=axis_weight,iota_target=None,
                                weight_function1=axis_weight,
                                weight_function2=axis_weight):
        """
        Evaluates shape gradient for objective function

        Parameters
        ----------
        boundary (np array) : coefficients defining boundary in the format of 
            self.boundary_opt. If None, objective
            will be computed from self.vmecOutputObject
        vmecOutputObject (VmecOutput) : instance of VmecOuput for evaluation of 
            objective. If None, self.vmecOutputObject will be used. 
        which_objective (string) : Which objective to compute shape gradient
            of. Must be in self.vmec_objectives.
        update (boolean) : If True, self.vmecOutputObject will be updated.
        weight_function (function) : weight function w(s) which takes the 
            normalized flux as input and returns a scalar (see axis_weight
            above for example). This is only used for iota, iota_prime, 
            iota_target, well, and axis_ripple targets.             
        iota_target (np array or float) : target iota profile. If np array,
            must be the same length as the half grid. Only used for
            iota_target object. 
        weight_function1 (function) : weight function w(s) which takes the 
            normalized flux as input and returns a scalar (see axis_weight
            above for example). This only used for the well_ratio objective.
        weight_function2 (function) : weight function w(s) which takes the 
            normalized flux as input and returns a scalar (see axis_weight
            above for example). This only used for the well_ratio objective.

        Returns
        ----------
        shape_gradient (np array) : shape gradient defined on nzeta,ntheta
            grid of specified objective function
        """
        if (which_objective not in self.vmec_objectives):
            raise ValueError('''incorrect value of which_objective in
                  vmec_shape_gradient.''')
        if (self.verbose):
            print("Evaluating "+which_objective+" shape gradient.")
            
        if (which_objective in ['iota','iota_prime','iota_target']):
            delta = self.delta_curr
        elif (which_objective in ['well','well_ratio','axis_ripple']):
            delta = self.delta_pres
        elif (which_objective not in ['area','modB_vol']):
            raise ValueError('''Error! vmec_shape_gradient called with incorrect 
                value of '''+which_objective)

        if (boundary is not None):
            if (np.size(boundary)!=np.size(self.boundary_opt)):
                raise ValueError('''Error! vmec_shape_gradient called with incorrect
                                 boundary shape.''')

        # Evaluate base equilibrium if necessary
        if (vmecOutputObject is None and boundary is not None \
            and np.any(np.copy(boundary) != self.boundary_opt)):
            [error_code, vmecOutputObject] = self.evaluate_vmec(\
                                                boundary=boundary,update=update)
            if (error_code != 0):
                raise RuntimeError('''Unable to evaluate base VMEC equilibrium 
                    in vmec_shape_gradient.''')
        elif (vmecOutputObject is None):
            vmecOutputObject = self.vmecOutputObject

        if (which_objective == 'iota'):
            if (weight_function is None):
                raise RuntimeError('''weight_function must be specified when 
                                   computing iota objective''')
            [Bx, By, Bz, theta_arclength] = \
                vmecOutputObject.B_on_arclength_grid()
            It_half = vmecOutputObject.current()
            It_new = It_half + \
                delta*weight_function(self.vmecOutputObject.s_half)

            [error_code, vmecOutput_delta] = self.evaluate_vmec(It=It_new)
            if (error_code != 0):
                raise RuntimeError('''Unable to evaluate VMEC equilibrium with 
                    current perturbation in vmec_shaep_gradient.''')

            [Bx_delta, By_delta, Bz_delta, theta_arclength_delta] = \
                vmecOutput_delta.B_on_arclength_grid()
            for izeta in range(self.vmecOutputObject.nzeta):
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],Bx_delta[izeta,:])
                Bx_delta[izeta,:] = f(theta_arclength[izeta,:])
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],By_delta[izeta,:])
                By_delta[izeta,:] = f(theta_arclength[izeta,:])
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],Bz_delta[izeta,:])
                Bz_delta[izeta,:] = f(theta_arclength[izeta,:])

            deltaB_dot_B = ((Bx_delta-Bx)*Bx + (By_delta-By)*By + \
                            (Bz_delta-Bz)*Bz)/delta

            shape_gradient = deltaB_dot_B/(2*np.pi*self.vmecOutputObject.mu0)
        elif (which_objective == 'iota_target'):
            if (weight_function is None):
                raise RuntimeError('''weight_function must be specified when 
                                   computing iota_target objective''')
            [Bx, By, Bz, theta_arclength] = \
                vmecOutputObject.B_on_arclength_grid()
            It_half = vmecOutputObject.current()

            weight_half = weight_function(self.vmecOutputObject.s_half)
            iota = self.vmecOutputObject.iota
            if (iota_target is None):
                iota_target = np.zeros(np.shape(self.vmecOutputObject.s_half))

            perturbation = weight_half*(iota-iota_target)\
                /(self.vmecOutputObject.psi[-1]*self.vmecOutputObject.sign_jac)

            It_new = It_half + delta*perturbation

            [error_code, vmecOutput_delta] = self.evaluate_vmec(It=It_new)
            if (error_code != 0):
                raise RuntimeError('''Unable to evaluate VMEC equilibrium with 
                    current perturbation in vmec_shape_gradient.''')

            [Bx_delta, By_delta, Bz_delta, theta_arclength_delta] = \
                vmecOutput_delta.B_on_arclength_grid()
            for izeta in range(self.vmecOutputObject.nzeta):
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],Bx_delta[izeta,:])
                Bx_delta[izeta,:] = f(theta_arclength[izeta,:])
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],By_delta[izeta,:])
                By_delta[izeta,:] = f(theta_arclength[izeta,:])
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],Bz_delta[izeta,:])
                Bz_delta[izeta,:] = f(theta_arclength[izeta,:])

            deltaB_dot_B = ((Bx_delta-Bx)*Bx + (By_delta-By)*By + \
                            (Bz_delta-Bz)*Bz)/delta

            shape_gradient = deltaB_dot_B/(2*np.pi*self.vmecOutputObject.mu0)

        elif (which_objective == 'iota_prime'):
            if (weight_function is None):
                raise RuntimeError('''weight_function must be specified when 
                                   computing iota_prime objective''')
            [Bx, By, Bz, theta_arclength] = \
                vmecOutputObject.B_on_arclength_grid()
            It_half = vmecOutputObject.current()

            weight_half = weight_function(self.vmecOutputObject.s_half)
            weight_full = weight_function(self.vmecOutputObject.s_full)
            # Derivative of weight function on half grid
            weight_prime = \
                (weight_full[1::]-weight_full[0:-1])/self.vmecOutputObject.ds
            # Derivative of iota on half grid
            iota_prime_half = (self.vmecOutputObject.iotaf[1::]
                     -self.vmecOutputObject.iotaf[0:-1]) \
                     /self.vmecOutputObject.ds
            # Derivative of iota on full grid
            iota_prime_full = np.zeros(self.vmecOutputObject.ns_half,1)
            iota_prime_full[1:-1] = (self.vmecOutputObject.iota[1::] 
                                     -self.vmecOutputObject.iota[0:-1]) \
                                 /self.vmecOutputObject.ds
            iota_prime_full[0] = 1.5*self.vmecOutputObject.iota[0] \
                            -0.5*self.vmecOutputObject.vmecOutputObject.iota[1]
            iota_prime_full[-1] = 1.5*self.vmecOutputObject.iota[-1] \
                            -0.5*self.vmecOutputObject.iota[-2]
            # Second derivative on half grid
            iota_prime_prime_half = (iota_prime_full[1::]\
                                -iota_prime_full[0:-1])/self.vmecOutputObject.ds

            perturbation = -(iota_prime_prime_half*weight_half \
                            + iota_prime_half*weight_prime)

            It_new = It_half + delta*perturbation

            [error_code, vmecOutput_delta] = self.evaluate_vmec(It=It_new)
            if (error_code != 0):
                raise RuntimeError('''Unable to evaluate VMEC equilibrium with 
                    current perturbation in vmec_shaep_gradient.''')

            [Bx_delta, By_delta, Bz_delta, theta_arclength_delta] = \
                vmecOutput_delta.B_on_arclength_grid()
            for izeta in range(self.vmecOutputObject.nzeta):
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],Bx_delta[izeta,:])
                Bx_delta[izeta,:] = f(theta_arclength[izeta,:])
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],By_delta[izeta,:])
                By_delta[izeta,:] = f(theta_arclength[izeta,:])
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],Bz_delta[izeta,:])
                Bz_delta[izeta,:] = f(theta_arclength[izeta,:])

            deltaB_dot_B = ((Bx_delta-Bx)*Bx + (By_delta-By)*By + \
                            (Bz_delta-Bz)*Bz)/delta

            shape_gradient = deltaB_dot_B/(2*np.pi*self.vmecOutputObject.mu0)           
            
        elif (which_objective == 'well_ratio'):  
            if (weight_function1 is None or weight_function2 is None):
                raise RuntimeError('''weight_function1 and weight_function2 must 
                            be specified when computing well_ratio objective''')
            [Bx, By, Bz, theta_arclength] = \
                vmecOutputObject.B_on_arclength_grid()
            pres = vmecOutputObject.pres
            
            numerator = np.sum(weight_function1(self.vmecOutputObject.s_half) 
                                   * self.vmecOutputObject.vp) 
            denominator = np.sum(weight_function2(self.vmecOutputObject.s_half)
                                  * self.vmecOutputObject.vp) 
            fW = numerator/denominator

            perturbation = (weight_function1(self.vmecOutputObject.s_half) 
                - fW * weight_function2(self.vmecOutputObject.s_half)) \
                / (denominator * self.vmecOutputObject.ds * 4 * np.pi * np.pi)
            pres_new = pres + delta*perturbation
            
            [error_code, vmecOutput_delta] = self.evaluate_vmec(pres=pres_new)
            if (error_code != 0):
                raise RuntimeError('''Unable to evaluate VMEC equilibrium with 
                    pressure perturbation in vmec_shaep_gradient.''')

            [Bx_delta, By_delta, Bz_delta, theta_arclength_delta] = \
                vmecOutput_delta.B_on_arclength_grid()
            for izeta in range(self.vmecOutputObject.nzeta):
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],Bx_delta[izeta,:])
                Bx_delta[izeta,:] = f(theta_arclength[izeta,:])
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],By_delta[izeta,:])
                By_delta[izeta,:] = f(theta_arclength[izeta,:])
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],Bz_delta[izeta,:])
                Bz_delta[izeta,:] = f(theta_arclength[izeta,:])

            deltaB_dot_B = ((Bx_delta-Bx)*Bx + (By_delta-By)*By + \
                            (Bz_delta-Bz)*Bz)/delta
            shape_gradient = deltaB_dot_B/(self.vmecOutputObject.mu0) \
                            + perturbation[-1]
            
        elif (which_objective == 'well'):   
            if (weight_function is None):
                raise RuntimeError('''weight_function must be specified when 
                                   computing well objective''')
            [Bx, By, Bz, theta_arclength] = \
                vmecOutputObject.B_on_arclength_grid()
            pres = vmecOutputObject.pres
            pres_new = pres + \
                delta*weight_function(self.vmecOutputObject.s_half) \
                / (self.vmecOutputObject.psi[-1])
            [error_code, vmecOutput_delta] = self.evaluate_vmec(pres=pres_new)
            if (error_code != 0):
                raise RuntimeError('''Unable to evaluate VMEC equilibrium with
                     pressure perturbation in vmec_shaep_gradient.''')

            [Bx_delta, By_delta, Bz_delta, theta_arclength_delta] = \
                vmecOutput_delta.B_on_arclength_grid()
            for izeta in range(self.vmecOutputObject.nzeta):
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],Bx_delta[izeta,:])
                Bx_delta[izeta,:] = f(theta_arclength[izeta,:])
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],By_delta[izeta,:])
                By_delta[izeta,:] = f(theta_arclength[izeta,:])
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],Bz_delta[izeta,:])
                Bz_delta[izeta,:] = f(theta_arclength[izeta,:])

            deltaB_dot_B = ((Bx_delta-Bx)*Bx + (By_delta-By)*By + \
                            (Bz_delta-Bz)*Bz)/delta
            shape_gradient = deltaB_dot_B/(self.vmecOutputObject.mu0) \
                            + weight_function(1)/(self.vmecOutputObject.psi[-1])  

        elif (which_objective == 'axis_ripple'):
            if (weight_function is None):
                raise RuntimeError('''weight_function must be specified when 
                                   computing axis_ripple objective''')
            [Bx, By, Bz, theta_arclength] = \
                vmecOutputObject.B_on_arclength_grid()
            
            pperp_end = vmecOutputObject.ripple_pperp(weight_function)
            
            pres_new = weight_function(self.vmecOutputObject.s_half)
            
            if (self.vmecInputObject.ncurr==1):
                delta_curr = np.zeros(np.shape(self.vmecOutputObject.s_half))
                Bbar = vmecOutputObject.ripple_Bbar(weight_function)
                normalization = \
                    vmecOutputObject.ripple_normalization(weight_function)
                ripple = vmecOutputObject.ripple_axis(weight_function)
                for isurf in range(self.vmecOutputObject.ns_half):
                    B = vmecOutputObject.modB(isurf = isurf)
                    [Bsubtheta, Bsubzeta] = \
                        vmecOutputObject.B_covariant(isurf=isurf)
                    delta_curr[isurf] = \
                        np.sum(((B - Bbar)/B - 0.5*ripple)*Bsubtheta) \
                        * vmecOutputObject.dtheta * vmecOutputObject.dzeta \
                        * vmecOutputObject.nfp
                delta_curr *= self.delta_curr \
                    * weight_function(vmecOutputObject.s_half) \
                    / (0.5 * normalization)
                It_half = vmecOutputObject.current()
                It_new = delta_curr + It_half

            [error_code, vmecOutput_delta] = self.evaluate_animec(pres=pres_new,\
                                                bcrit=delta,It=It_new)
            
            if (error_code != 0):
                raise RuntimeError('''Unable to evaluate VMEC equilibrium with 
                    pressure perturbation in vmec_shape_gradient.''')

            [Bx_delta, By_delta, Bz_delta, theta_arclength_delta] = \
                vmecOutput_delta.B_on_arclength_grid()
            for izeta in range(self.vmecOutputObject.nzeta):
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],Bx_delta[izeta,:])
                Bx_delta[izeta,:] = f(theta_arclength[izeta,:])
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],By_delta[izeta,:])
                By_delta[izeta,:] = f(theta_arclength[izeta,:])
                f = interpolate.InterpolatedUnivariateSpline(\
                            theta_arclength_delta[izeta,:],Bz_delta[izeta,:])
                Bz_delta[izeta,:] = f(theta_arclength[izeta,:])
                

            deltaB_dot_B = ((Bx_delta-Bx)*Bx + (By_delta-By)*By + \
                            (Bz_delta-Bz)*Bz)/delta
            
            shape_gradient = deltaB_dot_B + pperp_end
            
        elif (which_objective == 'area'):
            shape_gradient = \
                vmecOutputObject.mean_curvature(vmecOutputObject.ns-1)
        elif (which_objective == 'modB_vol'):
            modB = vmecOutputObject.modB(isurf=vmecOutputObject.ns, 
                                                 full=True)
            shape_gradient = 0.5 * modB * modB

        return shape_gradient

    def input_objective_grad(self,boundary=None,\
                                      which_objective='volume',update=True,\
                                      which_kappa=1,**kwargs):
        """
        Evaluates gradient of input objective

        Parameters
        ----------
        boundary (np array) : coefficients defining boundary in the format of 
            self.boundary_opt. If None, objective
            will be computed from self.vmecInputObject
        which_objective (string) : Which objective to compute shape gradient
            of. Must be in self.input_objectives.
        update (boolean) : If True, self.vmecInputObject will be updated.
        which_kappa (int) : (1 or 2) Only relevant for 'curvature_integrated'.
            Indicates which principal curvature for evaluation.
        **kwargs (dict) : additional keyword arguments to pass to objective 
            gradient function. See input.py for details. 

        Returns
        ----------
        gradient (np array) : gradient of scalar objective with respect to 
            the boundary harmonics. Same shape as self.boundary_opt
        """
        if (which_objective not in self.input_objectives):
            raise ValueError('''Error! input_objective_grad called with 
                incorrect value of which_objective.''')
        elif self.verbose:
            print("Evaluating ",which_objective," objective.")
            
        if (boundary is not None):
            if (np.shape(boundary)!=np.shape(self.boundary_opt)):
                raise ValueError('''Error! input_objective_grad called with 
                     incorrect boundary shape.''')
        
        # Get new input object if necessary
        if (boundary is not None and 
            np.any(np.copy(boundary)!=self.boundary_opt)):
            # Save old boundary if update=False
            if (update==False):
                boundary_old = np.copy(self.boundary_opt)
            # Update equilibrium count
            self.counter += 1
            # Update boundary
            self.update_boundary_opt(boundary)
            directory_name = self.name+"_"+str(self.counter)
            # Make directory for VMEC evaluations
            try:
                os.mkdir(directory_name)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    if (self.verbose):
                        print('Warning. Directory '+directory_name+\
                          ' already exists. Contents may be overwritten.')
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
            [dfdrmnc,dfdzmns] = vmecInputObject.jacobian_derivatives(\
                               self.xm_sensitivity,self.xn_sensitivity,**kwargs)
        elif (which_objective == 'radius'):
            [dfdrmnc,dfdzmns] = vmecInputObject.radius_derivatives(\
                               self.xm_sensitivity,self.xn_sensitivity,**kwargs)
        elif (which_objective == 'normalized_jacobian'):
            [dfdrmnc,dfdzmns] = \
                vmecInputObject.normalized_jacobian_derivatives(\
                               self.xm_sensitivity,self.xn_sensitivity,**kwargs)
        elif (which_objective == 'area'):
            [dfdrmnc,dfdzmns] = vmecInputObject.area_derivatives(\
                               self.xm_sensitivity,self.xn_sensitivity,**kwargs)
        elif (which_objective == 'volume'):
            [dfdrmnc,dfdzmns] = vmecInputObject.volume_derivatives(\
                               self.xm_sensitivity,self.xn_sensitivity,**kwargs)
        elif (which_objective == 'summed_proximity'):
            [dfdrmnc,dfdzmns] = vmecInputObject.summed_proximity_derivatives(\
                               self.xm_sensitivity,self.xn_sensitivity,**kwargs)
        elif (which_objective == 'summed_radius'):
             [dfdrmnc,dfdzmns] = \
                vmecInputObject.summed_radius_constraint_derivatives(\
                               self.xm_sensitivity,self.xn_sensitivity,**kwargs)           
        elif (which_objective == 'curvature'):
            [dfdrmnc,dfdzmns] = \
                vmecInputObject.surface_curvature_metric_derivatives(\
                               self.xm_sensitivity,self.xn_sensitivity,**kwargs)
        elif (which_objective == 'curvature_integrated'):
            [df1drmnc,df1dzmns,df2drmnc,df2dzmns] = \
                vmecInputObject.surface_curvature_metric_integrated_derivatives(\
                               self.xm_sensitivity,self.xn_sensitivity,**kwargs)
            if (which_kappa == 1):
                dfdrmnc = df1drmnc
                dfdzmns = df1dzmns
            elif (which_kappa == 2):
                dfdrmnc = df2drmnc
                dfdzmns = df2dzmns      
            else:
                raise ValueError('Incorrect value for which_kappa!')
        cond = np.logical_not(np.logical_and(self.xm_sensitivity==0, 
                                             self.xn_sensitivity==0))
        if (dfdrmnc.ndim==1):
            dfdzmns = dfdzmns[cond]
            gradient = np.hstack((dfdrmnc,dfdzmns))
        else:
            dfdzmns = dfdzmns[cond,...]
            gradient = np.vstack((dfdrmnc,dfdzmns))

        return np.squeeze(gradient)

    def vmec_objective_grad(self,boundary=None,which_objective='iota',\
                                     update=True,**kwargs):
        """
        Evaluates gradient of VMEC objective. Call VMEC with boundary specified 
        by boundaryObjective to evaluate which_objective and compute gradient.
        If boundary is not specified, boundary will not be updated.

        Parameters
        ----------
        boundary (np array) : coefficients defining boundary in the format of 
            self.boundary_opt. If None, objective
            will be computed from self.vmecOutputObject
        which_objective (string) : Which objective to compute shape gradient
            of. Must be in self.vmec_objectives.
        update (boolean) : If True, self.vmecOutputObject will be updated.
        **kwargs (dict) : additional keyword arguments to pass to objective 
            gradient function. See output.py for details. 

        Returns
        ----------
        gradient (np array) : gradient of scalar objective with respect to 
            the boundary harmonics. Same shape as self.boundary_opt
        """
        if (which_objective not in self.vmec_objectives):
            raise ValueError('''Error! vmec_objective_grad called with
                 incorrect value of which_objective.''')
        elif self.verbose:
            print("Evaluating ",which_objective," objective grad.")
        rank = MPI.COMM_WORLD.Get_rank()
        
        if (boundary is not None):
            if (np.shape(boundary)!=np.shape(self.boundary_opt)):
                raise ValueError('''Error! vmec_objective_grad called with 
                     incorrect boundary shape.''')

        # Evaluate base equilibrium if necessary
        if (((boundary is not None and 
              np.any(np.copy(boundary) != self.boundary_opt))\
            or self.vmecOutputObject is None)):
            [error_code, vmecOutputObject] = self.evaluate_vmec(\
                                                boundary=boundary,update=update)
            if (error_code != 0):
                raise RuntimeError('''Unable to evaluate base VMEC equilibrium
                     in vmec_shape_gradient.''')
        else:
            vmecOutputObject = self.vmecOutputObject

        if (which_objective == 'jacobian'):
            [dfdrmnc,dfdzmns] = vmecOutputObject.jacobian_derivatives(\
                                        self.xm_sensitivity,self.xn_sensitivity)
        elif (which_objective == 'radius'):
            [dfdrmnc,dfdzmns] = vmecOutputObject.radius_derivatives(\
                                        self.xm_sensitivity,self.xn_sensitivity)
        elif (which_objective == 'normalized_jacobian'):
            [dfdrmnc,dfdzmns] = \
                vmecOutputObject.normalized_jacobian_derivatives(\
                                        self.xm_sensitivity,self.xn_sensitivity)
        else:
        # Compute gradient
            shape_gradient = self.vmec_shape_gradient(vmecOutputObject=vmecOutputObject,\
                which_objective=which_objective,update=update,**kwargs)
            [dfdrmnc,dfdzmns] = \
                self.parameter_derivatives(shape_gradient,vmecOutputObject)
            
        cond = np.logical_not(np.logical_and(self.xm_sensitivity==0,
                                             self.xn_sensitivity==0))
        if (dfdrmnc.ndim==1):
            dfdzmns = dfdzmns[cond]
            gradient = np.hstack((dfdrmnc,dfdzmns))
        else:
            dfdzmns = dfdzmns[cond,...]
            gradient = np.vstack((dfdrmnc,dfdzmns))

        return np.squeeze(gradient)

    def test_jacobian(self,vmecInputObject=None):
        """
        Test for singularity in surface Jacobian. If normalized minimum falls
        below self.jacobian_threshold, returns False

        Parameters
        ----------
        vmecInputObject (VmecInput) : instance of VmecInput for evaluation. If
            None, self.vmecInputObject is used. 
         
        Returns
        ----------
        orientable (boolean) : if True, the Jacobian test is passed, and the
            surface is well-defined 
        """
        if (vmecInputObject is None):
            vmecInputObject = self.vmecInputObject
        jac_grid = vmecInputObject.jacobian()
        # Normalize by average Jacobian
        norm = np.sum(jac_grid)/np.size(jac_grid)
        jacobian_diagnostic = np.min(jac_grid)/norm
        orientable = jacobian_diagnostic > self.jacobian_threshold
        return orientable

    def call_vmec(self,vmecInputObject,animec=False):
        """
        Call VMEC after performing a few tests. 
        
        Parameters
        ----------
        vmecInputObject (VmecInput) : instance of VmecInput for evaluation. 
        animec (boolean) : If True, ANIMEC is called rather than VMEC.
         
        Returns
        ----------
        error_code (integer) : error code returned by VMEC. If non-zero, 
            indicates an error. 
                    (from vmec_params.f):
                    norm_term_flag=0
                    bad_jacobian_flag=1
                    more_iter_flag=2
                    jac75_flag=4
                    input_error_flag=5
                    phiedge_error_flag=7
                    ns_error_flag=8
                    misc_error_flag=9
                    successful_term_flag=11
                    bsub_bad_js1_flag=12
                    r01_bad_value_flag=13
                    arz_bad_value_flag=14
                    (other error codes defined locally):
                    not orientable boundary = 15
                    insecting boundary = 16
        """
        if (vmecInputObject is None):
            vmecInputObject = copy.deepcopy(vmecInputObject)

        intersecting = vmecInputObject.test_boundary()

        if (intersecting):
            # print input namelist for the record
            vmecInputObject.print_namelist()
            if (self.verbose):
                print('''Requested boundary is self-intersecting.
                    I will not call VMEC.''')
            return 16
        else:
            orientable = self.test_jacobian(vmecInputObject)
            if (not orientable):
                # print input namelist for the record
                vmecInputObject.print_namelist()
                if (self.verbose):
                    print('''Requested boundary has ill-conditioned Jacobian.
                        I will not call VMEC.''')
                return 15
            else:
                inSurface = vmecInputObject.test_axis()
                if (not inSurface):
                    if (self.verbose):
                        print('''Initial magnetic axis does not lie within
                           requested boundary! Trying a modified axis shape.''')
                    return_value = vmecInputObject.modify_axis()
                    if (return_value == -1):
                        # print input namelist for the record
                        vmecInputObject.print_namelist()
                        error_code = -1
        
        vmecInputObject.print_namelist()
        MPI.COMM_WORLD.Barrier()
        rank = MPI.COMM_WORLD.Get_rank()
        if (animec==False):
            if (self.callVMEC_function is None):
                raise RuntimeError('callVMEC_function is not defined.')
            else:
                self.callVMEC_function(vmecInputObject)
        else:
            if (self.callANIMEC_function is None):
                raise RuntimeError('callANIMEC_function is not defined.')
            else:
                self.callANIMEC_function(vmecInputObject)
        # Check for VMEC errors
        wout_filename = "wout_"+vmecInputObject.input_filename[6::]+".nc"
        MPI.COMM_WORLD.Barrier()
        if (rank == 0):
            try:
                f = netcdf.netcdf_file(wout_filename,'r',mmap=False)
                error_code = f.variables["ier_flag"][()]
                # Check if tolerances were met
                ftolv = f.variables["ftolv"][()]
                fsqr = f.variables["fsqr"][()]
                fsqz = f.variables["fsqz"][()]
                fsql = f.variables["fsql"][()]
                if (fsqr>ftolv or fsqz>ftolv or fsql>ftolv):
                    error_code = 2
            except:
                print('Unable to read '+wout_filename+' in call_vmec.')
                error_code = 15
        else:
            error_code = None
        error_code = MPI.COMM_WORLD.bcast(error_code,root=0)

        if (error_code != 0):
            if (self.verbose):
                print('VMEC completed with error code '+str(error_code))
        return error_code
    
    def parameter_derivatives(self,shape_gradient,vmecOutputObject):
        """
        Computes derivatives with respect to the boundary parameters from the
        shape gradinent. 
        
        Parameters
        ----------
        shape_gradient (np array) : shape gradient on nzeta, ntheta grid
        vmecOutputObject (VmecOutput) : output object for evaluation
        
        Returns
        ----------
        dfdrmnc (np array) : derivative of objective with respect to rmnc 
            (length = self.mnmax_sensitivity)
        dfdzmns (np array) : derivative of objective with respect to zmns
            (length = self.mnmax_sensitivity)
        """
        if (len(shape_gradient[:,0])!=self.nzeta or 
                len(shape_gradient[0,:])!=self.ntheta):
            raise ValueError('''shape_gradient has incorrect shape in 
                             parameter_derivatives''')
        
        [Nx,Ny,Nz] = vmecOutputObject.N(-1)

        dfdrmnc = np.zeros(self.mnmax_sensitivity)
        dfdzmns = np.zeros(self.mnmax_sensitivity)
        for imn in range(self.mnmax_sensitivity):
            angle = self.xm_sensitivity[imn]*vmecOutputObject.thetas_2d \
                - vmecOutputObject.nfp*self.xn_sensitivity[imn] \
                * vmecOutputObject.zetas_2d
            dfdrmnc[imn] = np.sum(np.sum(np.cos(angle) \
                           * (Nx*np.cos(vmecOutputObject.zetas_2d) \
                           + Ny*np.sin(vmecOutputObject.zetas_2d)) \
                           * shape_gradient))
            dfdzmns[imn] = np.sum(np.sum(np.sin(angle)*Nz*shape_gradient))
        dfdrmnc = -dfdrmnc*vmecOutputObject.dtheta*vmecOutputObject.dzeta \
                  * vmecOutputObject.nfp
        dfdzmns = -dfdzmns*vmecOutputObject.dtheta*vmecOutputObject.dzeta \
                  * vmecOutputObject.nfp
        return dfdrmnc, dfdzmns

