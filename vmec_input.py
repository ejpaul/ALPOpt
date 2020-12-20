import os
import sys
import logging
import numpy as np
import scipy
import scipy.linalg
from scipy.io import netcdf
import f90nml
from scanf import scanf
import numpy as np
from surface_utils import point_in_polygon, init_modes, \
    min_max_indices_2d, proximity_slice, self_contact, self_intersect, \
    global_curvature_surface, proximity_surface, proximity_derivatives_func, \
    cosine_IFT, sine_IFT, surface_intersect, read_bounary_harmonic

class VmecInput:

    def __init__(self, input_filename, ntheta=100, nzeta=100, verbose=False):
        """
        Reads VMEC input parameters from a VMEC input file. 

        Args:
            input_filename (str): The filename to read from.
            ntheta (int): Number of poloidal gridpoints
            nzeta (int): Number of toroidal gridpoints (per period)
            verbose (boolean) : if True, more output is written

        """
        logger = logging.getLogger(__name__)
        self.input_filename = input_filename
        self.directory = os.getcwd()

        # Read items from Fortran namelist
        nml = f90nml.read(input_filename)
        nml = nml.get("indata")
        self.nfp = nml.get("nfp")
        mpol_input = nml.get("mpol")
        if (mpol_input < 2):
            logger.error('Error! mpol must be > 1.')
            sys.exit(1)
        self.namelist = nml
        
        self.raxis = nml.get("raxis")
        self.zaxis = nml.get("zaxis")
        self.curtor = nml.get("curtor")
        self.ac_aux_f = nml.get("ac_aux_f")
        self.ac_aux_s = nml.get("ac_aux_s")
        self.pcurr_type = nml.get("pcurr_type")
        self.am_aux_f = nml.get("am_aux_f")
        self.am_aux_s = nml.get("am_aux_s")
        self.pmass_type = nml.get("pmass_type")
        self.ncurr = nml.get("ncurr")
        
        self.update_modes(nml.get("mpol"), nml.get("ntor"))

        self.update_grids(ntheta, nzeta)
        self.animec = False
        self.verbose = verbose

    def update_namelist(self):
        """
        Updates Fortran namelist with class attributes (except for rbc/zbs)
        """
        self.namelist['nfp'] = self.nfp
        self.namelist['raxis'] = self.raxis
        self.namelist['zaxis'] = self.zaxis
        self.namelist['curtor'] = self.curtor
        self.namelist['ac_aux_f'] = self.ac_aux_f
        self.namelist['ac_aux_s'] = self.ac_aux_s
        self.namelist['pcurr_type'] = self.pcurr_type
        self.namelist['am_aux_f'] = self.am_aux_f
        self.namelist['am_aux_s'] = self.am_aux_s
        self.namelist['pmass_type'] = self.pmass_type
        self.namelist['mpol'] = self.mpol
        self.namelist['ntor'] = self.ntor
        if self.animec:
            self.namelist['ah_aux_f'] = self.ah_aux_f
            self.namelist['ah_aux_s'] = self.ah_aux_s
            self.namelist['photp_type'] = self.photp_type
            self.namelist['bcrit'] = self.bcrit

    def update_modes(self, mpol, ntor):
        """
        Updates poloidal and toroidal mode numbers (xm and xn) given maximum
            mode numbers (mpol and ntor)
            
        Args:
            mpol (int): maximum poloidal mode number + 1
            ntor (int): maximum toroidal mode number
        """
        self.mpol = mpol
        self.ntor = ntor
        [self.mnmax, self.xm, self.xn] = init_modes(
                self.mpol - 1, self.ntor)
        [self.rbc,self.zbs] = self.read_boundary_input()
        
    def init_grid(self,ntheta,nzeta):
        """
        Initializes uniform grids in toroidal and poloidal angles
        
        Args:
            ntheta (int) : number of poloidal grid points
            nzeta (int) : number of toroidal grid points 
        """
        thetas = np.linspace(0, 2 * np.pi, ntheta + 1)
        zetas = np.linspace(0,2 * np.pi / self.nfp, nzeta + 1)
        thetas = np.delete(thetas, -1)
        zetas = np.delete(zetas, -1)
        [thetas_2d, zetas_2d] = np.meshgrid(thetas, zetas)
        dtheta = thetas[1] - thetas[0]
        dzeta = zetas[1] - zetas[0]
        
        return thetas_2d,zetas_2d, dtheta, dzeta
        
    def update_grids(self,ntheta,nzeta):
        """
        Updates poloidal and toroidal grids (thetas and zetas) given number of
            grid points
            
        Args:
            ntheta (int): number of poloidal grid points
            nzeta (int): number of toroidal grid points (per period)
        """
        self.ntheta = ntheta
        self.nzeta = nzeta
        self.thetas = np.linspace(0, 2 * np.pi, ntheta + 1)
        self.zetas = np.linspace(0,2 * np.pi / self.nfp, nzeta + 1)
        self.thetas = np.delete(self.thetas, -1)
        self.zetas = np.delete(self.zetas, -1)
        [self.thetas_2d, self.zetas_2d] = np.meshgrid(self.thetas, self.zetas)
        self.dtheta = self.thetas[1] - self.thetas[0]
        self.dzeta = self.zetas[1] - self.zetas[0]
        
        self.zetas_full = np.linspace(0, 2 * np.pi, self.nfp * nzeta + 1)
        self.zetas_full = np.delete(self.zetas_full, -1)
        [self.thetas_2d_full, self.zetas_2d_full] = np.meshgrid(
                self.thetas, self.zetas_full)
        
    def read_boundary_input(self):
        """
        Read in boundary harmonics (rbc, zbs) from Fortran namelist

        Returns:
            rbc (float array): boundary harmonics for radial coordinate
                (same size as xm and xn)
            zbs (float array): boundary harmonics for height coordinate
                (same size as xm and xn)
        """
        [xn_rbc,xm_rbc,rbc_input] = read_bounary_harmonic('rbc',
                                                          self.input_filename)
        [xn_zbs,xm_zbs,zbs_input] = read_bounary_harmonic('zbs',
                                                          self.input_filename)
        
        xn_zbs = np.asarray(xn_zbs)
        xn_rbc = np.asarray(xn_rbc)
        xm_rbc = np.asarray(xm_rbc)
        xm_zbs = np.asarray(xm_zbs)
        rbc_input = np.asarray(rbc_input)
        zbs_input = np.asarray(zbs_input)
                
        rbc = np.zeros(self.mnmax)
        zbs = np.zeros(self.mnmax)
        for imn in range(self.mnmax):
            rbc_indices = np.argwhere(np.logical_and(xm_rbc == self.xm[imn],\
                                                  xn_rbc == self.xn[imn]))
            if (rbc_indices.size==1):
                rbc[imn] = rbc_input[rbc_indices[0]]
            elif (len(rbc_indices)>1):
                raise RuntimeError('''Invalid boundary encountered in 
                                   read_boundary_input.''')
            zbs_indices = np.argwhere(np.logical_and((xm_zbs == self.xm[imn]),\
                                          xn_zbs == self.xn[imn]))
            if (zbs_indices.size==1):
                zbs[imn] = zbs_input[zbs_indices[0]]
            elif (len(zbs_indices)>1):
                raise RuntimeError('''Invalid boundary encountered in 
                                   read_boundary_input.''')
        
        return rbc, zbs

    def area(self, theta=None, zeta=None):
        """
        Computes area of boundary surface
        
        Args:
            theta (float array): poloidal grid for area evaluation (optional)
            zeta (float array): toroidal grid for area evaluation (optional)
            
        Returns:
            area (float): area of boundary surface
        """
        if (np.shape(theta) != np.shape(zeta)):
            raise ValueError('theta and zeta must have the same shape in area.')
        norm_normal = self.jacobian(theta, zeta)
        area = np.sum(norm_normal) * self.dtheta * self.dzeta * self.nfp
        return area
      
    def volume(self,theta=None,zeta=None):
        """
        Computes volume enclosed by boundary surface
        
        Args:
            theta (float array): poloidal grid for volume evaluation (optional)
            zeta (float array): toroidal grid for volume evaluation (optional)
        
        Returns:
            volume (float): volume enclosed by boundary surface
        """
        if (np.shape(theta) != np.shape(zeta)):
            raise ValueError('theta and zeta must have the same shape in area.')
        [Nx, Ny, Nz] = self.normal(theta, zeta)
        [X, Y, Z, R] = self.position(theta, zeta)
        volume = abs(np.sum(Z * Nz)) * self.dtheta * self.dzeta * self.nfp
        return volume
      
    def normal(self, theta=None, zeta=None):
        """
        Computes components of normal vector multiplied by surface Jacobian
        
        Args:
            theta (float array): poloidal grid for N evaluation (optional)
            zeta (float array): toroidal grid for N evaluation (optional)
        
        Returns:
            Nx (float array): x component of unit normal multiplied by Jacobian
            Ny (float array): y component of unit normal multiplied by Jacobian
            Nz (float array): z component of unit normal multiplied by Jacobian
        """
        if (np.shape(theta) != np.shape(zeta)):
            raise ValueError('theta and zeta must have the same shape in area.')
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                    dRdzeta] = self.position_first_derivatives(theta,zeta)
      
        Nx = -dydzeta * dzdtheta + dydtheta * dzdzeta
        Ny = -dzdzeta * dxdtheta + dzdtheta * dxdzeta
        Nz = -dxdzeta * dydtheta + dxdtheta * dydzeta
        return Nx, Ny, Nz
      
    def position_first_derivatives(self, theta=None, zeta=None):
        """
        Computes derivatives of position vector with respect to angles
        
        Args:
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
        
        Returns:
            dxdtheta (float array): derivative of x wrt poloidal angle
            dxdzeta (float array): derivative of x wrt toroidal angle
            dydtheta (float array): derivative of y wrt poloidal angle
            dydzeta (float array): derivative of y wrt toroidal angle
            dzdtheta (float array): derivative of z wrt poloidal angle
            dzdzeta (float array): derivative of z wrt toroidal angle
            dRdtheta (float array): derivative of R wrt poloidal angle
            dRdzeta (float array): derivative of R wrt toroidal angle

        """
        if (theta is None and zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
        elif (np.array(theta).shape != np.array(zeta).shape):
            raise ValueError('Incorrect shape for theta and zeta in \
                position_first_derivatives')
        
        R = cosine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,self.rbc)
        dRdtheta = sine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,
                            -self.xm*self.rbc)
        dzdtheta = cosine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,
                              self.xm*self.zbs)
        dRdzeta = sine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,
                           self.nfp*self.xn*self.rbc)
        dzdzeta = cosine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,
                             -self.nfp*self.xn*self.zbs)
    
        dxdtheta = dRdtheta * np.cos(zeta)
        dydtheta = dRdtheta * np.sin(zeta)
        dxdzeta = dRdzeta * np.cos(zeta) - R * np.sin(zeta)
        dydzeta = dRdzeta * np.sin(zeta) + R * np.cos(zeta)
        return dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, \
                dRdtheta, dRdzeta

    def jacobian(self, theta=None, zeta=None):
        """
        Computes surface Jacobian (differential area element)
        
        Args:
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
            
        Returns:
            norm_normal (float array): surface Jacobian

        """
        [Nx, Ny, Nz] = self.normal(theta, zeta)
        norm_normal = np.sqrt(Nx**2 + Ny**2 + Nz**2)
        return norm_normal
      
    def normalized_jacobian(self, theta=None, zeta=None):
        """
        Computes surface Jacobian normalized by surface area
        
        Args:
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
        
        Returns:
            normalized_norm_normal (float array): normalized surface Jacobian

        """
        norm_normal = self.jacobian(theta, zeta)
        area = self.area(theta, zeta)
        normalized_norm_normal = norm_normal * 4*np.pi * np.pi / area
        return normalized_norm_normal
      
    def normalized_jacobian_derivatives(self, xm_sensitivity, xn_sensitivity,
                                        theta=None, zeta=None):
        """
        Computes derivatives of normalized Jacobian with respect to Fourier
            harmonics of the boundary (rbc, zbs)
        
        Args:
            xm_sensitivity (int array): poloidal modes for derivative
                evaluation
            xn_sensitivity (int array): toroidal modes for derivative
                evaluation
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
            
        Returns:
            dnormalized_jacobiandrmnc (float array): derivatives with respect to
                radius boundary harmonics
            dnormalized_jacobiandzmns (float array): derivatives with respect to
                height boundary harmonics
        
        """
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            logger.error('Error! Incorrect dimensions for theta and zeta in '
                         'normalized_jacobian_derivatives.')
            sys.exit(0)
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:, 0])
            dim2 = len(theta[0, :])
        else:
            logger.error('Error! Incorrect dimensions for theta and zeta in '
                         'normalized_jacobian_derivatives.')
            sys.exit(0)
        [dareadrmnc, dareadzmns] = self.area_derivatives(
                    xm_sensitivity, xn_sensitivity, theta, zeta)
        [dNdrmnc, dNdzmns] = self.jacobian_derivatives(
                    xm_sensitivity, xn_sensitivity, theta, zeta)
        area = self.area(theta, zeta)
        N = self.jacobian(theta, zeta)
        
        mnmax_sensitivity = len(xm_sensitivity)
        dnormalized_jacobiandrmnc = np.zeros((mnmax_sensitivity, dim1, dim2))
        dnormalized_jacobiandzmns = np.zeros((mnmax_sensitivity, dim1, dim2))
        for imn in range(mnmax_sensitivity):
            dnormalized_jacobiandrmnc[imn, :, :] = dNdrmnc[imn]/area - \
                    N * dareadrmnc[imn] / (area * area)
            dnormalized_jacobiandzmns[imn, :, :] = dNdzmns[imn]/area - \
                    N * dareadzmns[imn] / (area * area)
        dnormalized_jacobiandrmnc *= 4 * np.pi * np.pi
        dnormalized_jacobiandzmns *= 4 * np.pi * np.pi

        return dnormalized_jacobiandrmnc, dnormalized_jacobiandzmns
        
    def position(self, theta=None, zeta=None):
        """
        Computes position vector
        
        Args:
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
            
        Returns:
            X (float array): x coordinate on angular grid
            Y (float array): y coordinate on angular grid
            Z (float array): z coordinate on angular grid
            R (float array): height coordinate on angular grid
        
        """
        if (theta is None and zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
        elif (np.array(theta).shape != np.array(zeta).shape):
            raise ValueError(
                'Error! Incorrect dimensions for theta and zeta in position.')
      
        R = cosine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,self.rbc)        
        Z = sine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,self.zbs)
        
        X = R * np.cos(zeta)
        Y = R * np.sin(zeta)
        return X, Y, Z, R
    
    def position_derivatives_surface(self, xm_sensitivity, xn_sensitivity, \
                                         theta=None, zeta=None):
        if (theta is None and zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
        elif (np.array(theta).shape != np.array(zeta).shape):
            raise ValueError(
                'Error! Incorrect dimensions for theta and zeta in position.')
      
        angle = xm_sensitivity[:,np.newaxis,np.newaxis]*theta[np.newaxis,:,:] \
         - self.nfp*xn_sensitivity[:,np.newaxis,np.newaxis]*zeta[np.newaxis,:,:]
        dRdrmnc = np.cos(angle)
        dZdzmns = np.sin(angle)
        dXdrmnc = dRdrmnc*np.cos(zeta[np.newaxis,:,:])
        dYdrmnc = dRdrmnc*np.sin(zeta[np.newaxis,:,:])

        return dXdrmnc, dYdrmnc, dZdzmns, dRdrmnc
      
    def position_second_derivatives(self, theta=None, zeta=None):
        """
        Computes second derivatives of position vector with respect to the
            toroidal and poloidal angles
            
        Args:
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
            
        Returns:
            d2xdtheta2 (float array): second derivative of x wrt poloidal angle
            d2xdzeta2 (float array): second derivative of x wrt toroidal angle
            d2xdthetadzeta (float array): second derivative of x wrt toroidal
                and poloidal angles
            d2ydtheta2 (float array): second derivative of y wrt poloidal angle
            d2ydzeta2 (float array): second derivative of y wrt toroidal angle
            d2ydthetadzeta (float array): second derivative of y wrt toroidal
                and poloidal angles
            d2Zdtheta2 (float array): second derivative of height wrt poloidal
                angle
            d2Zdzeta2 (float array): second derivative of height wrt toroidal
                angle
            d2Zdthetadzeta (float array): second derivative of height wrt
                toroidal and poloidal angles

        """
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
        elif (np.array(theta).shape != np.array(zeta).shape):
            raise ValueError('Incorrect shape of theta and zeta in '
                         'position_second_derivatives')
            
        R = cosine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,self.rbc)
        dRdtheta = sine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,\
                            -self.xm*self.rbc)
        dzdtheta = cosine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,\
                              self.xm*self.zbs)
        dRdzeta = sine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,\
                           float(self.nfp)*self.xn*self.rbc)
        dzdzeta = cosine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,\
                             -float(self.nfp)*self.xn*self.zbs)
        d2Rdtheta2 = -cosine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,\
                                 self.xm*self.xm*self.rbc)
        d2Zdtheta2 = -sine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,\
                                 self.xm*self.xm*self.zbs)
        d2Rdzeta2 = -cosine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,\
                     float(self.nfp)*float(self.nfp)*self.xn*self.xn*self.rbc)
        d2Zdzeta2 = -sine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,\
                     float(self.nfp)*self.nfp*self.xn*self.xn*self.zbs)
        d2Rdthetadzeta = cosine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,\
                         float(self.nfp)*self.xm*self.xn*self.rbc)
        d2Zdthetadzeta = sine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,\
                         float(self.nfp)*self.xm*self.xn*self.zbs)
        
        d2xdtheta2 = d2Rdtheta2 * np.cos(zeta)
        d2ydtheta2 = d2Rdtheta2 * np.sin(zeta)
        d2xdzeta2 = d2Rdzeta2 * np.cos(zeta) - 2 * dRdzeta * np.sin(zeta) - \
                R * np.cos(zeta)
        d2ydzeta2 = d2Rdzeta2 * np.sin(zeta) + 2 * dRdzeta * np.cos(zeta) - \
                R * np.sin(zeta)
        d2xdthetadzeta = d2Rdthetadzeta * np.cos(zeta) - dRdtheta * np.sin(zeta)
        d2ydthetadzeta = d2Rdthetadzeta * np.sin(zeta) + dRdtheta * np.cos(zeta)
        return d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, \
                d2ydthetadzeta, d2Zdtheta2, d2Zdzeta2, d2Zdthetadzeta, \
               d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta
    
    def position_second_derivatives_surface(self, xm_sensitivity, xn_sensitivity, 
                                            theta=None, zeta=None):
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            logger.error('Error! Incorrect dimensions for theta and zeta '
                         'in position_second_derivatives_surface.')
            sys.exit(0)
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:,0])
            dim2 = len(theta[0,:])
        else:
            logger.error('Error! Incorrect dimensions for theta and zeta '
                         'in position_second_derivatives_surface.')
            sys.exit(0)    

        angle = xm_sensitivity[:,np.newaxis,np.newaxis] * theta[np.newaxis,:,:] \
                   - self.nfp * xn_sensitivity[:,np.newaxis,np.newaxis] \
                   * zeta[np.newaxis,:,:]
        d3Rdtheta2drmnc = -xm_sensitivity[:,np.newaxis,np.newaxis] * \
                           xm_sensitivity[:,np.newaxis,np.newaxis] * \
                           np.cos(angle)
        d3Zdtheta2dzmns = -xm_sensitivity[:,np.newaxis,np.newaxis] * \
                           xm_sensitivity[:,np.newaxis,np.newaxis] * \
                           np.sin(angle)
        d3Rdzeta2drmnc = -  xn_sensitivity[:,np.newaxis,np.newaxis] * self.nfp \
                          * xn_sensitivity[:,np.newaxis,np.newaxis] * self.nfp \
                          * np.cos(angle)
        d3Zdzeta2dzmns = -xn_sensitivity[:,np.newaxis,np.newaxis] * self.nfp \
                          * xn_sensitivity[:,np.newaxis,np.newaxis] * self.nfp \
                          * np.sin(angle)
        d3Rdthetadzetadrmnc = xm_sensitivity[:,np.newaxis,np.newaxis] * \
                           xn_sensitivity[:,np.newaxis,np.newaxis] * self.nfp \
                          * np.cos(angle)
        d3Zdthetadzetadzmns = xm_sensitivity[:,np.newaxis,np.newaxis] * \
                           xn_sensitivity[:,np.newaxis,np.newaxis] * self.nfp \
                          * np.sin(angle)

        [d2Rdthetadrmnc, d2Xdthetadrmnc, d2Ydthetadrmnc, d2Zdthetadzmns, \
            d2Rdzetadrmnc, d2Xdzetadrmnc, d2Ydzetadrmnc, d2Zdzetadzmns] = \
            self.position_first_derivatives_surface(xm_sensitivity, \
                                        xn_sensitivity, theta=theta, zeta=zeta)
        [dXdrmnc, dYdrmnc, dZdzmns, dRdrmnc] = \
            self.position_derivatives_surface(xm_sensitivity, \
                                        xn_sensitivity, theta=theta, zeta=zeta)
        
        d3xdtheta2drmnc = d3Rdtheta2drmnc*np.cos(zeta[np.newaxis,:,:])
        d3ydtheta2drmnc = d3Rdtheta2drmnc*np.sin(zeta[np.newaxis,:,:])
        d3xdzeta2drmnc = d3Rdzeta2drmnc*np.cos(zeta[np.newaxis,:,:]) \
            - 2 * d2Rdzetadrmnc*np.sin(zeta[np.newaxis,:,:]) \
            - dRdrmnc*np.cos(zeta[np.newaxis,:,:])
        d3ydzeta2drmnc = d3Rdzeta2drmnc*np.sin(zeta[np.newaxis,:,:]) \
            + 2 * d2Rdzetadrmnc*np.cos(zeta[np.newaxis,:,:]) \
            - dRdrmnc*np.sin(zeta[np.newaxis,:,:])
        d3xdthetadzetadrmnc = d3Rdthetadzetadrmnc * np.cos(zeta) \
            - d2Rdthetadrmnc * np.sin(zeta)
        d3ydthetadzetadrmnc = d3Rdthetadzetadrmnc * np.sin(zeta) \
            + d2Rdthetadrmnc * np.cos(zeta)
      
        return d3xdtheta2drmnc, d3ydtheta2drmnc, d3Zdtheta2dzmns, \
               d3xdzeta2drmnc, d3ydzeta2drmnc, d3Zdzeta2dzmns, \
               d3xdthetadzetadrmnc, d3ydthetadzetadrmnc, \
               d3Zdthetadzetadzmns
        
    def surface_curvature(self, theta=None, zeta=None):
        """
        Computes mean curvature of boundary surface
        
        Args:
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)

        Returns:
            H (float array): mean curvature on angular grid
        
        """
        [E, F, G, e, f, g] = self.metric_tensor(theta,zeta)
        
        H = (e*G -2*f*F + g*E)/(E*G - F*F)
        K = (e*g - f*f)/(E*G - F*F)
        kappa1 = H + np.sqrt(H*H - K)
        kappa2 = H - np.sqrt(H*H - K)
        return H, K, kappa1, kappa2
    
    def metric_tensor (self, theta=None, zeta=None):
        [dXdtheta, dXdzeta, dYdtheta, dYdzeta, dZdtheta, dZdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta, zeta)
        [d2Xdtheta2, d2Xdzeta2, d2Xdthetadzeta, d2Ydtheta2, d2Ydzeta2, \
                d2Ydthetadzeta, d2Zdtheta2, d2Zdzeta2, d2Zdthetadzeta, \
                d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta] = \
                self.position_second_derivatives(theta, zeta)
        norm_normal = self.jacobian(theta=theta,zeta=zeta)
        [Nx, Ny, Nz] = self.normal(theta=theta,zeta=zeta)

        nx = Nx / norm_normal
        ny = Ny / norm_normal
        nz = Nz / norm_normal
        E = dXdtheta * dXdtheta + dYdtheta * dYdtheta + dZdtheta * dZdtheta
        F = dXdtheta * dXdzeta + dYdtheta * dYdzeta + dZdtheta * dZdzeta
        G = dXdzeta * dXdzeta + dYdzeta * dYdzeta + dZdzeta * dZdzeta
        e = nx * d2Xdtheta2 + ny * d2Ydtheta2 + nz * d2Zdtheta2
        f = nx * d2Xdthetadzeta + ny * d2Ydthetadzeta + nz * d2Zdthetadzeta
        g = nx * d2Xdzeta2 + ny * d2Ydzeta2 + nz * d2Zdzeta2

        return E, F, G, e, f, g
    
    def surface_curvature_derivatives(self, xm_sensitivity, xn_sensitivity,\
                                      theta=None, zeta=None):
        """
        Computes mean curvature of boundary surface
        
        Args:
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)

        Returns:
            H (float array): mean curvature on angular grid
        
        """
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            logger.error('Error! Incorrect dimensions for theta and zeta '
                         'in volume_derivatives.')
            sys.exit(0)
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:,0])
            dim2 = len(theta[0,:])
        else:
            logger.error('Error! Incorrect dimensions for theta and zeta '
                         'in volume_derivatives.')
            sys.exit(0)

        [dXdtheta, dXdzeta, dYdtheta, dYdzeta, dZdtheta, dZdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta, zeta)
        [d2Xdtheta2, d2Xdzeta2, d2Xdthetadzeta, d2Ydtheta2, d2Ydzeta2, \
                d2Ydthetadzeta, d2Zdtheta2, d2Zdzeta2, d2Zdthetadzeta, \
                d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta] = \
                self.position_second_derivatives(theta, zeta)
        [d2Rdthetadrmnc, d2Xdthetadrmnc, d2Ydthetadrmnc, d2Zdthetadzmns, \
            d2Rdzetadrmnc, d2Xdzetadrmnc, d2Ydzetadrmnc, d2Zdzetadzmns] = \
                self.position_first_derivatives_surface(xm_sensitivity, \
                            xn_sensitivity, theta=theta, zeta=zeta)
        [d3Xdtheta2drmnc, d3Ydtheta2drmnc, d3Zdtheta2dzmns, \
               d3Xdzeta2drmnc, d3Ydzeta2drmnc, d3Zdzeta2dzmns, \
               d3Xdthetadzetadrmnc, d3Ydthetadzetadrmnc, \
               d3Zdthetadzetadzmns] = \
                self.position_second_derivatives_surface(xm_sensitivity, \
                            xn_sensitivity, theta=theta, zeta=zeta)
        [dNxdrmnc, dNxdzmns, dNydrmnc, dNydzmns, dNzdrmnc] = \
            self.normal_derivatives(xm_sensitivity,xn_sensitivity,theta=theta,\
                                    zeta=zeta)
        [dNdrmnc, dNdzmns] = self.jacobian_derivatives(xm_sensitivity,\
                                    xn_sensitivity,theta=theta,zeta=zeta)
        norm_normal = self.jacobian(theta=theta,zeta=zeta)
        [Nx, Ny, Nz] = self.normal(theta=theta,zeta=zeta)
        nx = Nx / norm_normal
        ny = Ny / norm_normal
        nz = Nz / norm_normal
        dnxdrmnc = dNxdrmnc/norm_normal - nx*dNdrmnc/norm_normal
        dnydrmnc = dNydrmnc/norm_normal - ny*dNdrmnc/norm_normal
        dnzdrmnc = dNzdrmnc/norm_normal - nz*dNdrmnc/norm_normal
        dnxdzmns = dNxdzmns/norm_normal - nx*dNdzmns/norm_normal
        dnydzmns = dNydzmns/norm_normal - ny*dNdzmns/norm_normal
        dnzdzmns =                      - nz*dNdzmns/norm_normal        

        dEdrmnc = 2*d2Xdthetadrmnc*dXdtheta[np.newaxis,:,:] \
                + 2*d2Ydthetadrmnc*dYdtheta[np.newaxis,:,:] 
        dFdrmnc = d2Xdthetadrmnc*dXdzeta[np.newaxis,:,:] \
                + d2Xdzetadrmnc*dXdtheta[np.newaxis,:,:] \
                + d2Ydthetadrmnc*dYdzeta[np.newaxis,:,:] \
                + d2Ydzetadrmnc*dYdtheta[np.newaxis,:,:] 
        dGdrmnc = 2*d2Xdzetadrmnc*dXdzeta[np.newaxis,:,:] \
                + 2*d2Ydzetadrmnc*dYdzeta[np.newaxis,:,:] 
        
        dEdzmns = 2*d2Zdthetadzmns*dZdtheta[np.newaxis,:,:] 
        dFdzmns = d2Zdthetadzmns*dZdzeta[np.newaxis,:,:] \
                + d2Zdzetadzmns*dZdtheta[np.newaxis,:,:]
        dGdzmns = 2*d2Zdzetadzmns*dZdzeta[np.newaxis,:,:] 
        
        dedrmnc = dnxdrmnc * d2Xdtheta2 + nx * d3Xdtheta2drmnc \
                + dnydrmnc * d2Ydtheta2 + ny * d3Ydtheta2drmnc \
                + dnzdrmnc * d2Zdtheta2 
        dedzmns = dnxdzmns * d2Xdtheta2 \
                + dnydzmns * d2Ydtheta2 \
                + dnzdzmns * d2Zdtheta2 + nz * d3Zdtheta2dzmns
        dfdrmnc = dnxdrmnc * d2Xdthetadzeta + nx * d3Xdthetadzetadrmnc \
                + dnydrmnc * d2Ydthetadzeta + ny * d3Ydthetadzetadrmnc \
                + dnzdrmnc * d2Zdthetadzeta 
        dfdzmns = dnxdzmns * d2Xdthetadzeta \
                + dnydzmns * d2Ydthetadzeta \
                + dnzdzmns * d2Zdthetadzeta + nz * d3Zdthetadzetadzmns
        dgdrmnc = dnxdrmnc * d2Xdzeta2 + nx * d3Xdzeta2drmnc \
                + dnydrmnc * d2Ydzeta2 + ny * d3Ydzeta2drmnc \
                + dnzdrmnc * d2Zdzeta2 
        dgdzmns = dnxdzmns * d2Xdzeta2  \
                + dnydzmns * d2Ydzeta2  \
                + dnzdzmns * d2Zdzeta2 + nz * d3Zdzeta2dzmns

        [E, F, G, e, f, g] = self.metric_tensor(theta,zeta)
        [H, K, kappa1, kappa2] = self.surface_curvature(theta,zeta)

        det = E*G - F*F
        ddetdrmnc = dEdrmnc*G + E*dGdrmnc - 2*F*dFdrmnc
        ddetdzmns = dEdzmns*G + E*dGdzmns - 2*F*dFdzmns

        dHdrmnc = (dedrmnc * G + e * dGdrmnc - 2 * dfdrmnc * F \
                   - 2 * f * dFdrmnc + dgdrmnc * E + g * dEdrmnc)/det \
                - H*ddetdrmnc/det
        dHdzmns = (dedzmns * G + e * dGdzmns - 2 * dfdzmns * F \
                   - 2 * f * dFdzmns + dgdzmns * E + g * dEdzmns)/det \
                - H*ddetdzmns/det
        dKdrmnc = (dedrmnc*g + e*dgdrmnc - 2*f*dfdrmnc)/det - K*ddetdrmnc/det
        dKdzmns = (dedzmns*g + e*dgdzmns - 2*f*dfdzmns)/det - K*ddetdzmns/det
        
        dkappa1drmnc = dHdrmnc + 0.5*(2*H*dHdrmnc - dKdrmnc)/np.sqrt(H*H - K)
        dkappa2drmnc = dHdrmnc - 0.5*(2*H*dHdrmnc - dKdrmnc)/np.sqrt(H*H - K)
        
        dkappa1dzmns = dHdzmns + 0.5*(2*H*dHdzmns - dKdzmns)/np.sqrt(H*H - K)
        dkappa2dzmns = dHdzmns - 0.5*(2*H*dHdzmns - dKdzmns)/np.sqrt(H*H - K)

        return dHdrmnc, dHdzmns, dKdrmnc, dKdzmns, dkappa1drmnc, dkappa1dzmns, \
            dkappa2drmnc, dkappa2dzmns
    
    def surface_curvature_metric_integrated(self, ntheta=None, nzeta=None):
        if (ntheta == None or nzeta == None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            dtheta = self.dtheta
            dzeta = self.dzeta
            ntheta = self.ntheta
            nzeta = self.nzeta
        else:            
            [theta, zeta, dtheta, dzeta] = self.init_grid(ntheta,nzeta)
        [H, K, kappa1, kappa2] = self.surface_curvature(theta=theta,zeta=zeta)
        normalized_jacobian = self.normalized_jacobian(theta=theta,zeta=zeta)
        metric1 = 0.5 * np.sum(kappa1**2 * normalized_jacobian) * dtheta * dzeta
        metric2 = 0.5 * np.sum(kappa2**2 * normalized_jacobian) * dtheta * dzeta
        
        return metric1, metric2
    
    def surface_curvature_metric_integrated_derivatives(self, xm_sensitivity, \
                                     xn_sensitivity, ntheta=None, nzeta=None):
        if (ntheta == None or nzeta == None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            dtheta = self.dtheta
            dzeta = self.dzeta
            ntheta = self.ntheta
            nzeta = self.nzeta
        else:            
            [theta, zeta, dtheta, dzeta] = self.init_grid(ntheta,nzeta)
        [H, K, kappa1, kappa2] = self.surface_curvature(theta=theta,zeta=zeta)        
        normalized_jacobian = self.normalized_jacobian(theta=theta,zeta=zeta)
        [dnormalized_jacobiandrmnc, dnormalized_jacobiandzmns] = \
            self.normalized_jacobian_derivatives(xm_sensitivity,xn_sensitivity,\
                                                  theta=theta,zeta=zeta)
        [dHdrmnc, dHdzmns, dKdrmnc, dKdzmns, dkappa1drmnc, dkappa1dzmns, \
            dkappa2drmnc, dkappa2dzmns] \
            = self.surface_curvature_derivatives(xm_sensitivity,xn_sensitivity,\
                                                theta=theta,zeta=zeta)

        kappa1 = kappa1[np.newaxis,:,:]
        kappa2 = kappa2[np.newaxis,:,:]
        normalize_jacobian = normalized_jacobian[np.newaxis,:,:]

        dmetric1drmnc = dtheta * dzeta * \
                            np.sum(dkappa1drmnc*kappa1*normalized_jacobian \
                           + 0.5*kappa1**2*dnormalized_jacobiandrmnc,axis=(1,2))
        dmetric1dzmns = dtheta * dzeta * \
                            np.sum(dkappa1dzmns*kappa1*normalized_jacobian \
                           + 0.5*kappa1**2*dnormalized_jacobiandzmns,axis=(1,2))
        dmetric2drmnc = dtheta * dzeta * \
                            np.sum(dkappa2drmnc*kappa2*normalized_jacobian \
                           + 0.5*kappa2**2*dnormalized_jacobiandrmnc,axis=(1,2))
        dmetric2dzmns = dtheta * dzeta * \
                            np.sum(dkappa2dzmns*kappa2*normalized_jacobian \
                           + 0.5*kappa2**2*dnormalized_jacobiandzmns,axis=(1,2))    
                
        return dmetric1drmnc, dmetric1dzmns, dmetric2drmnc, dmetric2dzmns
        
    def surface_curvature_metric(self, ntheta=None, nzeta=None, max_curvature=30,
                                exp_weight=0.01):
        if (ntheta == None or nzeta == None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            dtheta = self.dtheta
            dzeta = self.dzeta
            ntheta = self.ntheta
            nzeta = self.nzeta
        else:            
            [theta, zeta, dtheta, dzeta] = self.init_grid(ntheta,nzeta)
        [H, K, kappa1, kappa2] = self.surface_curvature(theta=theta,zeta=zeta)
        normalized_jacobian = self.normalized_jacobian(theta=theta,zeta=zeta)
        metric = np.sum((np.exp((kappa1**2-max_curvature**2)/exp_weight**2) \
            + np.exp((kappa2**2-max_curvature**2)/exp_weight**2))*normalized_jacobian) \
            * dtheta * dzeta
        if (np.isinf(metric)):
            metric = 1e12
        return metric
    
    def surface_curvature_metric_derivatives(self, xm_sensitivity, xn_sensitivity, \
                                    ntheta=None, nzeta=None, max_curvature=30,
                                    exp_weight=0.01):
        if (ntheta == None or nzeta == None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            dtheta = self.dtheta
            dzeta = self.dzeta
            ntheta = self.ntheta
            nzeta = self.nzeta
        else:            
            [theta, zeta, dtheta, dzeta] = self.init_grid(ntheta,nzeta)
        [H, K, kappa1, kappa2] = self.surface_curvature(theta=theta,zeta=zeta)
        normalized_jacobian = self.normalized_jacobian(theta=theta,zeta=zeta)

        [dnormalized_jacobiandrmnc, dnormalized_jacobiandzmns] = \
            self.normalized_jacobian_derivatives(xm_sensitivity,xn_sensitivity,\
                                                  theta=theta,zeta=zeta)
        [dHdrmnc, dHdzmns, dKdrmnc, dKdzmns, dkappa1drmnc, dkappa1dzmns, \
            dkappa2drmnc, dkappa2dzmns] \
            = self.surface_curvature_derivatives(xm_sensitivity,xn_sensitivity,\
                                                theta=theta,zeta=zeta)

        kappa1 = kappa1[np.newaxis,:,:]
        kappa2 = kappa2[np.newaxis,:,:]
        normalize_jacobian = normalized_jacobian[np.newaxis,:,:]
        dmetricdrmnc = np.sum((np.exp((kappa1**2-max_curvature**2)/exp_weight**2) \
            + np.exp((kappa2**2-max_curvature**2)/exp_weight**2))*dnormalized_jacobiandrmnc \
            + (np.exp((kappa1**2-max_curvature**2)/exp_weight**2) \
            * 2*kappa1*dkappa1drmnc/exp_weight**2 + 
              np.exp((kappa2**2-max_curvature**2)/exp_weight**2) \
            * 2*kappa2*dkappa2drmnc/exp_weight**2)*normalized_jacobian,axis=(1,2)) \
            * dtheta * dzeta
        dmetricdzmns = np.sum((np.exp((kappa1**2-max_curvature**2)/exp_weight**2) \
            + np.exp((kappa2**2-max_curvature**2)/exp_weight**2))*dnormalized_jacobiandzmns \
            + (np.exp((kappa1**2-max_curvature**2)/exp_weight**2) \
            * 2*kappa1*dkappa1dzmns/exp_weight**2 + 
              np.exp((kappa2**2-max_curvature**2)/exp_weight**2) \
            * 2*kappa2*dkappa2dzmns/exp_weight**2)*normalized_jacobian,axis=(1,2)) \
            * dtheta * dzeta
        
        dmetricdrmnc = np.array(dmetricdrmnc)
        dmetricdzmns = np.array(dmetricdzmns)
        
        dmetricdrmnc[np.isinf(dmetricdrmnc)] = 1e12
        dmetricdzmns[np.isinf(dmetricdzmns)] = 1e12
        dmetricdrmnc[np.isnan(dmetricdrmnc)] = 1e12
        dmetricdzmns[np.isnan(dmetricdzmns)] = 1e12
            
        return dmetricdrmnc, dmetricdzmns
    
    def mean_curvature(self, theta=None, zeta=None):
        """
        Computes mean curvature of boundary surface
        
        Args:
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)

        Returns:
            H (float array): mean curvature on angular grid
        
        """
        [H, K, kappa1, kappa2] = self.surface_curvature(theta=theta,zeta=zeta)
        return H
      
    def print_namelist(self, input_filename=None, namelist=None):
        """
        Prints Fortran namelist to text file for passing to VMEC
        
        Args:
            input_filename (str): desired name of text file (optional)
            namelist (f90nml object): namelist to print (optional)
            
        """
        self.update_namelist()
        if (namelist is None):
            namelist = self.namelist
        if (input_filename is None):
            input_filename = self.input_filename
        f = open(input_filename, 'w')
        f.write("&INDATA\n")
        for item in namelist.keys():
            if namelist[item] is not None:
                if (not isinstance(namelist[item], 
                                   (list, tuple, np.ndarray, str))):
                    f.write(item + "=" + str(namelist[item]) + "\n")
                elif (isinstance(namelist[item], str)):
                    f.write(item + "=" + "'" + str(namelist[item]) + "'" + "\n")
                elif (item not in ['rbc', 'zbs']):
                    f.write(item + "=")
                    f.writelines(map(lambda x: str(x) + " ", namelist[item]))
                    f.write("\n")
        # Write RBC
        for imn in range(self.mnmax):
            if (self.rbc[imn] != 0):
                f.write('RBC(' + str(int(self.xn[imn])) + "," \
                     + str(int(self.xm[imn])) + ") = " \
                     + str(self.rbc[imn]) + '\n')
        # Write ZBS
        for imn in range(self.mnmax):
            if (self.zbs[imn] != 0):
                f.write('ZBS(' + str(int(self.xn[imn])) + "," \
                     + str(int(self.xm[imn])) + ") = " \
                     + str(self.zbs[imn]) + '\n')

        f.write("/\n")
        f.close()
        
    def axis_position(self):
        """
        Computes position of initial axis guess
        
        Returns:
            R0 (float array): Radius coordinate of axis on toroidal grid
            Z0 (float array): Height coordinate of axis on toroidal grid
            
        """
        R0 = np.zeros(self.nzeta)
        Z0 = np.zeros(self.nzeta)
        
        if isinstance(self.raxis, (np.ndarray, list)):
            length = len(self.raxis)
        else:
            length = 1
            self.raxis = [self.raxis]
        for im in range(length):
            angle = -self.nfp * im * self.zetas
            R0 += self.raxis[im] * np.cos(angle)

        if isinstance(self.zaxis, (np.ndarray, list)):
            length = len(self.zaxis)
        else:
            length = 1
            self.zaxis = [self.zaxis]

        for im in range(length):
            angle = -self.nfp * im * self.zetas
            Z0 += self.zaxis[im] * np.sin(angle)
     
        return R0, Z0
    
    def test_axis(self):
        """
        Given defining boundary shape, determine if axis guess lies
            inside boundary
            
        Returns:
            inSurface (bool): True if axis guess lies within boundary
            
        """
        [X, Y, Z, R] = self.position()
        [R0, Z0] = self.axis_position()
        
        inSurface = np.zeros(self.nzeta)
        for izeta in range(self.nzeta):
            inSurface[izeta] = point_in_polygon(R[izeta, :], Z[izeta, :], 
                                                R0[izeta], Z0[izeta])
        inSurface = np.all(inSurface)
        return inSurface
      
    def modify_axis(self):
        """
        Given defining boundary shape, attempts to find axis guess which lies
            within the boundary
            
        Returns:
            axisSuccess (bool): True if suitable axis was obtained
            
        """

        logger = logging.getLogger(__name__)
        [X, Y, Z, R] = self.position()
        zetas = self.zetas
          
        angle_1s = np.linspace(0, 2*np.pi, 20)
        angle_2s = np.linspace(-np.pi, np.pi, 20)
        angle_1s = np.delete(angle_1s, -1)
        angle_2s = np.delete(angle_2s, -1)
        inSurface = False
        for angle_1 in angle_1s:
            for angle_2 in angle_2s:
                if (angle_1 != angle_2 and angle_1 != (angle_2+2*np.pi)):
                    R0_angle1 = np.zeros(np.shape(zetas))
                    Z0_angle1 = np.zeros(np.shape(zetas))
                    R0_angle2 = np.zeros(np.shape(zetas))
                    Z0_angle2 = np.zeros(np.shape(zetas))
                    for im in range(self.mnmax):
                        angle = self.xm[im] * angle_1 - self.nfp * \
                                self.xn[im] * zetas
                        R0_angle1 = R0_angle1 + self.rbc[im] * np.cos(angle)
                        Z0_angle1 = Z0_angle1 + self.zbs[im] * np.sin(angle)
                        angle = self.xm[im] * angle_2 - self.nfp * \
                                self.xn[im] * zetas
                        R0_angle2 = R0_angle2 + self.rbc[im] * np.cos(angle)
                        Z0_angle2 = Z0_angle2 + self.zbs[im] * np.sin(angle)
                    Raxis = 0.5*(R0_angle1 + R0_angle2)
                    Zaxis = 0.5*(Z0_angle1 + Z0_angle2)
                    # Check new axis shape
                    inSurfaces = np.zeros(self.nzeta)
                    for izeta in range(self.nzeta):
                        inSurfaces[izeta] = point_in_polygon(
                                R[izeta, :], Z[izeta, :], Raxis[izeta], 
                                Zaxis[izeta])
                    inSurface = np.all(inSurfaces)
                    if (inSurface):
                        break
        if (not inSurface):
            logger.warning('Unable to find suitable axis shape in modify_axis.')
            axisSuccess = False
            return axisSuccess
        
        # Now Fourier transform 
        raxis_cc = np.zeros(self.ntor)
        zaxis_cs = np.zeros(self.ntor)
        for n in range(self.ntor):
            angle = -n * self.nfp * zetas
            raxis_cc[n] = np.sum(Raxis * np.cos(angle)) / \
                    np.sum(np.cos(angle)**2)
            if (n > 0):
                zaxis_cs[n] = np.sum(Zaxis * np.sin(angle)) / \
                        np.sum(np.sin(angle)**2)
        self.raxis = raxis_cc
        self.zaxis = zaxis_cs
        axisSuccess = True
        return axisSuccess
    
    def test_boundary(self,theta=None,zeta=None):
        import matplotlib.pyplot as plt
        if (theta is None or zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            ntheta = self.ntheta
            nzeta = self.nzeta
        else:
            ntheta = len(theta[0,:])
            nzeta = len(zeta[:,0])
        [X,Y,Z,R] = self.position(theta=theta,zeta=zeta)
        if np.any(R<0):
            return True
        # Compute izeta of intersection
        izeta = surface_intersect(R,Z)
        if (izeta == -1):
            return False
        else:
            if self.verbose:
                print('Intersection at izeta = '+str(izeta))
            return True
            
    def radius_derivatives(self, xm_sensitivity, xn_sensitivity, theta=None, 
                           zeta=None):
        """
        Computes derivatives of radius with respect to boundary harmoncics
            (rbc,zbs)
            
        Args:
            xm_sensitivity (int array): poloidal modes for derivative
            evaluation
            xn_sensitivity (int array): toroidal modes for derivative
            evaluation
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
        Returns:
            dRdrmnc (float array): derivative of radius with respect to rbc
            dZdzmns (float array): derivative of radius with respect to zbs
            
        """
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            raise ValueError('Error! Incorrect dimensions for theta and zeta '
                         'in radius_derivatives.')
        [dXdrmnc, dYdrmnc, dZdzmns, dRdrmnc] = \
            self.position_derivatives_surface(xm_sensitivity, \
                                        xn_sensitivity, theta=theta, zeta=zeta)
        dRdzmns = np.zeros(np.shape(dRdrmnc))
        return dRdrmnc, dRdzmns
      
    def volume_derivatives(self, xm_sensitivity, xn_sensitivity, theta=None, 
                           zeta=None):
        """
        Computes derivatives of volume with respect to boundary harmoncics
            (rbc,zbs)
            
        Args:
            xm_sensitivity (int array): poloidal modes for derivative
                evaluation
            xn_sensitivity (int array): toroidal modes for derivative
                evaluation
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
        Returns:
            dRdrmnc (float array): derivative of volume with respect to rbc
            dZdzmns (float array): derivative of volume with respect to zbs
            
        """
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            raise ValueError('Error! Incorrect dimensions for theta and zeta '
                         'in volume_derivatives.')
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta, zeta)
        [X, Y, Z, R] = self.position(theta, zeta)
        [Nx, Ny, Nz] = self.normal(theta, zeta)
        [dNxdrmnc, dNxdzmns, dNydrmnc, dNydzmns, dNzdrmnc] = \
                self.normal_derivatives(xm_sensitivity, \
                                        xn_sensitivity, theta=theta, zeta=zeta)
        [dXdrmnc, dYdrmnc, dZdzmns, dRdrmnc] = \
            self.position_derivatives_surface(xm_sensitivity, \
                                        xn_sensitivity, theta=theta, zeta=zeta)
        [d2Rdthetadrmnc, d2Xdthetadrmnc, d2Ydthetadrmnc, d2Zdthetadzmns, \
            d2Rdzetadrmnc, d2Xdzetadrmnc, d2Ydzetadrmnc, d2Zdzetadzmns] = \
            self.position_first_derivatives_surface(xm_sensitivity, \
                                        xn_sensitivity, theta=theta, zeta=zeta)
          
        dvolumedrmnc = -np.tensordot(dNzdrmnc, Z,axes=2) * self.dtheta * \
                self.dzeta * self.nfp
        dvolumedzmns = -(np.tensordot(dZdzmns, Nz, axes=2)) * self.dtheta * \
                self.dzeta * self.nfp
        return dvolumedrmnc, dvolumedzmns
      
    def area_derivatives(self, xm_sensitivity, xn_sensitivity, theta=None, 
                         zeta=None):
        """
        Computes derivatives of area of boundary with respect to boundary
            harmoncics (rbc,zbs)
            
        Args:
            xm_sensitivity (int array): poloidal modes for derivative
                evaluation
            xn_sensitivity (int array): toroidal modes for derivative
                evaluation
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
        Returns:
            dareadrmnc (float array): derivative of area with respect to rbc
            dareadzmns (float array): derivative of area with respect to zbs
            
        """
        [dNdrmnc, dNdzmns] = self.jacobian_derivatives(
                xm_sensitivity, xn_sensitivity, theta, zeta)
        dareadrmnc = np.sum(dNdrmnc, axis=(1, 2)) * self.dtheta * \
                self.dzeta * self.nfp
        dareadzmns = np.sum(dNdzmns, axis=(1, 2)) * self.dtheta * \
                self.dzeta * self.nfp
        return dareadrmnc, dareadzmns
        # The above 3 statements could be rewritten as 
        # return map(lambda x: np.sum(x, axis=(1, 2)) * self.dtheta * \
        #        self.dzeta * self.nfp, [dNdrmnc, dNdzmns])
        # or for readability
        # def some_fn(x):  # Choose appropriate name for some_fn
        #    return np.sum(x, axis=(1, 2)) * self.dtheta * self.dzeta * self.nfp
        # return map(some_fn, [dNdrmnc, dNdzmns])
        
    def normal_derivatives(self, xm_sensitivity, xn_sensitivity, theta=None, 
                             zeta=None):
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            raise ValueError('Error! Incorrect dimensions for theta and zeta '
                         'in normal_derivatives.')
            
        [dXdtheta, dXdzeta, dYdtheta, dYdzeta, dZdtheta, dZdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta, zeta)
        [d2Rdthetadrmnc, d2Xdthetadrmnc, d2Ydthetadrmnc, d2Zdthetadzmns, \
            d2Rdzetadrmnc, d2Xdzetadrmnc, d2Ydzetadrmnc, d2Zdzetadzmns] = \
            self.position_first_derivatives_surface(xm_sensitivity, \
                                        xn_sensitivity, theta=theta, zeta=zeta)
        dNxdrmnc = -d2Ydzetadrmnc * dZdtheta[np.newaxis,:,:] \
                + d2Ydthetadrmnc * dZdzeta[np.newaxis,:,:]
        dNxdzmns = -dYdzeta[np.newaxis,:,:] * d2Zdthetadzmns \
                + dYdtheta[np.newaxis,:,:] * d2Zdzetadzmns
        dNydrmnc = -dZdzeta[np.newaxis,:,:] * d2Xdthetadrmnc \
                + dZdtheta[np.newaxis,:,:] * d2Xdzetadrmnc
        dNydzmns = -d2Zdzetadzmns * dXdtheta[np.newaxis,:,:] \
                + d2Zdthetadzmns * dXdzeta[np.newaxis,:,:]
        dNzdrmnc = -d2Xdzetadrmnc * dYdtheta[np.newaxis,:,:] \
                - dXdzeta[np.newaxis,:,:] * d2Ydthetadrmnc \
                + d2Xdthetadrmnc * dYdzeta[np.newaxis,:,:] \
                + dXdtheta[np.newaxis,:,:] * d2Ydzetadrmnc

        return dNxdrmnc, dNxdzmns, dNydrmnc, dNydzmns, dNzdrmnc

    # Check that theta and zeta are of correct size
    # Shape of theta and zeta must be the same
    def jacobian_derivatives(self, xm_sensitivity, xn_sensitivity, theta=None, 
                             zeta=None):
        """
        Computes derivatives of surface Jacobian with respect to
            boundary harmoncics (rbc,zbs)
            
        Args:
            xm_sensitivity (int array): poloidal modes for derivative
                evaluation
            xn_sensitivity (int array): toroidal modes for derivative
                evaluation
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
        Returns:
            dNdrmnc (float array): derivative of jacobian with respect to rbc
            dNdzmns (float array): derivative of jacobian with respect to zbs
            
        """
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            raise ValueError('Error! Incorrect dimensions for theta and zeta '
                         'in jacobian_derivatives.')
 
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta, zeta)
        [Nx, Ny, Nz] = self.normal(theta, zeta)
        N = np.sqrt(Nx*Nx + Ny*Ny + Nz*Nz)
        nx = Nx / N
        ny = Ny / N
        nz = Nz / N
        
        [d2Rdthetadrmnc, d2Xdthetadrmnc, d2Ydthetadrmnc, d2Zdthetadzmns, \
            d2Rdzetadrmnc, d2Xdzetadrmnc, d2Ydzetadrmnc, d2Zdzetadzmns] = \
            self.position_first_derivatives_surface(xm_sensitivity, \
                                        xn_sensitivity, theta=theta, zeta=zeta)
        
        [dNxdrmnc, dNxdzmns, dNydrmnc, dNydzmns, dNzdrmnc] = \
            self.normal_derivatives(xm_sensitivity, \
                                        xn_sensitivity, theta=theta, zeta=zeta)

        dNdrmnc = dNxdrmnc*nx[np.newaxis,:,:] + dNydrmnc*ny[np.newaxis,:,:] \
            + dNzdrmnc*nz[np.newaxis,:,:] 
        dNdzmns = dNxdzmns*nx[np.newaxis,:,:] + dNydzmns*ny[np.newaxis,:,:]
        return dNdrmnc, dNdzmns
    
    def position_first_derivatives_surface(self, xm_sensitivity, xn_sensitivity, \
                                           theta=None, zeta=None):
        logger = logging.getLogger(__name__)
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            raise InputError('Error! Incorrect dimensions for theta and zeta '
                         'in position_firs_derivatives_surface.')
            
        angle = xm_sensitivity[:,np.newaxis,np.newaxis] * theta[np.newaxis,:,:] - \
                self.nfp * xn_sensitivity[:,np.newaxis,np.newaxis] * zeta[np.newaxis,:,:]
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        cos_zeta = np.cos(zeta[np.newaxis,:,:])
        sin_zeta = np.sin(zeta[np.newaxis,:,:])
        
        d2Rdthetadrmnc = -xm_sensitivity[:,np.newaxis,np.newaxis] * sin_angle
        d2Xdthetadrmnc = d2Rdthetadrmnc * cos_zeta
        d2Ydthetadrmnc = d2Rdthetadrmnc * sin_zeta  
        d2Zdthetadzmns = xm_sensitivity[:,np.newaxis,np.newaxis] * cos_angle
        d2Rdzetadrmnc = self.nfp * xn_sensitivity[:,np.newaxis,np.newaxis] \
                    * sin_angle
        d2Xdzetadrmnc = d2Rdzetadrmnc * cos_zeta - cos_angle * sin_zeta
        d2Ydzetadrmnc = d2Rdzetadrmnc * sin_zeta + cos_angle * cos_zeta
        d2Zdzetadzmns = -self.nfp * xn_sensitivity[:,np.newaxis,np.newaxis] \
                    * cos_angle
        
        return d2Rdthetadrmnc, d2Xdthetadrmnc, d2Ydthetadrmnc, d2Zdthetadzmns, \
            d2Rdzetadrmnc, d2Xdzetadrmnc, d2Ydzetadrmnc, d2Zdzetadzmns
    
    def global_curvature(self,theta=None,zeta=None):
        if (theta is None or zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            dtheta = self.dtheta
            dzeta = self.dzeta
            ntheta = self.ntheta
            nzeta = self.nzeta
        else:            
            dtheta = theta[0,1]-theta[0,0]
            dzeta = zeta[1,0]-zeta[0,0]
            ntheta = len(theta[0,:])
            nzeta = len(zeta[:,0])
        [X, Y, Z, R] = self.position(theta=theta,zeta=zeta)
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dZdtheta, dZdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta=theta,zeta=zeta)
        dldtheta = np.sqrt(dRdtheta**2 + dZdtheta**2)
        tR = dRdtheta/dldtheta
        tz = dZdtheta/dldtheta
        global_curvature_radius = global_curvature_surface(R,Z,tR,tz)

        return global_curvature_radius

    def summed_radius_constraint(self,R_min=0.2,exp_weight=0.01,ntheta=None,\
                                 nzeta=None):
        if (ntheta == None or nzeta == None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            dtheta = self.dtheta
            dzeta = self.dzeta
            ntheta = self.ntheta
            nzeta = self.nzeta
        else:            
            [theta, zeta, dtheta, dzeta] = self.init_grid(ntheta,nzeta)
        [X, Y, Z, R] = self.position(theta=theta,zeta=zeta)
        normalized_jacobian = self.normalized_jacobian(theta=theta,zeta=zeta)
        constraint_fun = np.exp(-(R-R_min)**2/exp_weight**2)
        return np.sum(constraint_fun*normalized_jacobian)*dtheta*dzeta
    
    def summed_radius_constraint_derivatives(self,xm_sensitivity,xn_sensitivity,\
                                    R_min=0.2,exp_weight=0.01,ntheta=None,nzeta=None):
        if (ntheta == None or nzeta == None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            dtheta = self.dtheta
            dzeta = self.dzeta
            ntheta = self.ntheta
            nzeta = self.nzeta
        else:            
            [theta, zeta, dtheta, dzeta] = self.init_grid(ntheta,nzeta)
        [X, Y, Z, R] = self.position(theta=theta,zeta=zeta)
        normalized_jacobian = self.normalized_jacobian(theta=theta,zeta=zeta)
        [dnormalized_jacobiandrmnc, dnormalized_jacobiandzmns] = \
            self.normalized_jacobian_derivatives(xm_sensitivity,xn_sensitivity,\
                                                  theta=theta,zeta=zeta)
        [dRdrmnc, dRdzmns] = self.radius_derivatives(xm_sensitivity, \
                                xn_sensitivity, theta=None, zeta=None)
        constraint_fun = np.exp(-(R-R_min)**2/exp_weight**2)
        dconstraintdrmnc = -(2*(R[np.newaxis,:,:]-R_min)/exp_weight**2) \
            * constraint_fun[np.newaxis,:,:] * dRdrmnc
        dconstraintdzmns = -(2*(R[np.newaxis,:,:]-R_min)/exp_weight**2) \
            * constraint_fun[np.newaxis,:,:] * dRdzmns
        return np.sum(dconstraintdrmnc*normalized_jacobian[np.newaxis,:,:],axis=(1,2))*dtheta*dzeta \
            +  np.sum(constraint_fun[np.newaxis,:,:]*dnormalized_jacobiandrmnc,axis=(1,2))*dtheta*dzeta,\
            np.sum(dconstraintdzmns*normalized_jacobian[np.newaxis,:,:],axis=(1,2))*dtheta*dzeta \
            +  np.sum(constraint_fun[np.newaxis,:,:]*dnormalized_jacobiandzmns,axis=(1,2))*dtheta*dzeta
    
    def summed_proximity(self,ntheta=None,nzeta=None,min_curvature_radius=0.2,
                        exp_weight=0.01):
        """
        Constraint function integrated over the two angles
        """
        if (ntheta == None or nzeta == None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            dtheta = self.dtheta
            dzeta = self.dzeta
            ntheta = self.ntheta
            nzeta = self.nzeta
        else:            
            [theta, zeta, dtheta, dzeta] = self.init_grid(ntheta,nzeta)
        this_proximity = self.proximity(ntheta=ntheta,nzeta=nzeta,\
                        min_curvature_radius=min_curvature_radius,\
                        exp_weight=exp_weight)
        normalized_jacobian = self.normalized_jacobian(theta=theta,zeta=zeta)
        return np.sum(this_proximity*normalized_jacobian)*dtheta*dzeta
  
    def summed_proximity_derivatives(self,xm_sensitivity,xn_sensitivity,\
                                     ntheta=None,nzeta=None,\
                                     min_curvature_radius=0.2,exp_weight=0.01):
        if (ntheta == None or nzeta == None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            dtheta = self.dtheta
            dzeta = self.dzeta
        else:            
            [theta, zeta, dtheta, dzeta] = self.init_grid(ntheta,nzeta)

        [dnormalized_jacobiandrmnc, dnormalized_jacobiandzmns] = \
            self.normalized_jacobian_derivatives(xm_sensitivity,xn_sensitivity,
                                                  theta=theta,zeta=zeta)
        [dQpdrmnc, dQpdzmns] = self.proximity_derivatives(xm_sensitivity,
                                 xn_sensitivity,ntheta=ntheta,nzeta=nzeta,
                                 min_curvature_radius=min_curvature_radius,
                                 exp_weight=exp_weight)
        normalized_jacobian = self.normalized_jacobian(theta=theta,zeta=zeta)
        this_proximity = self.proximity(ntheta=ntheta,nzeta=nzeta,\
                                        min_curvature_radius=min_curvature_radius,
                                        exp_weight=exp_weight)
        return np.sum(dQpdrmnc*normalized_jacobian[:,np.newaxis,:],axis=(0,2))*dtheta*dzeta \
            +  np.sum(this_proximity[np.newaxis,:,:]*dnormalized_jacobiandrmnc,axis=(1,2))*dtheta*dzeta, \
               np.sum(dQpdzmns*normalized_jacobian[:,np.newaxis,:],axis=(0,2))*dtheta*dzeta \
            +  np.sum(this_proximity[np.newaxis,:,:]*dnormalized_jacobiandzmns,axis=(1,2))*dtheta*dzeta

    def proximity(self,derivatives=False,ntheta=None,nzeta=None,\
                  min_curvature_radius=0.2,exp_weight=0.01):
        if (ntheta == None or nzeta == None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            dtheta = self.dtheta
            dzeta = self.dzeta
            ntheta = self.ntheta
            nzeta = self.nzeta
        else:            
            [theta, zeta, dtheta, dzeta] = self.init_grid(ntheta,nzeta)
        [X, Y, Z, R] = self.position(theta=theta,zeta=zeta)
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dZdtheta, dZdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta=theta,zeta=zeta)
        dldtheta = np.sqrt(dRdtheta**2 + dZdtheta**2)
        tR = dRdtheta/dldtheta
        tz = dZdtheta/dldtheta
        if derivatives:
            [Qp, dQpdp1, dQpdp2, dQpdtau2, dQpdlprime] = \
                proximity_surface(R,Z,tR,tz,dldtheta,exp_weight=exp_weight,
                     derivatives=True,min_curvature_radius=min_curvature_radius)
            return dtheta*Qp, dtheta*dQpdp1, dtheta*dQpdp2, dtheta*dQpdtau2,\
                   dtheta*dQpdlprime 
        else:
            [Qp, dQpdp1, dQpdp2, dQpdtau2, dQpdlprime] = \
                proximity_surface(R,Z,tR,tz,dldtheta,\
                                  min_curvature_radius=min_curvature_radius,\
                                  exp_weight=exp_weight,derivatives=False)
            return dtheta*Qp

    def proximity_derivatives(self,xm_sensitivity,xn_sensitivity,ntheta=None,\
                              nzeta=None,min_curvature_radius=0.2,\
                              exp_weight=0.01):
        if (ntheta == None or nzeta == None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            dtheta = self.dtheta
            dzeta = self.dzeta
        else:            
            [theta, zeta, dtheta, dzeta] = self.init_grid(ntheta,nzeta)
            
        [Qp, dQpdp1, dQpdp2, dQpdtau2, dQpdlprime] = \
            self.proximity(derivatives=True,ntheta=ntheta,nzeta=nzeta,\
                           min_curvature_radius=min_curvature_radius,\
                           exp_weight=exp_weight)
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dZdtheta, dZdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta=theta,zeta=zeta)
        lprime = np.sqrt(dRdtheta**2 + dZdtheta**2)
        
        [dQpdrmnc, dQpdzmns] = proximity_derivatives_func(theta,zeta,self.nfp,\
                                      xm_sensitivity,xn_sensitivity,dQpdp1,\
                                      dQpdp2,dQpdtau2,dQpdlprime,dRdtheta,\
                                      dZdtheta,lprime)

        return dQpdrmnc, dQpdzmns