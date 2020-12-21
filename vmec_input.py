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
        self.input_filename = input_filename
        self.directory = os.getcwd()

        # Read items from Fortran namelist
        nml = f90nml.read(input_filename)
        nml = nml.get("indata")
        self.nfp = nml.get("nfp")
        mpol_input = nml.get("mpol")
        if (mpol_input < 2):
            raise ValueError('Error! mpol must be > 1.')
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
            nzeta (int) : number of toroidal grid points per period
            
        Returns :
            thetas_2d (np array) : poloidal angle on (nzeta,ntheta) grid
            zetas_2d (np array) : toroidal angle on (nzeta,ntheta) grid
            dtheta (float) : grid spacing on poloidal grid 
            dzeta (float) : grid spacing on toroidal grid
        """
        thetas = np.linspace(0, 2 * np.pi, ntheta + 1)
        zetas = np.linspace(0,2 * np.pi / self.nfp, nzeta + 1)
        thetas = np.delete(thetas, -1)
        zetas = np.delete(zetas, -1)
        [thetas_2d, zetas_2d] = np.meshgrid(thetas, zetas)
        dtheta = thetas[1] - thetas[0]
        dzeta = zetas[1] - zetas[0]
        
        return thetas_2d, zetas_2d, dtheta, dzeta
        
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
        [x, y, z, R] = self.position(theta, zeta)
        volume = abs(np.sum(z * Nz)) * self.dtheta * self.dzeta * self.nfp
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
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            raise ValueError('Error! Incorrect dimensions for theta and zeta in '
                         'normalized_jacobian_derivatives.')
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:, 0])
            dim2 = len(theta[0, :])
        else:
            raise ValueError('Error! Incorrect dimensions for theta and zeta in '
                         'normalized_jacobian_derivatives.')
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
            x (float array): x coordinate on angular grid
            y (float array): y coordinate on angular grid
            z (float array): z coordinate on angular grid
            R (float array): height coordinate on angular grid
        """
        if (theta is None and zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
        elif (np.array(theta).shape != np.array(zeta).shape):
            raise ValueError(
                'Error! Incorrect dimensions for theta and zeta in position.')
      
        R = cosine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,self.rbc)        
        z = sine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,self.zbs)
        
        x = R * np.cos(zeta)
        y = R * np.sin(zeta)
        return x, y, z, R
    
    def position_derivatives_surface(self, xm_sensitivity, xn_sensitivity, \
                                         theta=None, zeta=None):
        """
        Computes derivative of position vector with respect to boundary 
        harmonics
        
        Args :
            xm_sensitivity (int array): poloidal modes for derivative
                evaluation
            xn_sensitivity (int array): toroidal modes for derivative
                evaluation
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)

        Returns :
            dxdrmnc (float array) : derivative of x with respect to rmnc on 
                (nzeta,ntheta) grid
            dydrmnc (float array) : derivative of y with respect to rmnc on 
                (nzeta,ntheta) grid
            dzdzmns (float array) : derivative of z with respect to rmnc on 
                (nzeta,ntheta) grid
            dRdrmnc (float array) : derivative of R with respect to rmnc on 
                (nzeta,ntheta) grid
        
        """
        if (theta is None and zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
        elif (np.array(theta).shape != np.array(zeta).shape):
            raise ValueError(
                'Error! Incorrect dimensions for theta and zeta in position.')
      
        angle = xm_sensitivity[:,np.newaxis,np.newaxis]*theta[np.newaxis,:,:] \
         - self.nfp*xn_sensitivity[:,np.newaxis,np.newaxis]*zeta[np.newaxis,:,:]
        dRdrmnc = np.cos(angle)
        dzdzmns = np.sin(angle)
        dxdrmnc = dRdrmnc*np.cos(zeta[np.newaxis,:,:])
        dydrmnc = dRdrmnc*np.sin(zeta[np.newaxis,:,:])

        return dxdrmnc, dydrmnc, dzdzmns, dRdrmnc
      
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
            d2zdtheta2 (float array): second derivative of height wrt poloidal
                angle
            d2zdzeta2 (float array): second derivative of height wrt toroidal
                angle
            d2zdthetadzeta (float array): second derivative of height wrt
                toroidal and poloidal angles

        """
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
        d2zdtheta2 = -sine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,\
                                 self.xm*self.xm*self.zbs)
        d2Rdzeta2 = -cosine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,\
                     float(self.nfp)*float(self.nfp)*self.xn*self.xn*self.rbc)
        d2zdzeta2 = -sine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,\
                     float(self.nfp)*self.nfp*self.xn*self.xn*self.zbs)
        d2Rdthetadzeta = cosine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,\
                         float(self.nfp)*self.xm*self.xn*self.rbc)
        d2zdthetadzeta = sine_IFT(self.xm,self.xn,float(self.nfp),theta,zeta,\
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
                d2ydthetadzeta, d2zdtheta2, d2zdzeta2, d2zdthetadzeta, \
               d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta
    
    def position_second_derivatives_surface(self, xm_sensitivity, xn_sensitivity, 
                                            theta=None, zeta=None):
        """
        Computes mixed derivatives with of position vector wrt angles
            and boundary harmonics 
        
        Args :
            xm_sensitivity (int array): poloidal modes for derivative
                evaluation
            xn_sensitivity (int array): toroidal modes for derivative
                evaluation
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)

        Returns :
            d3xdtheta2drmnc (float array) : derivative of x wrt poloidal angle
                and rmnc
            d3ydtheta2drmnc (float array) : derivative of y wrt poloidal angle
                and rmnc
            d3zdtheta2dzmns (float array) : derivative of z wrt poloidal angle
                and zmns
            d3xdzeta2drmnc (float array) : derivative of x wrt toroidal angle
                and rmnc
            d3ydzeta2drmnc (float array) : derivative of y wrt toroidal angle
                and rmnc
            d3zdzeta2dzmns (float array) : derivative of z wrt toroidal angle
                and zmns
            d3xdthetadzetadrmnc (float array) : derivative of x wrt 
                poloidal, toroidal angle, and rmnc
            d3ydthetadzetadrmnc (float array) : derivative of y wrt 
                poloidal, toroidal angle, and rmnc
            d3ydthetadzetadzmns (float array) : derivative of y wrt 
                poloidal, toroidal angle, and zmns  
            d3zdthetadzetadzmns (float array) : derivative of z wrt
                poloidal, toroidal angle, and zmns  
        """
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            raise ValueError('Error! Incorrect dimensions for theta and zeta '
                         'in position_second_derivatives_surface.')
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:,0])
            dim2 = len(theta[0,:])
        else:
            raise ValueError('Error! Incorrect dimensions for theta and zeta '
                         'in position_second_derivatives_surface.')

        angle = xm_sensitivity[:,np.newaxis,np.newaxis] * theta[np.newaxis,:,:] \
                   - self.nfp * xn_sensitivity[:,np.newaxis,np.newaxis] \
                   * zeta[np.newaxis,:,:]
        d3Rdtheta2drmnc = -xm_sensitivity[:,np.newaxis,np.newaxis] * \
                           xm_sensitivity[:,np.newaxis,np.newaxis] * \
                           np.cos(angle)
        d3zdtheta2dzmns = -xm_sensitivity[:,np.newaxis,np.newaxis] * \
                           xm_sensitivity[:,np.newaxis,np.newaxis] * \
                           np.sin(angle)
        d3Rdzeta2drmnc = -  xn_sensitivity[:,np.newaxis,np.newaxis] * self.nfp \
                          * xn_sensitivity[:,np.newaxis,np.newaxis] * self.nfp \
                          * np.cos(angle)
        d3zdzeta2dzmns = -xn_sensitivity[:,np.newaxis,np.newaxis] * self.nfp \
                          * xn_sensitivity[:,np.newaxis,np.newaxis] * self.nfp \
                          * np.sin(angle)
        d3Rdthetadzetadrmnc = xm_sensitivity[:,np.newaxis,np.newaxis] * \
                           xn_sensitivity[:,np.newaxis,np.newaxis] * self.nfp \
                          * np.cos(angle)
        d3zdthetadzetadzmns = xm_sensitivity[:,np.newaxis,np.newaxis] * \
                           xn_sensitivity[:,np.newaxis,np.newaxis] * self.nfp \
                          * np.sin(angle)

        [d2Rdthetadrmnc, d2xdthetadrmnc, d2ydthetadrmnc, d2zdthetadzmns, \
            d2Rdzetadrmnc, d2xdzetadrmnc, d2ydzetadrmnc, d2zdzetadzmns] = \
            self.position_first_derivatives_surface(xm_sensitivity, \
                                        xn_sensitivity, theta=theta, zeta=zeta)
        [dxdrmnc, dydrmnc, dzdzmns, dRdrmnc] = \
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
      
        return d3xdtheta2drmnc, d3ydtheta2drmnc, d3zdtheta2dzmns, \
               d3xdzeta2drmnc, d3ydzeta2drmnc, d3zdzeta2dzmns, \
               d3xdthetadzetadrmnc, d3ydthetadzetadrmnc, \
               d3zdthetadzetadzmns
        
    def surface_curvature(self, theta=None, zeta=None):
        """
        Computes mean curvature of boundary surface
        
        Args:
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)

        Returns:
            H (float array): mean curvature on (nzeta,ntheta) grid
            K (float array): Gaussian curvature on (nzeta,ntheta) grid
            kappa1 (float array) : first principal curvature on (nzeta,ntheta)
                grid (<= kappa2)
            kappa2 (float array) : second principal curvature on (nzeta,ntheta)
                grid (=> kappa1)
        """
        [E, F, G, e, f, g] = self.metric_tensor(theta,zeta)
        
        H = (e*G -2*f*F + g*E)/(E*G - F*F)
        K = (e*g - f*f)/(E*G - F*F)
        kappa1 = H + np.sqrt(H*H - K)
        kappa2 = H - np.sqrt(H*H - K)
        return H, K, kappa1, kappa2
    
    def metric_tensor(self, theta=None, zeta=None):
        """
        Computes first and second fundamental forms on (nzeta,ntheta) grid
        
        Args:
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)

        Returns : 
            E (np array) : drdtheta \cdot drdtheta on (nzeta,ntheta) grid
            F (np array) : drdtheta \cdot drdzeta on (nzeta,ntheta) grid
            G (np array) : drdzeta \cdot drdzeta on (nzeta,ntheta) grid
            e (np array) : n \cdot d2rdtheta2 on (nzeta,ntheta) grid
            f (np array) : n \cdot d2rdthetadzeta on (nzeta,ntheta) grid
            g (np array) : n \cdot d2rdzeta2 on (nzeta,ntheta) grid
        """
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta, zeta)
        [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, \
                d2ydthetadzeta, d2zdtheta2, d2zdzeta2, d2zdthetadzeta, \
                d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta] = \
                self.position_second_derivatives(theta, zeta)
        norm_normal = self.jacobian(theta=theta,zeta=zeta)
        [Nx, Ny, Nz] = self.normal(theta=theta,zeta=zeta)

        nx = Nx / norm_normal
        ny = Ny / norm_normal
        nz = Nz / norm_normal
        E = dxdtheta * dxdtheta + dydtheta * dydtheta + dzdtheta * dzdtheta
        F = dxdtheta * dxdzeta + dydtheta * dydzeta + dzdtheta * dzdzeta
        G = dxdzeta * dxdzeta + dydzeta * dydzeta + dzdzeta * dzdzeta
        e = nx * d2xdtheta2 + ny * d2ydtheta2 + nz * d2zdtheta2
        f = nx * d2xdthetadzeta + ny * d2ydthetadzeta + nz * d2zdthetadzeta
        g = nx * d2xdzeta2 + ny * d2ydzeta2 + nz * d2zdzeta2

        return E, F, G, e, f, g
    
    def surface_curvature_derivatives(self, xm_sensitivity, xn_sensitivity,\
                                      theta=None, zeta=None):
        """
        Computes derivatives of surface curvatures with respect to boundary
            harmonics - dimensions (mnmax_sensitivity,nzeta,ntheta)

        Args:
            xm_sensitivity (int array): poloidal modes for derivative
                evaluation
            xn_sensitivity (int array): toroidal modes for derivative
                evaluation
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)

        Returns:
            dHdrmnc (np array) : derivative of mean curvature wrt rmnc
            dHdzmns (np array) : derivative of mean curvature wrt zmns
            dKdrmnc (np array) : derivative of Gaussian curvature wrt rmnc
            dKdzmns (np array) : derivative of Gaussian curvature wrt zmns
            dkappa1drmnc (np array) : derivative of first principal curvature
                wrt rmnc
            dkappa1dzmns (np array) : derivative of first princiapl curvature
                wrt zmns
            dkappa2drmnc (np array) : derivative of second principal curvature
                wrt rmnc
            dkappa2dzmns (np array) : derivative of second princiapl curvature
                wrt zmns            
        """
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            raise ValueError('Error! Incorrect dimensions for theta and zeta '
                         'in volume_derivatives.')
        if (theta.ndim == 1):
            dim1 = len(theta)
            dim2 = 1
        elif (theta.ndim == 2):
            dim1 = len(theta[:,0])
            dim2 = len(theta[0,:])
        else:
            raise ValueError('Error! Incorrect dimensions for theta and zeta '
                         'in volume_derivatives.')

        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta, zeta)
        [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, \
                d2ydthetadzeta, d2zdtheta2, d2zdzeta2, d2zdthetadzeta, \
                d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta] = \
                self.position_second_derivatives(theta, zeta)
        [d2Rdthetadrmnc, d2xdthetadrmnc, d2ydthetadrmnc, d2zdthetadzmns, \
            d2Rdzetadrmnc, d2xdzetadrmnc, d2ydzetadrmnc, d2zdzetadzmns] = \
                self.position_first_derivatives_surface(xm_sensitivity, \
                            xn_sensitivity, theta=theta, zeta=zeta)
        [d3xdtheta2drmnc, d3ydtheta2drmnc, d3zdtheta2dzmns, \
               d3xdzeta2drmnc, d3ydzeta2drmnc, d3zdzeta2dzmns, \
               d3xdthetadzetadrmnc, d3ydthetadzetadrmnc, \
               d3zdthetadzetadzmns] = \
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

        dEdrmnc = 2*d2xdthetadrmnc*dxdtheta[np.newaxis,:,:] \
                + 2*d2ydthetadrmnc*dydtheta[np.newaxis,:,:] 
        dFdrmnc = d2xdthetadrmnc*dxdzeta[np.newaxis,:,:] \
                + d2xdzetadrmnc*dxdtheta[np.newaxis,:,:] \
                + d2ydthetadrmnc*dydzeta[np.newaxis,:,:] \
                + d2ydzetadrmnc*dydtheta[np.newaxis,:,:] 
        dGdrmnc = 2*d2xdzetadrmnc*dxdzeta[np.newaxis,:,:] \
                + 2*d2ydzetadrmnc*dydzeta[np.newaxis,:,:] 
        
        dEdzmns = 2*d2zdthetadzmns*dzdtheta[np.newaxis,:,:] 
        dFdzmns = d2zdthetadzmns*dzdzeta[np.newaxis,:,:] \
                + d2zdzetadzmns*dzdtheta[np.newaxis,:,:]
        dGdzmns = 2*d2zdzetadzmns*dzdzeta[np.newaxis,:,:] 
        
        dedrmnc = dnxdrmnc * d2xdtheta2 + nx * d3xdtheta2drmnc \
                + dnydrmnc * d2ydtheta2 + ny * d3ydtheta2drmnc \
                + dnzdrmnc * d2zdtheta2 
        dedzmns = dnxdzmns * d2xdtheta2 \
                + dnydzmns * d2ydtheta2 \
                + dnzdzmns * d2zdtheta2 + nz * d3zdtheta2dzmns
        dfdrmnc = dnxdrmnc * d2xdthetadzeta + nx * d3xdthetadzetadrmnc \
                + dnydrmnc * d2ydthetadzeta + ny * d3ydthetadzetadrmnc \
                + dnzdrmnc * d2zdthetadzeta 
        dfdzmns = dnxdzmns * d2xdthetadzeta \
                + dnydzmns * d2ydthetadzeta \
                + dnzdzmns * d2zdthetadzeta + nz * d3zdthetadzetadzmns
        dgdrmnc = dnxdrmnc * d2xdzeta2 + nx * d3xdzeta2drmnc \
                + dnydrmnc * d2ydzeta2 + ny * d3ydzeta2drmnc \
                + dnzdrmnc * d2zdzeta2 
        dgdzmns = dnxdzmns * d2xdzeta2  \
                + dnydzmns * d2ydzeta2  \
                + dnzdzmns * d2zdzeta2 + nz * d3zdzeta2dzmns

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
        """
        Returns surface-integrated principal curvatures
        
        \int d^2 x \, kappa_1 /(area/(4*pi*pi))
        \int d^2 x \, kappa_2 /(area/(4*pi*pi))

        Args :
            ntheta (int) : number of poloidal grid points for evaluation (optional)
            nzeta (int) : number of toroidal grid points for evaluation (optional)
            
        Returns : 
            metric1 (float) : surface-integrated first principal curvature
            metric2 (float) : surface-integrated second principal curvature
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
        [H, K, kappa1, kappa2] = self.surface_curvature(theta=theta,zeta=zeta)
        normalized_jacobian = self.normalized_jacobian(theta=theta,zeta=zeta)
        metric1 = 0.5 * np.sum(kappa1**2 * normalized_jacobian) * dtheta * dzeta
        metric2 = 0.5 * np.sum(kappa2**2 * normalized_jacobian) * dtheta * dzeta
        
        return metric1, metric2
    
    def surface_curvature_metric_integrated_derivatives(self, xm_sensitivity, 
                                     xn_sensitivity, ntheta=None, nzeta=None):
        """
        Returns derivativatives of surface-integrated principal curvatures wrt
            boundary harmonics
        
        Args :
            xm_sensitivity (int array) : poloidal modes for derivative
                evaluation
            xn_sensitivity (int array) : toroidal modes for derivative
                evaluation
            ntheta (int) : number of poloidal grid points for evaluation (optional)
            nzeta (int) : number of toroidal grid points for evaluation (optional)

        Returns :
            dmetric1drmnc (np array) : derivative of surface-integrated first 
                principal curvature wrt rmnc
            dmetric1dzmns (np array) : derivative of surface-integrated first 
                principal curvature wrt zmns
            dmetric2drmnc (np array) : derivative of surface-integrated second
                principal curvature wrt rmnc
            dmetric2dzmns (np array) : derivative of surface-integrated second
                principal curvature wrt zmns
        
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
        """
        Returns curvature penalty function
        \int d^2 x \, \exp ((kappa1**2 - max_curvature**2 )/exp_weight**2)/(area/(4*pi*pi)) + 
            \int d^2 x \, \exp ((kappa1**2 - max_curvature**2 )/exp_weight**2)/(area/(4*pi*pi))
            
        Args :
            ntheta (int) : number of poloidal grid points for evaluation (optional)
            nzeta (int) : number of toroidal grid points for evaluation (optional)
            max_curvature (float) : maximum value of curvature of penalty function
            exp_weight (float) : weighting for exponential barrier function
        Returns :
            metric (float) : curvature penalty function
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
        [H, K, kappa1, kappa2] = self.surface_curvature(theta=theta,zeta=zeta)
        normalized_jacobian = self.normalized_jacobian(theta=theta,zeta=zeta)
        metric = np.sum((np.exp((kappa1**2-max_curvature**2)/exp_weight**2) \
            + np.exp((kappa2**2-max_curvature**2)/exp_weight**2))*normalized_jacobian) \
            * dtheta * dzeta
        if (np.isinf(metric)):
            metric = 1e12
        return metric
    
    def surface_curvature_metric_derivatives(self, xm_sensitivity, xn_sensitivity, 
                                    ntheta=None, nzeta=None, max_curvature=30,
                                    exp_weight=0.01):
        """
        Derivative of curvature penalty function wrt bondary harmonics
        
        Args :
            xm_sensitivity (int array) : poloidal modes for derivative
                evaluation
            xn_sensitivity (int array) : toroidal modes for derivative
                evaluation
            ntheta (int) : number of poloidal grid points for evaluation (optional)
            nzeta (int) : number of toroidal grid points for evaluation (optional)
            max_curvature (float) : maximum value of curvature of penalty function
            exp_weight (float) : weighting for exponential barrier function

        Returns :
            dmetricdrmnc (np array) : derivative of metric wrt rmnc 
                (len = mnmax_sensitivity)
            dmetricdzmns (np array) : derivative of metric wrt zmns
                (len = mnmax_sensitivity)
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
            H (float array): mean curvature on (zeta,theta) grid
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
                (len = nzeta)
            z0 (float array): Height coordinate of axis on toroidal grid 
                (len = nzeta) 
        """
        R0 = np.zeros(self.nzeta)
        z0 = np.zeros(self.nzeta)
        
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
            z0 += self.zaxis[im] * np.sin(angle)
     
        return R0, z0
    
    def test_axis(self):
        """
        Given defining boundary shape, determine if axis guess lies
            inside boundary
            
        Returns:
            inSurface (bool): True if axis guess lies within boundary
        """
        [x, y, z, R] = self.position()
        [R0, z0] = self.axis_position()
        
        inSurface = np.zeros(self.nzeta)
        for izeta in range(self.nzeta):
            inSurface[izeta] = point_in_polygon(R[izeta, :], z[izeta, :], 
                                                R0[izeta], z0[izeta])
        inSurface = np.all(inSurface)
        return inSurface
      
    def modify_axis(self):
        """
        Given defining boundary shape, attempts to find axis guess which lies
            within the boundary. self.raxis and self.zaxis are modified if 
            successful. 
            
        Returns:
            axisSuccess (bool): True if suitable axis was obtained
        """
        [x, y, z, R] = self.position()
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
                    z0_angle1 = np.zeros(np.shape(zetas))
                    R0_angle2 = np.zeros(np.shape(zetas))
                    z0_angle2 = np.zeros(np.shape(zetas))
                    for im in range(self.mnmax):
                        angle = self.xm[im] * angle_1 - self.nfp * \
                                self.xn[im] * zetas
                        R0_angle1 = R0_angle1 + self.rbc[im] * np.cos(angle)
                        z0_angle1 = z0_angle1 + self.zbs[im] * np.sin(angle)
                        angle = self.xm[im] * angle_2 - self.nfp * \
                                self.xn[im] * zetas
                        R0_angle2 = R0_angle2 + self.rbc[im] * np.cos(angle)
                        z0_angle2 = z0_angle2 + self.zbs[im] * np.sin(angle)
                    Raxis = 0.5*(R0_angle1 + R0_angle2)
                    zaxis = 0.5*(z0_angle1 + z0_angle2)
                    # Check new axis shape
                    inSurfaces = np.zeros(self.nzeta)
                    for izeta in range(self.nzeta):
                        inSurfaces[izeta] = point_in_polygon(
                                R[izeta, :], z[izeta, :], Raxis[izeta], 
                                zaxis[izeta])
                    inSurface = np.all(inSurfaces)
                    if (inSurface):
                        break
        if (not inSurface):
            if (self.verbose):
                print('Unable to find suitable axis shape in modify_axis.')
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
                zaxis_cs[n] = np.sum(zaxis * np.sin(angle)) / \
                        np.sum(np.sin(angle)**2)
        self.raxis = raxis_cc
        self.zaxis = zaxis_cs
        axisSuccess = True
        return axisSuccess
    
    def test_boundary(self,theta=None,zeta=None):
        """
        Given a boundary, tests if boundary is self-intersecting on the 
            provided angular grid. The algorithm to do so is based on
            https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/
            
        Args :
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)

        Returns :
            intersection (boolean) : True if boundary is self-intersecting
        
        """
        if (theta is None or zeta is None):
            theta = self.thetas_2d
            zeta = self.zetas_2d
            ntheta = self.ntheta
            nzeta = self.nzeta
        else:
            ntheta = len(theta[0,:])
            nzeta = len(zeta[:,0])
        [x,y,z,R] = self.position(theta=theta,zeta=zeta)
        if np.any(R<0):
            return True
        # Compute izeta of intersection
        izeta = surface_intersect(R,z)
        if (izeta == -1):
            intersecting = False
            return intersecting
        else:
            if self.verbose:
                print('Intersection at izeta = '+str(izeta))
            intersecting = True
            return intersecting
            
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
            dzdzmns (float array): derivative of radius with respect to zbs
        """
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            raise ValueError('Error! Incorrect dimensions for theta and zeta '
                         'in radius_derivatives.')
        [dxdrmnc, dydrmnc, dzdzmns, dRdrmnc] = \
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
            dzdzmns (float array): derivative of volume with respect to zbs
            
        """
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            raise ValueError('Error! Incorrect dimensions for theta and zeta '
                         'in volume_derivatives.')
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta, zeta)
        [x, y, z, R] = self.position(theta, zeta)
        [Nx, Ny, Nz] = self.normal(theta, zeta)
        [dNxdrmnc, dNxdzmns, dNydrmnc, dNydzmns, dNzdrmnc] = \
                self.normal_derivatives(xm_sensitivity, \
                                        xn_sensitivity, theta=theta, zeta=zeta)
        [dxdrmnc, dydrmnc, dzdzmns, dRdrmnc] = \
            self.position_derivatives_surface(xm_sensitivity, \
                                        xn_sensitivity, theta=theta, zeta=zeta)
        [d2Rdthetadrmnc, d2xdthetadrmnc, d2ydthetadrmnc, d2zdthetadzmns, \
            d2Rdzetadrmnc, d2xdzetadrmnc, d2ydzetadrmnc, d2zdzetadzmns] = \
            self.position_first_derivatives_surface(xm_sensitivity, \
                                        xn_sensitivity, theta=theta, zeta=zeta)
          
        dvolumedrmnc = -np.tensordot(dNzdrmnc, z,axes=2) * self.dtheta * \
                self.dzeta * self.nfp
        dvolumedzmns = -(np.tensordot(dzdzmns, Nz, axes=2)) * self.dtheta * \
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
        
    def normal_derivatives(self, xm_sensitivity, xn_sensitivity, theta=None, 
                             zeta=None):
        """
        Computes derivatives of normal vector with respect to boundary harmonics
        
        Args : 
            xm_sensitivity (int array): poloidal modes for derivative
                evaluation
            xn_sensitivity (int array): toroidal modes for derivative
                evaluation
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)

        Returns : 
            dNxdrmnc (np array) : derivative of x component of normal vector
                wrt rmnc on (nzeta,ntheta) grid
            dNxdzmns (np array) : derivative of x component of normal vector
                wrt zmns on (nzeta,ntheta) grid
            dNydrmnc (np array) : derivative of y component of normal vector
                wrt rmnc on (nzeta,ntheta) grid
            dNydzmns (np array) : derivative of y component of normal vector
                wrt zmns on (nzeta,ntheta) grid
            dNzdrmnc (np array) : derivative of z component of normal vector
                wrt rmnc on (nzeta,ntheta) grid
            dNzdzmns (np array) : derivative of z component of normal vector
                wrt zmns on (nzeta,ntheta) grid
        """
        if (theta is None and zeta is None):
            zeta = self.zetas_2d
            theta = self.thetas_2d
        if (theta.ndim != zeta.ndim):
            raise ValueError('Error! Incorrect dimensions for theta and zeta '
                         'in normal_derivatives.')
            
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta, zeta)
        [d2Rdthetadrmnc, d2xdthetadrmnc, d2ydthetadrmnc, d2zdthetadzmns, \
            d2Rdzetadrmnc, d2xdzetadrmnc, d2ydzetadrmnc, d2zdzetadzmns] = \
            self.position_first_derivatives_surface(xm_sensitivity, \
                                        xn_sensitivity, theta=theta, zeta=zeta)
        dNxdrmnc = -d2ydzetadrmnc * dzdtheta[np.newaxis,:,:] \
                + d2ydthetadrmnc * dzdzeta[np.newaxis,:,:]
        dNxdzmns = -dydzeta[np.newaxis,:,:] * d2zdthetadzmns \
                + dydtheta[np.newaxis,:,:] * d2zdzetadzmns
        dNydrmnc = -dzdzeta[np.newaxis,:,:] * d2xdthetadrmnc \
                + dzdtheta[np.newaxis,:,:] * d2xdzetadrmnc
        dNydzmns = -d2zdzetadzmns * dxdtheta[np.newaxis,:,:] \
                + d2zdthetadzmns * dxdzeta[np.newaxis,:,:]
        dNzdrmnc = -d2xdzetadrmnc * dydtheta[np.newaxis,:,:] \
                - dxdzeta[np.newaxis,:,:] * d2ydthetadrmnc \
                + d2xdthetadrmnc * dydzeta[np.newaxis,:,:] \
                + dxdtheta[np.newaxis,:,:] * d2ydzetadrmnc

        return dNxdrmnc, dNxdzmns, dNydrmnc, dNydzmns, dNzdrmnc

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
        
        [d2Rdthetadrmnc, d2xdthetadrmnc, d2ydthetadrmnc, d2zdthetadzmns, \
            d2Rdzetadrmnc, d2xdzetadrmnc, d2ydzetadrmnc, d2zdzetadzmns] = \
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
        """
        Compute mixed derivatives of position vector with respect to angles and 
            boundary harmonics. Output has dimensions
            (mnmax_sensitivity,nzeta,ntheta)
            
        Args : 
            xm_sensitivity (int array): poloidal modes for derivative
                evaluation
            xn_sensitivity (int array): toroidal modes for derivative
                evaluation
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)
            
        Returns : 
            d2Rdthetadrmnc : derivative of R wrt poloidal angle and rmnc
            d2xdthetadrmnc : derivative of x wrt poloidal angle and rmnc
            d2ydthetadrmnc : derivative of y wrt poloidal angle and rmnc
            d2zdthetadzmns : derivative of z wrt poloidal angle and zmns
            d2Rdzetadrmnc : derivative of R wrt toroidal angle and rmnc
            d2xdzetadrmnc : derivative of x wrt toroidal angle and rmnc
            d2ydzetadrmnc : derivative of y wrt toroidal angle and rmnc
            d2zdzetadzmns : derivative of z wrt toroidal angle and zmns
        """
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
        d2xdthetadrmnc = d2Rdthetadrmnc * cos_zeta
        d2ydthetadrmnc = d2Rdthetadrmnc * sin_zeta  
        d2zdthetadzmns = xm_sensitivity[:,np.newaxis,np.newaxis] * cos_angle
        d2Rdzetadrmnc = self.nfp * xn_sensitivity[:,np.newaxis,np.newaxis] \
                    * sin_angle
        d2xdzetadrmnc = d2Rdzetadrmnc * cos_zeta - cos_angle * sin_zeta
        d2ydzetadrmnc = d2Rdzetadrmnc * sin_zeta + cos_angle * cos_zeta
        d2zdzetadzmns = -self.nfp * xn_sensitivity[:,np.newaxis,np.newaxis] \
                    * cos_angle
        
        return d2Rdthetadrmnc, d2xdthetadrmnc, d2ydthetadrmnc, d2zdthetadzmns, \
            d2Rdzetadrmnc, d2xdzetadrmnc, d2ydzetadrmnc, d2zdzetadzmns
    
    def global_curvature(self,theta=None,zeta=None):
        """
        For each cross-section, computes minimum of the S_C function 
            between each point and each other point on the cross-section
            
            S_C(x1,x2) = |x1 - x2|/(2\sqrt(1 - t(x2) \cdot (x1 - x2)/|x1-x1|))
            
        See Section 3.1 in Paul, Landreman, Antonsen 
        
        Args :
            theta (float array): poloidal grid for evaluation (optional)
            zeta (float array): toroidal grid for evaluation (optional)

        Returns : 
            global_curvature_radius (float array) : global curvature on 
                (nzeta,ntheta) grid
        """
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
        [x, y, z, R] = self.position(theta=theta,zeta=zeta)
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta=theta,zeta=zeta)
        dldtheta = np.sqrt(dRdtheta**2 + dzdtheta**2)
        tR = dRdtheta/dldtheta
        tz = dzdtheta/dldtheta
        global_curvature_radius = global_curvature_surface(R,z,tR,tz)

        return global_curvature_radius

    def summed_radius_constraint(self,R_min=0.2,exp_weight=0.01,ntheta=None,\
                                 nzeta=None):
        """
        
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
        [x, y, z, R] = self.position(theta=theta,zeta=zeta)
        normalized_jacobian = self.normalized_jacobian(theta=theta,zeta=zeta)
        constraint_fun = np.exp(-(R-R_min)**2/exp_weight**2)
        return np.sum(constraint_fun*normalized_jacobian)*dtheta*dzeta
    
    def summed_radius_constraint_derivatives(self,xm_sensitivity,xn_sensitivity,\
                                    R_min=0.2,exp_weight=0.01,ntheta=None,nzeta=None):
        """
        
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
        [x, y, z, R] = self.position(theta=theta,zeta=zeta)
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
        """
        
        """
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
        """
        
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
        [x, y, z, R] = self.position(theta=theta,zeta=zeta)
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta=theta,zeta=zeta)
        dldtheta = np.sqrt(dRdtheta**2 + dzdtheta**2)
        tR = dRdtheta/dldtheta
        tz = dzdtheta/dldtheta
        if derivatives:
            [Qp, dQpdp1, dQpdp2, dQpdtau2, dQpdlprime] = \
                proximity_surface(R,z,tR,tz,dldtheta,exp_weight=exp_weight,
                     derivatives=True,min_curvature_radius=min_curvature_radius)
            return dtheta*Qp, dtheta*dQpdp1, dtheta*dQpdp2, dtheta*dQpdtau2,\
                   dtheta*dQpdlprime 
        else:
            [Qp, dQpdp1, dQpdp2, dQpdtau2, dQpdlprime] = \
                proximity_surface(R,z,tR,tz,dldtheta,\
                                  min_curvature_radius=min_curvature_radius,\
                                  exp_weight=exp_weight,derivatives=False)
            return dtheta*Qp

    def proximity_derivatives(self,xm_sensitivity,xn_sensitivity,ntheta=None,\
                              nzeta=None,min_curvature_radius=0.2,\
                              exp_weight=0.01):
        """
        
        """
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
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, \
                dRdzeta] = self.position_first_derivatives(theta=theta,zeta=zeta)
        lprime = np.sqrt(dRdtheta**2 + dzdtheta**2)
        
        [dQpdrmnc, dQpdzmns] = proximity_derivatives_func(theta,zeta,self.nfp,\
                                      xm_sensitivity,xn_sensitivity,dQpdp1,\
                                      dQpdp2,dQpdtau2,dQpdlprime,dRdtheta,\
                                      dzdtheta,lprime)

        return dQpdrmnc, dQpdzmns