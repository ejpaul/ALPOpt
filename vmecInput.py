import numpy as np
from scipy.io import netcdf
import os
from FortranNamelistTools import *
import optimization_utils
from optimization_utils import *
import f90nml
import sys

class readVmecInput:
  
  def __init__(self,input_filename,ntheta=100,nzeta=100):
    self.input_filename = input_filename
    self.directory = os.getcwd()
    
    # Read items from Fortran namelist
    nml = f90nml.read(input_filename)
    nml = nml.get("indata")
    self.nfp = nml.get("nfp")
    mpol_input = nml.get("mpol")
    if (mpol_input<2):
      print('Error! mpol must be > 1.')
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
    
    self.update_modes(nml.get("mpol"),nml.get("ntor"))
    [self.mnmax,self.xm,self.xn] = optimization_utils.init_modes(self.mpol-1,self.ntor)

    # Returns rbc and zbs on grid of input modes 
    [self.rbc,self.zbs] = self.read_boundary_input()
    
    self.update_grids(ntheta,nzeta)
    
  def update_modes(self,mpol,ntor):
    self.mpol = mpol
    self.ntor = ntor
    [self.mnmax,self.xm,self.xn] = optimization_utils.init_modes(self.mpol-1,self.ntor)
    
  def update_grids(self,ntheta,nzeta):
    self.ntheta = ntheta
    self.nzeta = nzeta
    self.thetas = np.linspace(0,2*np.pi,ntheta+1)
    self.zetas = np.linspace(0,2*np.pi/self.nfp,nzeta+1)
    self.thetas = np.delete(self.thetas,-1)
    self.zetas = np.delete(self.zetas,-1)
    [self.thetas_2d,self.zetas_2d] = np.meshgrid(self.thetas,self.zetas)
    self.dtheta = self.thetas[1]-self.thetas[0]
    self.dzeta = self.zetas[1]-self.zetas[0]
    
    self.zetas_full = np.linspace(0,2*np.pi,self.nfp*nzeta+1)
    self.zetas_full = np.delete(self.zetas_full,-1)
    [self.thetas_2d_full,self.zetas_2d_full] = np.meshgrid(self.thetas,self.zetas_full)
    
  def read_boundary_input(self):
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
    rbc_array = np.array(self.namelist["rbc"]).T
    zbs_array = np.array(self.namelist["zbs"]).T

    rbc_array[rbc_array==None] = 0
    zbs_array[zbs_array==None] = 0
    [dim1_rbc,dim2_rbc] = np.shape(rbc_array)
    [dim1_zbs,dim2_zbs] = np.shape(zbs_array)

    [nmin_rbc,mmin_rbc,nmax_rbc,mmax_rbc] = min_max_indices_2d('rbc',self.input_filename)
    nmax_rbc = max(nmax_rbc,-nmin_rbc)
    [nmin_zbs,mmin_zbs,nmax_zbs,mmax_zbs] = min_max_indices_2d('zbs',self.input_filename)
    nmax_zbs = max(nmax_zbs,-nmin_zbs)

    rbc = np.zeros(self.mnmax)
    zbs = np.zeros(self.mnmax)
    for imn in range(self.mnmax):
        if ((self.xn[imn]-nmin_rbc) < dim1_rbc and (self.xm[imn]-mmin_rbc) < dim2_rbc \
           and (self.xn[imn]-nmin_rbc)> -1 and (self.xm[imn]-mmin_rbc)> -1):
          rbc[imn] = rbc_array[int(self.xn[imn]-nmin_rbc),int(self.xm[imn]-mmin_rbc)]
        if ((self.xn[imn]-nmin_zbs) < dim1_zbs and (self.xm[imn]-mmin_zbs) < dim2_zbs \
           and (self.xn[imn]-nmin_rbc)> -1 and (self.xm[imn]-mmin_rbc)> -1):
          zbs[imn] = zbs_array[int(self.xn[imn]-nmin_zbs),int(self.xm[imn]-mmin_zbs)]

    return rbc, zbs

  def area(self,theta=None,zeta=None):
    norm_normal = self.jacobian(theta,zeta)
    area = np.sum(norm_normal)*self.dtheta*self.dzeta*self.nfp
    return area
  
  def volume(self,theta=None,zeta=None):
    [Nx,Ny,Nz] = self.N(theta,zeta)
    [X,Y,Z,R] = self.position(theta,zeta)
    return abs(np.sum(Z*Nz))*self.dtheta*self.dzeta*self.nfp
  
  def N(self,theta=None,zeta=None):
    [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta] = \
      self.position_first_derivatives(theta,zeta)
  
    Nx = -dydzeta*dzdtheta + dydtheta*dzdzeta
    Ny = -dzdzeta*dxdtheta + dzdtheta*dxdzeta
    Nz = -dxdzeta*dydtheta + dxdtheta*dydzeta
    return Nx, Ny, Nz
  
  def position_first_derivatives(self,theta=None,zeta=None):
    if (theta is None and zeta is None):
      theta = self.thetas_2d
      zeta = self.zetas_2d
    elif (np.array(theta).shape != np.array(zeta).shape):
      print('Incorrect shape of theta and zeta in position_first_derivatives')
      sys.exit(0)
    dRdtheta = np.zeros(np.shape(theta))
    dzdtheta = np.zeros(np.shape(theta))
    dRdzeta = np.zeros(np.shape(theta))
    dzdzeta = np.zeros(np.shape(theta))
    R = np.zeros(np.shape(theta))
    for im in range(self.mnmax):
      angle = self.xm[im]*theta - self.nfp*self.xn[im]*zeta
      cos_angle = np.cos(angle)
      sin_angle = np.sin(angle)
      dRdtheta = dRdtheta - self.xm[im]*self.rbc[im]*sin_angle
      dzdtheta = dzdtheta + self.xm[im]*self.zbs[im]*cos_angle
      dRdzeta = dRdzeta + self.nfp*self.xn[im]*self.rbc[im]*sin_angle
      dzdzeta = dzdzeta - self.nfp*self.xn[im]*self.zbs[im]*cos_angle
      R = R + self.rbc[im]*cos_angle
    dxdtheta = dRdtheta*np.cos(zeta)
    dydtheta = dRdtheta*np.sin(zeta)
    dxdzeta = dRdzeta*np.cos(zeta) - R*np.sin(zeta)
    dydzeta = dRdzeta*np.sin(zeta) + R*np.cos(zeta)
    return dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta

  def jacobian(self,theta=None,zeta=None):
    [Nx,Ny,Nz] = self.N(theta,zeta)
    norm_normal = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    return norm_normal
  
  def normalized_jacobian(self,theta=None,zeta=None):
    norm_normal = self.jacobian(theta,zeta)
    area = self.area(theta,zeta)
    return norm_normal*4*np.pi*np.pi/(area)
  
  def normalized_jacobian_derivatives(self,xm_sensitivity,xn_sensitivity,theta=None,zeta=None):
    if (theta is None and zeta is None):
      zeta = self.zetas_2d
      theta = self.thetas_2d
    if (theta.ndim != zeta.ndim):
      print('Error! Incorrect dimensions for theta and zeta in normalized_jacobian_derivatives.')
      sys.exit(0)
    if (theta.ndim == 1):
      dim1 = len(theta)
      dim2 = 1
    elif (theta.ndim == 2):
      dim1 = len(theta[:,0])
      dim2 = len(theta[0,:])
    else:
      print('Error! Incorrect dimensions for theta and zeta in normalized_jacobian_derivatives.')
      sys.exit(0)
    [dareadrmnc, dareadzmns] = self.area_derivatives(xm_sensitivity,xn_sensitivity,theta,zeta)
    [dNdrmnc, dNdzmns] = self.jacobian_derivatives(xm_sensitivity,xn_sensitivity,theta,zeta)
    area = self.area(theta,zeta)
    N = self.jacobian(theta,zeta)
    
    mnmax_sensitivity = len(xm_sensitivity)
    dnormalized_jacobiandrmnc = np.zeros((mnmax_sensitivity,dim1,dim2))
    dnormalized_jacobiandzmns = np.zeros((mnmax_sensitivity,dim1,dim2))
    for imn in range(mnmax_sensitivity):
      dnormalized_jacobiandrmnc[imn,:,:] = dNdrmnc[imn]/area - N*dareadrmnc[imn]/(area*area)
      dnormalized_jacobiandzmns[imn,:,:] = dNdzmns[imn]/area - N*dareadzmns[imn]/(area*area)
    
    return 4*np.pi*np.pi*dnormalized_jacobiandrmnc, 4*np.pi*np.pi*dnormalized_jacobiandzmns
    
  # Returns X, Y, Z at isurf point on full mesh 
  def position(self,theta=None,zeta=None):
    if (theta is None and zeta is None):
      theta = self.thetas_2d
      zeta = self.zetas_2d
    elif (np.array(theta).shape != np.array(zeta).shape):
      print('Incorrect shape of theta and zeta in position_first_derivatives')
      sys.exit(0)
    R = np.zeros(np.shape(theta))
    Z = np.zeros(np.shape(zeta))
    for im in range(self.mnmax):
      angle = self.xm[im]*theta - self.nfp*self.xn[im]*zeta
      cos_angle = np.cos(angle)
      sin_angle = np.sin(angle)
      R = R + self.rbc[im]*cos_angle
      Z = Z + self.zbs[im]*sin_angle
    X = R*np.cos(zeta)
    Y = R*np.sin(zeta)
    return X, Y, Z, R
  
  def position_second_derivatives(self,theta=None,zeta=None):
    if (theta is None and zeta is None):
      theta = self.thetas_2d
      zeta = self.zetas_2d
    elif (np.array(theta).shape != np.array(zeta).shape):
      print('Incorrect shape of theta and zeta in position_first_derivatives')
      sys.exit(0)
   
    d2Rdtheta2 = np.zeros(np.shape(theta))
    d2Rdzeta2 = np.zeros(np.shape(theta))
    d2Zdtheta2 = np.zeros(np.shape(theta))
    d2Zdzeta2 = np.zeros(np.shape(theta))
    d2Rdthetadzeta = np.zeros(np.shape(theta))
    d2Zdthetadzeta = np.zeros(np.shape(theta))
    dRdtheta = np.zeros(np.shape(theta))
    dzdtheta = np.zeros(np.shape(theta))
    dRdzeta = np.zeros(np.shape(theta))
    dzdzeta = np.zeros(np.shape(theta))
    R = np.zeros(np.shape(theta))
    for im in range(self.mnmax):
      angle = self.xm[im]*theta - self.nfp*self.xn[im]*zeta
      cos_angle = np.cos(angle)
      sin_angle = np.sin(angle)
      R = R + this_rmnc[im]*cos_angle
      d2Rdtheta2 = d2Rdtheta2 - self.xm[im]*self.xm[im]*self.rbc[im]*cos_angle
      d2Zdtheta2 = d2Zdtheta2 - self.xm[im]*self.xm[im]*self.zbs[im]*sin_angle
      d2Rdzeta2 = d2Rdzeta2 - self.nfp*self.nfp*self.xn[im]*self.xn[im]*self.rbc[im]*cos_angle
      d2Zdzeta2 = d2Zdzeta2 - self.nfp*self.nfp*self.xn[im]*self.xn[im]*self.zbs[im]*sin_angle
      d2Rdthetadzeta = d2Rdthetadzeta + self.nfp*self.xm[im]*self.xn[im]*self.rbc[im]*cos_angle
      d2Zdthetadzeta = d2Zdthetadzeta + self.nfp*self.xm[im]*self.xn[im]*self.zbs[im]*sin_angle
      dRdtheta = dRdtheta - self.xm[im]*self.rbc[im]*sin_angle
      dzdtheta = dzdtheta + self.xm[im]*self.zbs[im]*cos_angle
      dRdzeta = dRdzeta + self.nfp*self.xn[im]*self.rbc[im]*sin_angle
      dzdzeta = dzdzeta - self.nfp*self.xn[im]*self.zbs[im]*cos_angle
    d2xdtheta2 = d2Rdtheta2*np.cos(zeta)
    d2ydtheta2 = d2Rdtheta2*np.sin(zeta)
    d2xdzeta2 = d2Rdzeta2*np.cos(zeta) \
      - 2*dRdzeta*np.sin(zeta) - R*np.cos(zeta)
    d2ydzeta2 = d2Rdzeta2*np.sin(zeta) \
      + 2*dRdzeta*np.cos(zeta) - R*np.sin(zeta)
    d2xdthetadzeta = d2Rdthetadzeta*np.cos(zeta) - dRdtheta*np.sin(zeta)
    d2ydthetadzeta = d2Rdthetadzeta*np.sin(zeta) + dRdtheta*np.cos(zeta)
    return d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, d2ydthetadzeta, \
      d2Zdtheta2, d2Zdzeta2, d2Zdthetadzeta
    
  def mean_curvature(self,theta=None,zeta=None):
    [dxdtheta, dxdzeta, dydtheta, dydzeta, dZdtheta, dZdzeta] = \
      self.position_first_derivatives(theta,zeta)
    [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, d2ydthetadzeta, \
      d2Zdtheta2, d2Zdzeta2, d2Zdthetadzeta] = self.position_second_derivatives(theta,zeta)
    
    norm_x = dydtheta*dZdzeta - dydzeta*dZdtheta
    norm_y = dZdtheta*dxdzeta - dZdzeta*dxdtheta
    norm_z = dxdtheta*dydzeta - dxdzeta*dydtheta
    norm_normal = np.sqrt(norm_x**2 + norm_y**2 + norm_z**2)
    nx = norm_x/norm_normal
    ny = norm_y/norm_normal
    nz = norm_z/norm_normal
    E = dxdtheta*dxdtheta + dydtheta*dydtheta + dZdtheta*dZdtheta
    F = dxdtheta*dxdzeta + dydtheta*dydzeta + dZdtheta*dZdzeta
    G = dxdzeta*dxdzeta + dydzeta*dydzeta + dZdzeta*dZdzeta
    e = nx*d2xdtheta2 + ny*d2ydtheta2 + nz*d2Zdtheta2
    f = nx*d2xdthetadzeta + ny*d2ydthetadzeta + nz*d2Zdthetadzeta
    g = nx*d2xdzeta2 + ny*d2ydzeta2 + nz*d2Zdzeta2
    H = (e*G-2*f*F+g*E)/(E*G-F*F)
    return H
  
  def print_namelist(self,input_filename=None,namelist=None):
    if (namelist is None):
      namelist = self.namelist
    if (input_filename is None):
      input_filename = self.input_filename
    f = open(input_filename,'w')
    f.write("&INDATA\n")
    for item in namelist.keys():
      if (not isinstance(namelist[item],(list, tuple, np.ndarray,str))):
        f.write(item+"="+str(namelist[item])+"\n")
      elif (isinstance(namelist[item],str)):
        f.write(item+"="+"'"+str(namelist[item])+"'"+"\n")
      elif (item not in ['rbc','zbs']):
        f.write(item+"=")
        f.writelines(map(lambda x:str(x)+" ",namelist[item]))
        f.write("\n")
    # Write RBC
    for imn in range(self.mnmax):
      if (self.rbc[imn]!=0):
        f.write('RBC('+str(int(self.xn[imn]))+","\
                 +str(int(self.xm[imn]))+") = "\
                 +str(self.rbc[imn])+'\n')
    # Write ZBS
    for imn in range(self.mnmax):
      if (self.zbs[imn]!=0):
        f.write('ZBS('+str(int(self.xn[imn]))+","\
                 +str(int(self.xm[imn]))+") = "\
                 +str(self.zbs[imn])+'\n')

    f.write("/\n")
    f.close()
    
  def axis_position(self):
    R0 = np.zeros(self.nzeta)
    Z0 = np.zeros(self.nzeta)
    for im in range(len(self.raxis)):
      angle = -self.nfp*im*self.zetas
      R0 = R0 + self.raxis[im]*np.cos(angle)
      Z0 = Z0 + self.zaxis[im]*np.sin(angle)
    return R0, Z0
  
  # Given (R,Z) defining boundary shape, determine if (R0,Z0) lies inside boundary
  def test_axis(self):
    [X, Y, Z, R] = self.position()
    [R0,Z0] = self.axis_position()
    
    inSurface = np.zeros(self.nzeta)
    for izeta in range(self.nzeta):
      inSurface[izeta] = point_in_polygon(R[izeta,:],Z[izeta,:],R0[izeta],Z0[izeta])
    return np.all(inSurface)
  
  def modify_axis(self):
#     if (boundary is None):
#       boundary = self.boundary
    [X, Y, Z, R] = self.position()
#     RBC = boundary[0:self.mnmax]
#     ZBS = boundary[self.mnmax::]
#     if (self.vmecOutputObject is None):
#       zetas = np.linspace(0,2*np.pi/self.nfp,self.nzeta+1)
#       zetas = np.delete(zetas,-1)
#     else:
#       zetas = self.vmecOutputObject.zetas
    zetas = self.zetas
      
    angle_1s = np.linspace(0,2*np.pi,20)
    angle_2s = np.linspace(0,2*np.pi,20)
    angle_1s = np.delete(angle_1s,-1)
    angle_2s = np.delete(angle_2s,-1)
    inSurface = False
    for angle_1 in angle_1s:
      for angle_2 in angle_2s:
        if (angle_1 != angle_2):
          R0_angle1 = np.zeros(np.shape(zetas))
          Z0_angle1 = np.zeros(np.shape(zetas))
          R0_angle2 = np.zeros(np.shape(zetas))
          Z0_angle2 = np.zeros(np.shape(zetas))
          for im in range(self.mnmax):
            angle = self.xm[im]*angle_1 - self.nfp*self.xn[im]*zetas
            R0_angle1 = R0_angle1 + self.rbc[im]*np.cos(angle)
            Z0_angle1 = Z0_angle1 + self.zbs[im]*np.sin(angle)
            angle = self.xm[im]*angle_2 - self.nfp*self.xn[im]*zetas
            R0_angle2 = R0_angle2 + self.rbc[im]*np.cos(angle)
            Z0_angle2 = Z0_angle2 + self.zbs[im]*np.sin(angle)
          Raxis = 0.5*(R0_angle1 + R0_angle2)
          Zaxis = 0.5*(Z0_angle1 + Z0_angle2)
          # Check new axis shape
          inSurfaces = np.zeros(self.nzeta)
          for izeta in range(self.nzeta):
            inSurfaces[izeta] = point_in_polygon(R[izeta,:],Z[izeta,:],Raxis[izeta],Zaxis[izeta])
#           if (not inSurfaces[izeta]):
#             plt.plot(R[izeta,:],Z[izeta,:])
#             plt.plot(Raxis[izeta],Zaxis[izeta],marker='*')
#             plt.title('angle_1 = '+str(angle_1)+', angle_2 ='+str(angle_2))
#             plt.show()
          inSurface = np.all(inSurfaces)
          if (inSurface):
            break
    if (not inSurface):
      print('Unable to find suitable axis shape in modify_axis.')
      sys.exit(0)
    
    # Now Fourier transform 
    raxis_cc = np.zeros(self.ntor)
    zaxis_cs = np.zeros(self.ntor)
    for n in range(self.ntor):
      angle = -n*self.nfp*zetas
      raxis_cc[n] = np.sum(Raxis*np.cos(angle))/np.sum(np.cos(angle)**2)
      if (n > 0):
        zaxis_cs[n] = np.sum(Zaxis*np.sin(angle))/np.sum(np.sin(angle)**2)
    self.raxis = raxis_cc
    self.zaxis = zaxis_cs
        
  def radius_derivatives(self,xm_sensitivity,xn_sensitivity,theta=None,zeta=None):
    if (theta is None and zeta is None):
      zeta = self.zetas_2d
      theta = self.thetas_2d
    if (theta.ndim != zeta.ndim):
      print('Error! Incorrect dimensions for theta and zeta in radius_derivatives.')
      sys.exit(0)
    if (theta.ndim == 1):
      dim1 = len(theta)
      dim2 = 1
    elif (theta.ndim == 2):
      dim1 = len(theta[:,0])
      dim2 = len(theta[0,:])
    else:
      print('Error! Incorrect dimensions for theta and zeta in radius_derivatives.')
      sys.exit(0)
      
    mnmax_sensitivity = len(xm_sensitivity)
    
    dRdrmnc = np.zeros((mnmax_sensitivity,dim1,dim2))
    dRdzmns = np.zeros((mnmax_sensitivity,dim1,dim2))
    for imn in range(mnmax_sensitivity):
      angle = xm_sensitivity[imn]*theta - self.nfp*xn_sensitivity[imn]*zeta
      cos_angle = np.cos(angle)
      dRdrmnc[imn,:,:] = cos_angle
    return dRdrmnc, dRdzmns
  
  def volume_derivatives(self,xm_sensitivity,xn_sensitivity,theta=None,zeta=None):
    if (theta is None and zeta is None):
      zeta = self.zetas_2d
      theta = self.thetas_2d
    if (theta.ndim != zeta.ndim):
      print('Error! Incorrect dimensions for theta and zeta in volume_derivatives.')
      sys.exit(0)
    if (theta.ndim == 1):
      dim1 = len(theta)
      dim2 = 1
    elif (theta.ndim == 2):
      dim1 = len(theta[:,0])
      dim2 = len(theta[0,:])
    else:
      print('Error! Incorrect dimensions for theta and zeta in volume_derivatives.')
      sys.exit(0)
    [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta] = \
      self.position_first_derivatives(theta,zeta)
    [X,Y,Z,R] = self.position(theta,zeta)
    [Nx, Ny, Nz] = self.N(theta,zeta)
    
    mnmax_sensitivity = len(xm_sensitivity)
    
    d2rdthetadrmnc = np.zeros((3,mnmax_sensitivity,dim1,dim2))
    d2rdzetadrmnc = np.zeros((3,mnmax_sensitivity,dim1,dim2))
    d2rdthetadzmns = np.zeros((3,mnmax_sensitivity,dim1,dim2))
    d2rdzetadzmns = np.zeros((3,mnmax_sensitivity,dim1,dim2))
    dzdzmns = np.zeros((mnmax_sensitivity,dim1,dim2))
    for imn in range(mnmax_sensitivity):
      angle = xm_sensitivity[imn]*theta - self.nfp*xn_sensitivity[imn]*zeta
      cos_angle = np.cos(angle)
      sin_angle = np.sin(angle)
      cos_zeta = np.cos(zeta)
      sin_zeta = np.sin(zeta)
      d2rdthetadrmnc[0,imn,:,:] = -xm_sensitivity[imn]*sin_angle*cos_zeta
      d2rdthetadrmnc[1,imn,:,:] = -xm_sensitivity[imn]*sin_angle*sin_zeta
      d2rdzetadrmnc[0,imn,:,:] = self.nfp*xn_sensitivity[imn]*sin_angle*cos_zeta - cos_angle*sin_zeta
      d2rdzetadrmnc[1,imn,:,:] = self.nfp*xn_sensitivity[imn]*sin_angle*sin_zeta + cos_angle*cos_zeta
      dzdzmns[imn,:,:] = dzdzmns[imn,:,:] = sin_angle
      
    dNzdrmnc = -d2rdzetadrmnc[0,:,:,:]*dydtheta - dxdzeta*d2rdthetadrmnc[1,:,:,:] \
      + d2rdthetadrmnc[0,:,:,:]*dydzeta + dxdtheta*d2rdzetadrmnc[1,:,:,:] 
    dNzdzmns = -d2rdzetadzmns[0,:,:,:]*dydtheta - dxdzeta*d2rdthetadzmns[1,:,:,:] \
      + d2rdthetadzmns[0,:,:,:]*dydzeta + dxdtheta*d2rdzetadzmns[1,:,:,:]
    dvolumedrmnc = -np.tensordot(dNzdrmnc,Z,axes=2)*self.dtheta*self.dzeta*self.nfp
    dvolumedzmns = -(np.tensordot(dNzdzmns,Z,axes=2) + np.tensordot(dzdzmns,Nz,axes=2))*self.dtheta*self.dzeta*self.nfp
    return dvolumedrmnc, dvolumedzmns
  
  def area_derivatives(self,xm_sensitivity,xn_sensitivity,theta=None,zeta=None):
    [dNdrmnc, dNdzmns] = self.jacobian_derivatives(xm_sensitivity,xn_sensitivity,theta,zeta)
    dareadrmnc = np.sum(dNdrmnc,axis=(1,2))*self.dtheta*self.dzeta*self.nfp
    dareadzmns = np.sum(dNdzmns,axis=(1,2))*self.dtheta*self.dzeta*self.nfp
    return dareadrmnc, dareadzmns
  
  # Check that theta and zeta are of correct size
  # Shape of theta and zeta must be the same
  def jacobian_derivatives(self,xm_sensitivity,xn_sensitivity,theta=None,zeta=None):
    if (theta is None and zeta is None):
      zeta = self.zetas_2d
      theta = self.thetas_2d
    if (theta.ndim != zeta.ndim):
      print('Error! Incorrect dimensions for theta and zeta in jacobian_derivatives.')
      sys.exit(0)
    if (theta.ndim == 1):
      dim1 = len(theta)
      dim2 = 1
    elif (theta.ndim == 2):
      dim1 = len(theta[:,0])
      dim2 = len(theta[0,:])
    else:
      print('Error! Incorrect dimensions for theta and zeta in jacobian_derivatives.')
      sys.exit(0)
    [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta] = \
      self.position_first_derivatives(theta,zeta)
    [Nx, Ny, Nz] = self.N(theta,zeta)
    N = np.sqrt(Nx*Nx + Ny*Ny + Nz*Nz)
    nx = Nx/N
    ny = Ny/N
    nz = Nz/N
    
    mnmax_sensitivity = len(xm_sensitivity)
    
    d2rdthetadrmnc = np.zeros((3,mnmax_sensitivity,dim1,dim2))
    d2rdzetadrmnc = np.zeros((3,mnmax_sensitivity,dim1,dim2))
    d2rdthetadzmns = np.zeros((3,mnmax_sensitivity,dim1,dim2))
    d2rdzetadzmns = np.zeros((3,mnmax_sensitivity,dim1,dim2))
    for imn in range(mnmax_sensitivity):
      angle = xm_sensitivity[imn]*theta - self.nfp*xn_sensitivity[imn]*zeta
      cos_angle = np.cos(angle)
      sin_angle = np.sin(angle)
      cos_zeta = np.cos(zeta)
      sin_zeta = np.sin(zeta)
      d2rdthetadrmnc[0,imn,:,:] = -xm_sensitivity[imn]*sin_angle*cos_zeta
      d2rdthetadrmnc[1,imn,:,:] = -xm_sensitivity[imn]*sin_angle*sin_zeta
      d2rdzetadrmnc[0,imn,:,:] = self.nfp*xn_sensitivity[imn]*sin_angle*cos_zeta - cos_angle*sin_zeta
      d2rdzetadrmnc[1,imn,:,:] = self.nfp*xn_sensitivity[imn]*sin_angle*sin_zeta + cos_angle*cos_zeta
      d2rdthetadzmns[2,imn,:,:] = xm_sensitivity[imn]*cos_angle
      d2rdzetadzmns[2,imn,:,:] = -self.nfp*xn_sensitivity[imn]*cos_angle
    dNdrmnc = (d2rdthetadrmnc[1,:,:,:]*dzdzeta - d2rdthetadrmnc[2,:,:,:]*dydzeta)*nx \
      + (d2rdthetadrmnc[2,:,:,:]*dxdzeta - d2rdthetadrmnc[0,:,:,:]*dzdzeta)*ny \
      + (d2rdthetadrmnc[0,:,:,:]*dydzeta - d2rdthetadrmnc[1,:,:,:]*dxdzeta)*nz \
      + (dydtheta*d2rdzetadrmnc[2,:,:,:] - dzdtheta*d2rdzetadrmnc[1,:,:,:])*nx \
      + (dzdtheta*d2rdzetadrmnc[0,:,:,:] - dxdtheta*d2rdzetadrmnc[2,:,:,:])*ny \
      + (dxdtheta*d2rdzetadrmnc[1,:,:,:] - dydtheta*d2rdzetadrmnc[0,:,:,:])*nz
    dNdzmns = (d2rdthetadzmns[1,:,:,:]*dzdzeta - d2rdthetadzmns[2,:,:,:]*dydzeta)*nx \
      + (d2rdthetadzmns[2,:,:,:]*dxdzeta - d2rdthetadzmns[0,:,:,:]*dzdzeta)*ny \
      + (d2rdthetadzmns[0,:,:,:]*dydzeta - d2rdthetadzmns[1,:,:,:]*dxdzeta)*nz \
      + (dydtheta*d2rdzetadzmns[2,:,:,:] - dzdtheta*d2rdzetadzmns[1,:,:,:])*nx \
      + (dzdtheta*d2rdzetadzmns[0,:,:,:] - dxdtheta*d2rdzetadzmns[2,:,:,:])*ny \
      + (dxdtheta*d2rdzetadzmns[1,:,:,:] - dydtheta*d2rdzetadzmns[0,:,:,:])*nz
    return dNdrmnc, dNdzmns

  # Determine if (R0,Z0) lies in boundary defined by (R,Z) in toroidal plane
  #TODO : test for correct sizes
def point_in_polygon(R,Z,R0,Z0):
  ntheta = len(R)
  oddNodes = False
  j = ntheta-1
  for i in range(ntheta):
      if ((Z[i]<Z0 and Z[j]>=Z0) or (Z[j]<Z0 and Z[i]>=Z0)):
          if (R[i]+(Z0-Z[i])/(Z[j]-Z[i])*(R[j]-R[i])<R0):
              oddNodes = not oddNodes
      j = i
  return oddNodes



