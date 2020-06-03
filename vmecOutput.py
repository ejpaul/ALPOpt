import numpy as np
from scipy.io import netcdf
import os
from scipy import interpolate

class readVmecOutput:
  
  def __init__(self,wout_filename,ntheta=100,nzeta=100):
    self.wout_filename = wout_filename
    self.input_filename = 'input.'+wout_filename.split('.')[0][5::]
    self.directory = os.getcwd()
    
    self.ntheta = ntheta
    self.nzeta = nzeta
    thetas = np.linspace(0,2*np.pi,ntheta+1)
    zetas = np.linspace(0,2*np.pi,nzeta+1)
    thetas = np.delete(thetas,-1)
    zetas = np.delete(zetas,-1)
    [thetas_2d,zetas_2d] = np.meshgrid(thetas,zetas)
    self.thetas = thetas
    self.zetas = zetas
    self.dtheta = thetas[1]-thetas[0]
    self.dzeta = zetas[1]-zetas[0]
    self.thetas_2d = thetas_2d
    self.zetas_2d = zetas_2d
    self.mu0 = 4*np.pi*1.0e-7
    
    f = netcdf.netcdf_file(self.wout_filename,'r',mmap=False)
    self.rmnc = f.variables["rmnc"][()]
    self.zmns = f.variables["zmns"][()]
    self.bsubumnc = f.variables["bsubumnc"][()]
    self.bsubvmnc = f.variables["bsubvmnc"][()]
    self.bsupumnc = f.variables["bsupumnc"][()]
    self.bsupvmnc = f.variables["bsupvmnc"][()]
    self.xm = f.variables["xm"][()]
    self.xn = f.variables["xn"][()]
    self.mnmax = f.variables["mnmax"][()]
    self.gmnc = f.variables["gmnc"][()]
    self.psi = f.variables["phi"][()]/(2*np.pi)
    self.mpol = f.variables["mpol"][()]
    self.ntor = f.variables["ntor"][()]
    self.iota = f.variables["iotas"][()]
    self.vp = f.variables["vp"][()]
    self.pres = f.variables["pres"][()]
    self.volume = f.variables["volume_p"][()]
    
    # Remove axis point from half grid quantities
    self.bsubumnc = np.delete(self.bsubumnc,0,0)
    self.bsubvmnc = np.delete(self.bsubvmnc,0,0)
    self.bsupumnc = np.delete(self.bsupumnc,0,0)
    self.bsupvmnc = np.delete(self.bsupvmnc,0,0)
    self.gmnc = np.delete(self.gmnc,0,0)
    self.iota = np.delete(self.iota,0)
    self.vp = np.delete(self.vp,0)
    self.pres = np.delete(self.pres,0)
    
    self.s_full = self.psi/self.psi[-1]
    self.ds = self.s_full[1]-self.s_full[0]
    self.s_half = self.s_full - 0.5*self.ds
    self.s_half = np.delete(self.s_half,0)
    self.ns = len(self.s_full)
    self.ns_half = len(self.s_half)
    
    self.nfp = f.variables["nfp"][()]
    self.sign_jac = f.variables["signgs"][()]
  
#   def d_r_d_psi(self,isurf):
#     drmncdpsi = np.zeros(np.shape(self.xm))
#     dzmnsdpsi = np.zeros(np.shape(self.xm))
#     for im in range(self.mnmax):
#       drmncdpsi_spline = interpolate.InterpolatedUnivariateSpline(self.s_full,self.rmnc[:,im])
#       drmncdpsi[im] = drmncdpsi_spline(self.s_full[isurf])
#       dzmnsdpsi_spline = interpolate.InterpolatedUnivariateSpline(self.s_full,self.zmns[:,im])
#       dzmnsdpsi[im] = dzmnsdpsi_spline(self.s_full[isurf])
    
#     dRdpsi = np.zeros(np.shape(self.thetas_2d))
#     dZdpsi = np.zeros(np.shape(self.thetas_2d))
#     for im in range(self.mnmax):
#       angle = self.xm[im]*self.thetas_2d - self.xn[im]*self.zetas_2d
#       cos_angle = np.cos(angle)
#       sin_angle = np.sin(angle)
#       dRdpsi = dRdpsi + drmncdpsi[im]*np.cos(angle)
#       dZdpsi = dZdpsi + dzmnsdpsi[im]*np.sin(angle)
#     dXdpsi = dRdpsi*np.cos(self.zetas_2d)
#     dYdpsi = dRdpsi*np.sin(self.zetas_2d)
#     return dXdpsi, dYdpsi, dZdpsi
  
  def B_on_arclength_grid(self):
    # Extrapolate to last full mesh grid point
    bsupumnc_end = 1.5*self.bsupumnc[-1,:] - 0.5*self.bsupumnc[-2,:]
    bsupvmnc_end = 1.5*self.bsupvmnc[-1,:] - 0.5*self.bsupvmnc[-2,:]
    rmnc_end = self.rmnc[-1,:]
    zmns_end = self.zmns[-1,:]
      
    # Evaluate at last full mesh grid point
    [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, dRdzeta] = \
      self.position_first_derivatives(isurf)
    [x, y, z, R] = self.position(isurf)
    
    Bsupu_end = np.zeros(np.shape(self.thetas_2d))
    Bsupv_end = np.zeros(np.shape(self.thetas_2d))
    for im in range(self.mnmax):
      angle = self.xm[im]*self.thetas_2d - self.xn[im]*self.zetas_2d
      cos_angle = np.cos(angle)
      sin_angle = np.sin(angle)
      Bsupu_end = Bsupu_end + bsupumnc_end[im]*cos_angle
      Bsupv_end = Bsupv_end + bsupvmnc_end[im]*cos_angle
    Bx_end = Bsupu_end*dxdtheta + Bsupv_end*dxdzeta
    By_end = Bsupu_end*dydtheta + Bsupv_end*dydzeta
    Bz_end = Bsupu_end*dzdtheta + Bsupv_end*dzdzeta

    theta_arclength = np.zeros(np.shape(self.zetas_2d))
    for izeta in range(self.nzeta):
      for itheta in range(1,self.ntheta):
        dr = np.sqrt((R[izeta,itheta]-R[izeta,itheta-1])**2 \
                     + (z[izeta,itheta]-z[izeta,itheta-1])**2)
        theta_arclength[izeta,itheta] = theta_arclength[izeta,itheta-1] + dr
    
    return Bx_end, By_end, Bz_end, theta_arclength

  def compute_current(self):
    It_half = self.sign_jac*2*np.pi*self.bsubumnc[:,0]/self.mu0
    return It_half

  def compute_N(self,isurf):  
    [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, \
      dRdtheta, dRdzeta] = \
      self.position_first_derivatives(isurf)
  
    Nx = dydzeta*dzdtheta - dydtheta*dzdzeta
    Ny = dzdzeta*dxdtheta - dzdtheta*dxdzeta
    Nz = dxdzeta*dydtheta - dxdtheta*dydzeta
    return Nx, Ny, Nz
  
  def compute_n(self,isurf):
    [Nx, Ny, Nz] = self.compute_N(isurf)
    norm_normal = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    nx = Nx/norm_normal
    ny = Ny/norm_normal
    nz = Nz/norm_normal
    return nx, ny, nz
  
  def compute_N_derivatives(self,isurf):
    [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, dRdzeta] = \
      self.position_first_derivatives(isurf)
    [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, d2ydthetadzeta, \
      d2zdtheta2, d2zdzeta2, d2zdthetadzeta, d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta] = \
      self.position_second_derivatives(isurf)

    dNxdtheta = d2ydthetadzeta*dzdtheta + dydzeta*d2zdtheta2 \
      - d2ydtheta2*dzdzeta - dydtheta*d2zdthetadzeta
    dNxdzeta = d2ydzeta2*dzdtheta + dydzeta*d2zdthetadzeta \
      - d2ydthetadzeta*dzdzeta - dydtheta*d2zdzeta2
    dNydtheta = d2zdthetadzeta*dxdtheta + dzdzeta*d2xdtheta2 \
      - d2zdtheta2*dxdzeta - dzdtheta*d2xdthetadzeta 
    dNydzeta = d2zdzeta2*dxdtheta + dzdzeta*d2xdthetadzeta \
      - d2zdthetadzeta*dxdzeta - dzdtheta*d2xdzeta2 
    dNzdtheta = d2xdthetadzeta*dydtheta + dxdzeta*d2ydtheta2 \
      - d2xdtheta2*dydzeta - dxdtheta*d2ydthetadzeta
    dNzdzeta = d2xdzeta2*dydtheta + dxdzeta*d2ydthetadzeta \
      - d2xdthetadzeta*dydzeta - dxdtheta*d2ydzeta2
    return dNxdtheta, dNxdzeta, dNydtheta, dNydzeta, dNzdtheta, dNzdzeta
  
  def compute_n_derivatives(self,isurf):
    [Nx,Ny,Nz] = self.compute_N(isurf)
    [dNxdtheta, dNxdzeta, dNydtheta, dNydzeta, dNzdtheta, dNzdzeta] = \
      self.compute_N_derivatives(isurf)
    norm_normal = np.sqrt(Nx*Nx + Ny*Ny + Nz*Nz)
    dnorm_normaldtheta = (Nx*dNxdtheta + Ny*dNydtheta + Nz*dNzdtheta)/norm_normal
    dnorm_normaldzeta = (Nx*dNxdzeta + Ny*dNydzeta + Nz*dNzdzeta)/norm_normal
    dnxdtheta = dNxdtheta/norm_normal - Nx*dnorm_normaldtheta/(norm_normal**2)
    dnxdzeta = dNxdzeta/norm_normal - Nx*dnorm_normaldzeta/(norm_normal**2)
    dnydtheta = dNydtheta/norm_normal - Ny*dnorm_normaldtheta/(norm_normal**2)
    dnydzeta = dNydzeta/norm_normal - Ny*dnorm_normaldzeta/(norm_normal**2)
    dnzdtheta = dNzdtheta/norm_normal - Nz*dnorm_normaldtheta/(norm_normal**2)
    dnzdzeta = dNzdzeta/norm_normal - Nz*dnorm_normaldzeta/(norm_normal**2)
    return dnxdtheta, dnxdzeta, dnydtheta, dnydzeta, dnzdtheta, dnzdzeta
  
  # Returns X, Y, Z at isurf point on full mesh 
  def position(self,isurf):
    this_rmnc = self.rmnc[isurf,:]
    this_zmns = self.zmns[isurf,:] 

    R = np.zeros(np.shape(self.thetas_2d))
    z = np.zeros(np.shape(self.thetas_2d))
    for im in range(self.mnmax):
      angle = self.xm[im]*self.thetas_2d - self.xn[im]*self.zetas_2d
      cos_angle = np.cos(angle)
      sin_angle = np.sin(angle)
      R = R + this_rmnc[im]*cos_angle
      z = z + this_zmns[im]*sin_angle
    x = R*np.cos(self.zetas_2d)
    y = R*np.sin(self.zetas_2d)
    return x, y, z, R

  # Integrated rotational transform with weight function
  def evaluate_iota_objective(self,weight):
    return np.sum(weight(self.s_half)*self.iota)*self.ds*self.psi[-1]*self.sign_jac
  
  # Integrated differential volume with weight function
  def evaluate_well_objective(self,weight):
    return 4*np.pi*np.pi*np.sum(weight(self.s_half)*self.vp)*self.ds
  
  def area(self,isurf):
    [Nx,Ny,Nz] = self.compute_N(isurf)
    norm_normal = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    area = np.sum(norm_normal)*self.dtheta*self.dzeta
    return area
  
  def position_first_derivatives(self,isurf):
    [x, y, z, R] = self.position(isurf)
    this_rmnc = self.rmnc[isurf,:]
    this_zmns = self.zmns[isurf,:] 
    dRdtheta = np.zeros(np.shape(self.thetas_2d))
    dzdtheta = np.zeros(np.shape(self.thetas_2d))
    dRdzeta = np.zeros(np.shape(self.thetas_2d))
    dzdzeta = np.zeros(np.shape(self.thetas_2d))
    for im in range(self.mnmax):
      angle = self.xm[im]*self.thetas_2d - self.xn[im]*self.zetas_2d
      cos_angle = np.cos(angle)
      sin_angle = np.sin(angle)
      dRdtheta = dRdtheta - self.xm[im]*this_rmnc[im]*sin_angle
      dzdtheta = dzdtheta + self.xm[im]*this_zmns[im]*cos_angle
      dRdzeta = dRdzeta + self.xn[im]*this_rmnc[im]*sin_angle
      dzdzeta = dzdzeta - self.xn[im]*this_zmns[im]*cos_angle
    dxdtheta = dRdtheta*np.cos(self.zetas_2d)
    dydtheta = dRdtheta*np.sin(self.zetas_2d)
    dxdzeta = dRdzeta*np.cos(self.zetas_2d) - R*np.sin(self.zetas_2d)
    dydzeta = dRdzeta*np.sin(self.zetas_2d) + R*np.cos(self.zetas_2d)
    return dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, dRdzeta
  
  def position_second_derivatives(self,isurf):
    [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, dRdzeta] = \
      self.position_first_derivatives(isurf)
    [x, y, z, R] = self.position(isurf)
    
    this_rmnc = self.rmnc[isurf,:]
    this_zmns = self.zmns[isurf,:] 
    
    d2Rdtheta2 = np.zeros(np.shape(self.thetas_2d))
    d2Rdzeta2 = np.zeros(np.shape(self.thetas_2d))
    d2Zdtheta2 = np.zeros(np.shape(self.thetas_2d))
    d2Zdzeta2 = np.zeros(np.shape(self.thetas_2d))
    d2Rdthetadzeta = np.zeros(np.shape(self.thetas_2d))
    d2Zdthetadzeta = np.zeros(np.shape(self.thetas_2d))
    for im in range(self.mnmax):
      angle = self.xm[im]*self.thetas_2d - self.xn[im]*self.zetas_2d
      cos_angle = np.cos(angle)
      sin_angle = np.sin(angle)
      d2Rdtheta2 = d2Rdtheta2 - self.xm[im]*self.xm[im]*this_rmnc[im]*cos_angle
      d2Zdtheta2 = d2Zdtheta2 - self.xm[im]*self.xm[im]*this_zmns[im]*sin_angle
      d2Rdzeta2 = d2Rdzeta2 - self.xn[im]*self.xn[im]*this_rmnc[im]*cos_angle
      d2Zdzeta2 = d2Zdzeta2 - self.xn[im]*self.xn[im]*this_zmns[im]*sin_angle
      d2Rdthetadzeta = d2Rdthetadzeta + self.xm[im]*self.xn[im]*this_rmnc[im]*cos_angle
      d2Zdthetadzeta = d2Zdthetadzeta + self.xm[im]*self.xn[im]*this_zmns[im]*sin_angle
    d2xdtheta2 = d2Rdtheta2*np.cos(self.zetas_2d)
    d2ydtheta2 = d2Rdtheta2*np.sin(self.zetas_2d)
    d2xdzeta2 = d2Rdzeta2*np.cos(self.zetas_2d) \
      - 2*dRdzeta*np.sin(self.zetas_2d) - R*np.cos(self.zetas_2d)
    d2ydzeta2 = d2Rdzeta2*np.sin(self.zetas_2d) \
      + 2*dRdzeta*np.cos(self.zetas_2d) - R*np.sin(self.zetas_2d)
    d2xdthetadzeta = d2Rdthetadzeta*np.cos(self.zetas_2d) - dRdtheta*np.sin(self.zetas_2d)
    d2ydthetadzeta = d2Rdthetadzeta*np.sin(self.zetas_2d) + dRdtheta*np.cos(self.zetas_2d)
    return d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, d2ydthetadzeta, \
      d2Zdtheta2, d2Zdzeta2, d2Zdthetadzeta, d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta
  
  def position_third_derivatives(self,isurf):
    [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, dRdzeta] = \
      self.position_first_derivatives(isurf)
    [x, y, z, R] = self.position(isurf)
    [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, d2ydthetadzeta, \
      d2zdtheta2, d2zdzeta2, d2zdthetadzeta, d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta] = \
      self.position_second_derivatives(isurf)

    this_rmnc = self.rmnc[isurf,:]
    this_zmns = self.zmns[isurf,:] 

    d3Rdtheta3 = np.zeros(np.shape(self.thetas_2d))
    d3Rdzeta3 = np.zeros(np.shape(self.thetas_2d))
    d3Rdzetadtheta2 = np.zeros(np.shape(self.thetas_2d))
    d3Rdthetadzeta2 = np.zeros(np.shape(self.thetas_2d))
    d3zdtheta3 = np.zeros(np.shape(self.thetas_2d))
    d3zdzeta3 = np.zeros(np.shape(self.thetas_2d))
    d3zdzetadtheta2 = np.zeros(np.shape(self.thetas_2d))
    d3zdthetadzeta2 = np.zeros(np.shape(self.thetas_2d))   
    for im in range(self.mnmax):
      angle = self.xm[im]*self.thetas_2d - self.xn[im]*self.zetas_2d
      cos_angle = np.cos(angle)
      sin_angle = np.sin(angle)
      d3Rdtheta3 = d3Rdtheta3 - self.xm[im]**3 * sin_angle
      d3Rdzeta3 = d3Rdzeta3 + self.xn[im]**3 * sin_angle
      d3Rdzetadtheta2 = d3Rdzetadtheta2 + self.xn[im]*self.xm[im]**2 * sin_angle
      d3Rdthetadzeta2 = d3Rdthetadzeta2 - self.xm[im]*self.xn[im]**2 * sin_angle
      d3zdtheta3 = d3zdtheta3 - self.xm[im]**3 * cos_angle
      d3zdzeta3 = d3zdzeta3 + self.xn[im]**3 * cos_angle
      d3zdzetadtheta2 = d3zdzetadtheta2 + self.xm[im]**2 * self.xn[im] * cos_angle
      d3zdthetadzeta2 = d3zdthetadzeta2 - self.xn[im]**2 * self.xm[im] * cos_angle
    cos_zeta = np.cos(self.zetas_2d)
    sin_zeta = np.sin(self.zetas_2d)
    d3xdtheta3 = d3Rdtheta3*cos_zeta
    d3ydtheta3 = d3Rdtheta3*sin_zeta
    d3xdzeta3 = d3Rdzeta3*cos_zeta - 3*d2Rdzeta2*sin_zeta \
      - 3*dRdzeta*cos_zeta + R*sin_zeta
    d3ydzeta3 = -R*cos_zeta - 3*dRdzeta*sin_zeta \
      + 3*cos_zeta*d2Rdzeta2 + d3Rdzeta3*sin_zeta
    d3xdzetadtheta2 = -sin_zeta*d2Rdtheta2 \
      + cos_zeta*d3Rdzetadtheta2
    d3ydzetadtheta2 = cos_zeta*d2Rdtheta2 \
      + sin_zeta*d3Rdzetadtheta2
    d3xdthetadzeta2 = -cos_zeta*dRdtheta - 2*sin_zeta*d2Rdthetadzeta \
      + cos_zeta*d3Rdthetadzeta2
    d3ydthetadzeta2 = -sin_zeta*dRdtheta + 2*cos_zeta*d2Rdthetadzeta \
      + sin_zeta*d3Rdthetadzeta2

    return d3xdtheta3, d3xdzeta3, d3xdthetadzeta2, d3xdzetadtheta2, \
      d3ydtheta3, d3ydzeta3, d3ydthetadzeta2, d3ydzetadtheta2, \
      d3zdtheta3, d3zdzeta3, d3zdthetadzeta2, d3zdzetadtheta2, \
      d3Rdtheta3, d3Rdzeta3, d3Rdthetadzeta2, d3Rdzetadtheta2
  
  def position_fourth_derivatives(self,isurf):
    [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, dRdzeta] = \
      self.position_first_derivatives(isurf)
    [x, y, z, R] = self.position(isurf)
    [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, d2ydthetadzeta, \
    d2Zdtheta2, d2Zdzeta2, d2Zdthetadzeta, d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta] = \
      self.position_second_derivatives(isurf)

    this_rmnc = self.rmnc[isurf,:]
    this_zmns = self.zmns[isurf,:] 
    
    d4Rdtheta4 = np.zeros(np.shape(self.thetas_2d))
    d4Rdzeta4 = np.zeros(np.shape(self.thetas_2d))
    d4Rdzetadtheta3 = np.zeros(np.shape(self.thetas_2d))
    d4Rdthetadzeta3 = np.zeros(np.shape(self.thetas_2d))
    d4Rdtheta2dzeta2 = np.zeros(np.shape(self.thetas_2d))
    d4zdtheta4 = np.zeros(np.shape(self.thetas_2d))
    d4zdzeta4 = np.zeros(np.shape(self.thetas_2d))
    d4zdzetadtheta3 = np.zeros(np.shape(self.thetas_2d))
    d4zdthetadzeta3 = np.zeros(np.shape(self.thetas_2d))
    d4zdtheta2dzeta2 = np.zeros(np.shape(self.thetas_2d))
    for im in range(self.mnmax):
      angle = self.xm[im]*self.thetas_2d - self.xn[im]*self.zetas_2d
      cos_angle = np.cos(angle)
      sin_angle = np.sin(angle)
      d4Rdtheta4 = d4Rdtheta4 + self.xm[im]**4 * cos_angle
      d4Rdzeta4 = d4Rdzeta4 + self.xn[im]**4 * sin_angle
      d4Rdzetadtheta3 = d4Rdzetadtheta3 - self.xm[im]**3 * self.xn[im] * cos_angle
      d4Rdthetadzeta3 = d4Rdthetadzeta3 - self.xm[im] * self.xn[im]**3 * cos_angle
      d4Rdtheta2dzeta2 = d4Rdtheta2dzeta2 + self.xm[im]**2 * self.xn[im]**2 * cos_angle
      d4zdtheta4 = d4Zdtheta4 - self.xm[im]**4 * sin_angle
      d4zdzeta4 = d4Zdzeta4 - self.xn[im]**4 * sin_angle
      d4zdzetadtheta3 = d4Zdzetadtheta3 + self.xm[im]**3 * self.xn[im] * sin_angle
      d4zdthetadzeta3 = d4Zdthetadzeta3 + self.xm[im] * self.xn[im]**3 * sin_angle
      d4zdtheta2dzeta2 = d4Zdtheta2dzeta2 - self.xm[im] **2 * self.xn[im]**2 * sin_angle
    cos_zeta = np.cos(self.zetas_2d)
    sin_zeta = np.sin(self.zetas_2d)
    d4xdtheta4 = d4Rdtheta4*cos_zeta
    d4xdzeta4 = R*cos_zeta + 4*sin_zeta*dRdzeta - 6*cos_zeta*d2Rdzeta2 - 4*sin_zeta*d3Rdzeta3 \
      + cos_zeta*d4Rdzeta4
    d4xdzetadtheta3 = -sin_zeta*d3Rdtheta3 + cos_zeta*d4Rdzetadtheta3
    d4xdthetadzeta3 = sin_zeta*dRdtheta - 3*cos_zeta*d2Rdthetadzeta \
      - 3*sin_zeta*d3Rdthetadzeta2 + cos_zeta*d4Rdthetadzeta3
    d4xdtheta2dzeta2 = -cos_zeta*d2Rdtheta2 - 2*sin_zeta*d3Rdzetadtheta2 \
      + cos_zeta*d4Rdtheta2dzeta2
    d4ydtheta4 = d4Rdtheta4*sin_zeta
    d4ydzeta4 = R*sin_zeta - 4*cos_zeta*dRdzeta - 6*sin_zeta*d2Rdzeta2 \
      + 4*cos_zeta*d3Rdzeta3 + sin_zeta*d4Rdzeta4
    d4ydzetadtheta3 = d3Rdtheta3*cos_zeta + d4Rdzetadtheta3*sin_zeta
    d4ydthetadzeta3 = -cos_zeta*dRdtheta - 3*sin_zeta*d2Rdthetadzeta \
      + 3*cos_zeta*d3Rdthetadzeta2 + sin_zeta*d4Rdthetadzeta3 
    d4ydtheta2dzeta2 = -sin_zeta*d2Rdtheta2 + 2*cos_zeta*d3Rdzetadtheta2 \
      + sin_zeta*d4Rdtheta2dzeta2
    return d4xdtheta4, d4xdzeta4, d4xdzetadtheta3, d4xdthetadzeta3, d4xdtheta2dzeta2, \
      d4ydtheta4, d4ydzeta4, d4ydzetadtheta3, d4ydthetadzeta3, d4ydtheta2dzeta2, \
      d4zdtheta4, d4zdzeta4, d4zdzetadtheta3, d4zdthetadzeta3, d4zdtheta2dzeta2, \
      d4Rdtheta4, d4Rdzeta4, d4Rdzetadtheta3, d4Rdthetadzeta3, d4Rdtheta2dzeta2 
  
  def metric_elements(self,isurf):
    [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, dRdzeta] = \
      self.position_first_derivatives(isurf)
    gthetatheta = dxdtheta*dxdtheta + dydtheta*dydtheta + dzdtheta*dzdtheta
    gthetazeta = dxdtheta*dxdzeta + dydtheta*dydzeta + dzdtheta*dzdzeta
    gzetazeta = dxdzeta*dxdzeta + dydzeta*dydzeta + dzdzeta*dzdzeta
    return gthetatheta, gthetazeta, gzetazeta
  
  def metric_elements_derivatives(self,isurf):
    [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, dRdzeta] = \
      self.position_first_derivatives(isurf)
    [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, d2ydthetadzeta, \
      d2zdtheta2, d2zdzeta2, d2zdthetadzeta, d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta] = \
      self.position_second_derivatives(isurf)
    [gthetatheta, gthetazeta, gzetazeta] = self.metric_elements(isurf)
    dgthetathetadtheta = 2*(d2xdtheta2*dxdtheta + d2ydtheta2*dydtheta + d2zdtheta2*dzdtheta)
    dgthetathetadzeta = 2*(d2xdthetadzeta*dxdtheta + d2ydthetadzeta*dydtheta
      + d2zdthetadzeta*dzdtheta)
    dgthetazetadtheta = d2xdtheta2*dxdzeta + dxdtheta*d2xdthetadzeta \
      + d2ydtheta2*dydzeta + dydtheta*d2ydthetadzeta \
      + d2zdtheta2*dzdzeta + dzdtheta*d2zdthetadzeta
    dgthetazetadzeta = d2xdthetadzeta*dxdzeta + dxdtheta*d2xdzeta2 \
      + d2ydthetadzeta*dydzeta + dydtheta*d2ydzeta2 \
      + d2zdthetadzeta*dzdzeta + dzdtheta*d2zdzeta2 
    dgzetazetadzeta = 2*(d2xdzeta2*dxdzeta + d2ydzeta2*dydzeta + d2zdzeta2*dzdzeta)
    dgzetazetadtheta = 2*(d2xdthetadzeta*dxdzeta + d2ydthetadzeta*dydzeta 
      + d2zdthetadzeta*dzdzeta)
    return dgthetathetadtheta, dgthetathetadzeta, dgthetazetadtheta, dgthetazetadzeta, \
      dgzetazetadzeta, dgzetazetadtheta
  
  def metric_elements_second_derivatives(self,isurf):
    [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, dRdtheta, dRdzeta] = \
      self.position_first_derivatives(isurf)
    [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, d2ydthetadzeta, \
      d2zdtheta2, d2zdzeta2, d2zdthetadzeta, d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta] = \
      self.position_second_derivatives(isurf)
    [gthetatheta, gthetazeta, gzetazeta] = self.metric_elements(isurf)
    [dgthetathetadtheta, dgthetathetadzeta, dgthetazetadtheta, dgthetazetadzeta, \
      dgzetazetadzeta, dgzetazetadtheta] = self.metric_elements_derivatives(isurf)
    
    d2gthetathetadtheta2 = 2*(d3xdtheta3*dxdtheta + d2xdtheta2**2 + d3ydtheta3*dydtheta \
      + d2ydtheta2**2 + d3zdtheta3*dzdtheta + d2zdtheta2**2)
    d2gthetathetadthetadzeta = 2*(d3xdzetadtheta2*dxdtheta + d2xdtheta2*d2xdthetadzeta \
      + d3ydzetadtheta2*dydtheta + d2ydtheta2*d2ydthetadzeta + d3zdzetadtheta2*dzdtheta \
      + d2zdtheta2**2)
    d2gthetazetadtheta2 = d3xdtheta3*dxdzeta + d2xdtheta2*d2xdthetadzeta \
      + d2xdtheta2*d2xdthetadzeta + dxdtheta*d3xdzetadtheta2 \
      + d3ydtheta2*dydzeta + d2ydtheta2*d2ydthetadzeta \
      + d2ydtheta2*d2ydthetadzeta + dydtheta*d3ydzetadtheta2 \
      + d3zdtheta2*dzdzeta + d2zdtheta2*d2zdthetadzeta \
      + d2zdtheta2
#     dgthetazetadtheta = d2xdtheta2*dxdzeta + dxdtheta*d2xdthetadzeta \
#       + d2ydtheta2*dydzeta + dydtheta*d2ydthetadzeta \
#       + d2zdtheta2*dzdzeta + dzdtheta*d2zdthetadzeta
#     dgthetazetadzeta = d2xdthetadzeta*dxdzeta + dxdtheta*d2xdzeta2 \
#       + d2ydthetadzeta*dydzeta + dydtheta*d2ydzeta2 \
#       + d2zdthetadzeta*dzdzeta + dzdtheta*d2zdzeta2 
#     dgzetazetadzeta = 2*(d2xdzeta2*dxdzeta + d2ydzeta2*dydzeta + d2zdzeta2*dzdzeta)
#     dgzetazetadtheta = 2*(d2xdthetadzeta*dxdzeta + d2ydthetadzeta*dydzeta 
#       + d2zdthetadzeta*dzdzeta)
    return dgthetathetadtheta, dgthetathetadzeta, dgthetazetadtheta, dgthetazetadzeta, \
      dgzetazetadzeta, dgzetazetadtheta    
  
  def mean_curvature(self,isurf):
    [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, d2ydthetadzeta, \
      d2zdtheta2, d2zdzeta2, d2zdthetadzeta, d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta] = \
      self.position_second_derivatives(isurf)
    [gthetatheta, gthetazeta, gzetazeta] = self.metric_elements(isurf)
    [nx, ny, nz] = self.compute_n(isurf)   

    E = gthetatheta
    F = gthetazeta
    G = gzetazeta
    e = nx*d2xdtheta2 + ny*d2ydtheta2 + nz*d2zdtheta2
    f = nx*d2xdthetadzeta + ny*d2ydthetadzeta + nz*d2zdthetadzeta
    g = nx*d2xdzeta2 + ny*d2ydzeta2 + nz*d2zdzeta2
    H = (e*G-2*f*F+g*E)/(E*G-F*F)
    return H
  
  # Second fundamental form
  def fun_form_2(self,isurf):
    [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, d2ydthetadzeta, \
      d2zdtheta2, d2zdzeta2, d2zdthetadzeta, d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta] = \
      self.position_second_derivatives(isurf)
    [nx,ny,nz] = self.compute_n(isurf)
    e = nx*d2xdtheta2 + ny*d2ydtheta2 + nz*d2zdtheta2
    f = nx*d2xdthetadzeta + ny*d2ydthetadzeta + nz*d2zdthetadzeta
    g = nx*d2xdzeta2 + ny*d2ydzeta2 + nz*d2zdzeta2

    return e, f, g
    
  def fun_form_2_derivatives(self,isurf):
    [nx,ny,nz] = self.compute_n(isurf)
    [dnxdtheta, dnxdzeta, dnydtheta, dnydzeta, dnzdtheta, dnzdzeta] = \
      self.compute_n_derivatives(isurf)
    [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, d2ydthetadzeta, \
      d2zdtheta2, d2zdzeta2, d2zdthetadzeta, d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta] = \
      self.position_second_derivatives(isurf)
    [d3xdtheta3, d3xdzeta3, d3xdthetadzeta2, d3xdzetadtheta2, \
      d3ydtheta3, d3ydzeta3, d3ydthetadzeta2, d3ydzetadtheta2, \
      d3zdtheta3, d3zdzeta3, d3zdthetadzeta2, d3zdzetadtheta2, \
      d3Rdtheta3, d3Rdzeta3, d3Rdthetadzeta2, d3Rdzetadtheta2] = \
      self.position_third_derivatives(isurf)
    dedtheta = dnxdtheta*d2xdtheta2 + nx*d3xdtheta3 \
             + dnydtheta*d2ydtheta2 + ny*d3ydtheta3 \
             + dnzdtheta*d2zdtheta2 + nz*d3zdtheta3
    dfdtheta = dnxdtheta*d2xdthetadzeta + nx*d3xdzetadtheta2 \
             + dnydtheta*d2ydthetadzeta + ny*d3ydzetadtheta2 \
             + dnzdtheta*d2zdthetadzeta + nz*d3zdzetadtheta2
    dgdtheta = dnxdtheta*d2xdzeta2 + nx*d3xdthetadzeta2 \
             + dnydtheta*d2ydzeta2 + ny*d3ydthetadzeta2 \
             + dnzdtheta*d2zdzeta2 + nz*d3zdthetadzeta2 
    dedzeta = dnxdzeta*d2xdtheta2 + nx*d3xdzetadtheta2 \
            + dnydzeta*d2ydtheta2 + ny*d3ydzetadtheta2 \
            + dnzdzeta*d2zdtheta2 + nz*d3zdzetadtheta2
    dfdzeta = dnxdzeta*d2xdthetadzeta + nx*d3xdthetadzeta2 \
            + dnydzeta*d2ydthetadzeta + ny*d3ydthetadzeta2 \
            + dnzdzeta*d2zdthetadzeta + nz*d3zdthetadzeta2
    dgdzeta = dnxdzeta*d2xdzeta2 + nx*d3xdzeta3 \
            + dnydzeta*d2ydzeta2 + ny*d3ydzeta3 \
            + dnzdzeta*d2zdzeta2 + nz*d3zdzeta3 
    return dedtheta, dfdtheta, dgdtheta, dedzeta, dfdzeta, dgdzeta

  
  def mean_curvature_derivatives(self,isurf):
    [dgthetathetadtheta, dgthetathetadzeta, dgthetazetadtheta, dgthetazetadzeta, \
      dgzetazetadzeta, dgzetazetadtheta] = self.metric_elements_derivatives(isurf)
    [gthetatheta, gthetazeta, gzetazeta] = self.metric_elements(isurf)
    [e,f,g] = self.fun_form_2(isurf)
    [dedtheta, dfdtheta, dgdtheta, dedzeta, dfdzeta, dgdzeta] = self.fun_form_2_derivatives(isurf)

    E = gthetatheta
    F = gthetazeta
    G = gzetazeta

    dEdtheta = dgthetathetadtheta
    dFdtheta = dgthetazetadtheta
    dGdtheta = dgzetazetadtheta
        
    dEdzeta = dgthetathetadzeta
    dFdzeta = dgthetazetadzeta
    dGdzeta = dgzetazetadzeta    
    
    dHdtheta = (G*dedtheta - 2*F*dfdtheta - 2*f*dFdtheta + e*dgdtheta + e*dGdtheta) \
      /(e*G-F**2) - ((-2*f*F+e*g+e*G)*(-2*F*dFdtheta+e*dGdtheta))/((e*G-F*F)**2)    
    dHdzeta = (G*dedzeta - 2*F*dfdzeta - 2*f*dFdzeta + e*dgdzeta + e*dGdzeta) \
      /(e*G-F**2) - ((-2*f*F+e*g+e*G)*(-2*F*dFdzeta+e*dGdzeta))/((e*G-F*F)**2)
    
    return dHdtheta, dHdzeta