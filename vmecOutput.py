import numpy as np
from scipy.io import netcdf
import os

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
    
    self.nfp = f.variables["nfp"][()]
    self.sign_jac = f.variables["signgs"][()]
    
  def d_r_d_u(self,isurf):
    this_rmnc = self.rmnc[isurf,:]
    this_zmns = self.zmns[isurf,:]
    
    dRdu = np.zeros(np.shape(self.thetas_2d))
    dZdu = np.zeros(np.shape(self.thetas_2d))
    for im in range(self.mnmax):
      angle = self.xm[im]*self.thetas_2d - self.xn[im]*self.zetas_2d
      cos_angle = np.cos(angle)
      sin_angle = np.sin(angle)
      dRdu = dRdu - self.xm[im]*this_rmnc[im]*sin_angle
      dZdu = dZdu + self.xm[im]*this_zmns[im]*cos_angle
    dXdu = dRdu*np.cos(self.zetas_2d)
    dYdu = dRdu*np.sin(self.zetas_2d)
    return dXdu, dYdu, dZdu
  
  def d_r_d_v(self,isurf):
    this_rmnc = self.rmnc[isurf,:]
    this_zmns = self.zmns[isurf,:]
    
    dRdv = np.zeros(np.shape(self.thetas_2d))
    dZdv = np.zeros(np.shape(self.thetas_2d))
    R = np.zeros(np.shape(self.thetas_2d))
    for im in range(self.mnmax):
      angle = self.xm[im]*self.thetas_2d - self.xn[im]*self.zetas_2d
      cos_angle = np.cos(angle)
      sin_angle = np.sin(angle)
      dRdv = dRdv + self.xn[im]*this_rmnc[im]*sin_angle
      dZdv = dZdv - self.xn[im]*this_zmns[im]*cos_angle
      R = R + this_rmnc[im]*cos_angle
    dXdv = dRdv*np.cos(self.zetas_2d) - R*np.sin(self.zetas_2d)
    dYdv = dRdv*np.sin(self.zetas_2d) + R*np.cos(self.zetas_2d)
    return dXdv, dYdv, dZdv
  
  def B_on_arclength_grid(self):
    # Extrapolate to last full mesh grid point
    bsupumnc_end = 1.5*self.bsupumnc[-1,:] - 0.5*self.bsupumnc[-2,:]
    bsupvmnc_end = 1.5*self.bsupvmnc[-1,:] - 0.5*self.bsupvmnc[-2,:]
    rmnc_end = self.rmnc[-1,:]
    zmns_end = self.zmns[-1,:]
      
    # Evaluate at last full mesh grid point
    [dXdu_end,dYdu_end,dZdu_end] = self.d_r_d_u(-1)
    [dXdv_end,dYdv_end,dZdv_end] = self.d_r_d_v(-1)
    
    Bsupu_end = np.zeros(np.shape(self.thetas_2d))
    Bsupv_end = np.zeros(np.shape(self.thetas_2d))
    R_end = np.zeros(np.shape(self.thetas_2d))
    Z_end = np.zeros(np.shape(self.thetas_2d))
    for im in range(self.mnmax):
      angle = self.xm[im]*self.thetas_2d - self.xn[im]*self.zetas_2d
      cos_angle = np.cos(angle)
      sin_angle = np.sin(angle)
      Bsupu_end = Bsupu_end + bsupumnc_end[im]*cos_angle
      Bsupv_end = Bsupv_end + bsupvmnc_end[im]*cos_angle
      R_end = R_end + rmnc_end[im]*cos_angle
      Z_end = Z_end + zmns_end[im]*sin_angle
    Bx_end = Bsupu_end*dXdu_end + Bsupv_end*dXdv_end
    By_end = Bsupu_end*dYdu_end + Bsupv_end*dYdv_end
    Bz_end = Bsupu_end*dZdu_end + Bsupv_end*dZdv_end

    theta_arclength = np.zeros(np.shape(self.zetas_2d))
    for izeta in range(self.nzeta):
      for itheta in range(1,self.ntheta):
        dr = np.sqrt((R_end[izeta,itheta]-R_end[izeta,itheta-1])**2 \
                     + (Z_end[izeta,itheta]-Z_end[izeta,itheta-1])**2)
        theta_arclength[izeta,itheta] = theta_arclength[izeta,itheta-1] + dr
    
    return Bx_end, By_end, Bz_end, theta_arclength

  def compute_current(self):
    It_half = self.sign_jac*2*np.pi*self.bsubumnc[:,0]/self.mu0
    return It_half

  def compute_N(self):  
    rmnc_end = self.rmnc[-1,:]
    zmns_end = self.zmns[-1,:]
  
    dRdv = np.zeros(np.shape(self.thetas_2d))
    dRdu = np.zeros(np.shape(self.thetas_2d))
    dZdv = np.zeros(np.shape(self.thetas_2d))
    dZdu = np.zeros(np.shape(self.thetas_2d))
    R = np.zeros(np.shape(self.thetas_2d))
    for im in range(self.mnmax):
      angle = self.xm[im]*self.thetas_2d - self.xn[im]*self.zetas_2d
      sin_angle = np.sin(angle)
      cos_angle = np.cos(angle)
      R = R + rmnc_end[im]*cos_angle
      dRdu = dRdu - self.xm[im]*rmnc_end[im]*sin_angle
      dRdv = dRdv + self.xn[im]*rmnc_end[im]*sin_angle
      dZdu = dZdu + self.xm[im]*zmns_end[im]*cos_angle
      dZdv = dZdv - self.xn[im]*zmns_end[im]*sin_angle
    dXdu = dRdu*np.cos(self.zetas_2d)
    dYdu = dRdu*np.sin(self.zetas_2d)
    dXdv = dRdv*np.cos(self.zetas_2d) - R*np.sin(self.zetas_2d)
    dYdv = dRdv*np.sin(self.zetas_2d) + R*np.cos(self.zetas_2d)
    Nx = dYdv*dZdu - dYdu*dZdv
    Ny = dZdv*dXdu - dZdu*dXdv
    Nz = dXdv*dYdu - dXdu*dYdv
    return Nx, Ny, Nz

  # Returns X, Y, Z at isurf point on full mesh 
  def compute_position(self,isurf):
    this_rmnc = self.rmnc[isurf,:]
    this_zmns = self.zmns[isurf,:] 

    R = np.zeros(np.shape(self.thetas_2d))
    Z = np.zeros(np.shape(self.thetas_2d))
    for im in range(self.mnmax):
      angle = self.xm[im]*self.thetas_2d - self.xn[im]*self.zetas_2d
      cos_angle = np.cos(angle)
      sin_angle = np.sin(angle)
      R = R + this_rmnc[im]*cos_angle
      Z = Z + this_zmns[im]*sin_angle
    X = R*np.cos(self.zetas_2d)
    Y = R*np.sin(self.zetas_2d)
    return X, Y, Z

  # Integrated rotational transform with weight function
  def evaluate_iota_objective(self,weight):
    return np.sum(weight(self.s_half)*self.iota)*self.ds*self.psi[-1]*self.sign_jac
  
  # Integrated differential volume with weight function
  def evaluate_well_objective(self,weight):
    return 4*np.pi*np.pi*np.sum(weight(self.s_half)*self.vp)*self.ds*self.psi[-1]