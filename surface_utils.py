import numpy as np
import numba
from numba import jit, prange
from scanf import scanf
import re

@jit(nopython=True,cache=True)
def cosine_IFT(xm,xn,nfp,theta,zeta,fmn):
    """
    Performs cosine inverse fourier transform
    
    Args:
        xm (np array)
        xn (np array)
        nfp (int) : number of field periods
        theta (np array) : poloidal grid created with meshgrid
        zeta (np array) : toroidal grid created with meshgrid
        fmn (np array) : fourier modes of function 
        
    - Assumes len(xm) = len(xn) = len(fmn)
    - Assumes shape(zeta) = shape(theta) = (nzeta,ntheta)
    
    f = sum_i fmn[i] cos(xm[i]*theta - nfp*xn[i]*zeta)
    
    Returns: 
        f (np array) : IFT function. Same shape as theta and zeta. 
    """
    f = np.zeros(np.shape(theta))
    mnmax = len(xm)
    for im in range(mnmax):
        angle = xm[im] * theta - nfp * xn[im] * zeta
        cos_angle = np.cos(angle)
        f += fmn[im]*cos_angle
    return f

@jit(nopython=True,cache=True)
def sine_IFT(xm,xn,nfp,theta,zeta,fmn):
    """
    Performs sine inverse fourier transform
    
    Args:
        xm (np array)
        xn (np array)
        nfp (int) : number of field periods
        theta (np array) : poloidal grid created with meshgrid
        zeta (np array) : toroidal grid created with meshgrid
        fmn (np array) : fourier modes of function 
        
    - Assumes len(xm) = len(xn) = len(fmn)
    - Assumes shape(zeta) = shape(theta) = (nzeta,ntheta)
    
    f = sum_i fmn[i] sin(xm[i]*theta - nfp*xn[i]*zeta)
    
    Returns: 
        f (np array) : IFT function. Same shape as theta and zeta. 
    """
    f = np.zeros(np.shape(theta))
    mnmax = len(xm)
    for im in range(mnmax):
        angle = xm[im] * theta - nfp * xn[im] * zeta
        sin_angle = np.sin(angle)
        f += fmn[im]*sin_angle
    return f

@jit(nopython=True,cache=True)
def proximity_surface(R,Z,tR,tz,dldtheta,min_curvature_radius=0.2,exp_weight=0.01,\
                      derivatives=False):
    """
    Compute proximity function for curves defined as 
    
    """
    nzeta = len(R[:,0])
    ntheta = len(R[0,:])
    Qp = np.zeros((nzeta,ntheta))
    dQpdp1 = np.zeros((nzeta,ntheta,2))
    dQpdp2 = np.zeros((nzeta,ntheta,ntheta,2))
    dQpdtau2 = np.zeros((nzeta,ntheta,ntheta,2))
    dQpdlprime = np.zeros((nzeta,ntheta,ntheta))
    for izeta in range(nzeta):
        if derivatives:
            [Qp[izeta,:],dQpdp1[izeta,:,:],dQpdp2[izeta,:,:,:],\
                 dQpdtau2[izeta,:,:,:],dQpdlprime[izeta,:]] = \
                 proximity_slice(R[izeta,:],Z[izeta,:],tR[izeta,:],\
                                 tz[izeta,:],dldtheta[izeta,:],
            derivatives=derivatives,min_curvature_radius=min_curvature_radius,\
                                 exp_weight=exp_weight)
        else:
            [Qp[izeta,:],dQpdp1[izeta,:,:],dQpdp2[izeta,:,:,:],\
                 dQpdtau2[izeta,:,:,:],dQpdlprime[izeta,:]] = \
                 proximity_slice(R[izeta,:],Z[izeta,:],tR[izeta,:],\
                          tz[izeta,:],dldtheta[izeta,:],derivatives=derivatives,\
                min_curvature_radius=min_curvature_radius,exp_weight=exp_weight)
    return Qp, dQpdp1, dQpdp2, dQpdtau2, dQpdlprime

@jit(nopython=True,cache=True)
def global_curvature_surface(R,Z,tR,tz):
    nzeta = len(R[:,0])
    ntheta = len(R[0,:])
    global_curvature_radius = np.zeros((nzeta,ntheta))
    for izeta in range(nzeta):
        for itheta in range(ntheta):
            p1 = np.array([R[izeta,itheta],Z[izeta,itheta]])
            this_global_curvature = np.zeros(ntheta)
            for ithetap in range(ntheta):
                p2 = np.array([R[izeta,ithetap],Z[izeta,ithetap]])
                t2 = np.array([tR[izeta,ithetap],tz[izeta,ithetap]])
                if (np.any(p1 != p2)):
                    this_global_curvature[ithetap] = self_contact(p1,p2,t2)
                else:
                    this_global_curvature[ithetap] = 1e12
            global_curvature_radius[izeta,itheta] = np.min(this_global_curvature)
    return global_curvature_radius

#TODO : test for correct sizes
@jit(nopython=True,cache=True)
def point_in_polygon(R, Z, R0, Z0):
    """
    Determines if point on the axis (R0,Z0) lies within boundary defined by
        (R,Z), a toroidal slice of the boundary
        
    Args:
        R (float array): radius defining toroidal slice of boundary
        Z (float array): height defining toroidal slice of boundary
            evaluation
        R0 (float): radius of trial axis point
        Z0 (float): height of trial axis point
    Returns:
        oddNodes (bool): True if (R0,Z0) lies in (R,Z)
        
    """

    ntheta = len(R)
    oddNodes = False
    j = ntheta-1
    for i in range(ntheta):
        if ((Z[i] < Z0 and Z[j] >= Z0) or (Z[j] < Z0 and Z[i] >= Z0)):
            if (R[i] + (Z0 - Z[i]) / (Z[j] - Z[i]) * (R[j] - R[i]) < R0):
                oddNodes = not oddNodes
        j = i
    return oddNodes

# Note that xn is not multiplied by nfp
@jit(nopython=True,cache=True)
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

def read_bounary_harmonic(varName,inputFilename):
    varName = varName.lower()
    index_1 = []
    index_2 = []
    value = []
    with open(inputFilename, 'r') as f:
        inputFile = f.readlines()
        for line in inputFile:
            line3 = line.strip().lower()
            line3 = re.sub(r"\s+", "", line3, flags=re.UNICODE)
            # Ignore any comments
            if (line3[0]=='!'):
                continue
            find_index = line3.find(varName+'(')
            # Line contains desired varName
            if (find_index > -1):
                out = scanf(varName+"(%d,%d)=%f",line3[find_index::].lower(),\
                            collapseWhitespace=True)
                index_1.append(out[0])
                index_2.append(out[1])
                value.append(out[2])
    return index_1, index_2, value
    

# Computes minimum indices of 2d array in Fortran namelist
# @jit(nopython=True)
def min_max_indices_2d(varName,inputFilename):
    varName = varName.lower()
    index_1 = []
    index_2 = []
    with open(inputFilename, 'r') as f:
        inputFile = f.readlines()
        for line in inputFile:
            line3 = line.strip().lower()
            find_index = line3.find(varName+'(')
            # Line contains desired varName
            if (find_index > -1):
                out = scanf(varName+"(%d,%d)",line[find_index::].lower())
                index_1.append(out[0])
                index_2.append(out[1])
    return min(index_1), min(index_2), max(index_1), max(index_2)

@jit(nopython=True,cache=True)
def proximity_slice(R,Z,tR,tZ,dldtheta,min_curvature_radius=0.2,exp_weight=0.01,\
                    derivatives=False):
#     assert(np.all((np.shape(R) == np.shape(Z))) and \
#            np.all((np.shape(R) == np.shape(tR))) and \
#            np.all((np.shape(R) == np.shape(tZ))) and \
#            np.all(np.shape(R) == np.shape(dldtheta)))
    ntheta = len(R)
    Qp = np.zeros((ntheta))
    dQpdp1 = np.zeros((ntheta,2))
    dQpdp2 = np.zeros((ntheta,ntheta,2))
    dQpdtau2 = np.zeros((ntheta,ntheta,2))
    dQpdlprime = np.zeros((ntheta,ntheta))
    for itheta in range(ntheta):
        p1 = np.array([R[itheta],Z[itheta]])
        Sc = np.zeros(ntheta)
        if derivatives:
            dScdp1 = np.zeros((ntheta,2))
        for ithetap in range(ntheta):
            p2 = np.array([R[ithetap],Z[ithetap]])
            tau2 = np.array([tR[ithetap],tZ[ithetap]])
            if (np.any(p1 != p2)):
                # Sc(p1,p2)
                Sc[ithetap] = self_contact_exp(p1,p2,tau2,min_curvature_radius,exp_weight)
                if derivatives:
                    [dScdp1[ithetap,:],dScdp2,dScdtau2] = \
                        self_contact_exp_gradient(p1,p2,tau2,min_curvature_radius,exp_weight)
                    dQpdp2[itheta,ithetap,:] = dScdp2*dldtheta[ithetap]
                    dQpdtau2[itheta,ithetap,:] = dScdtau2*dldtheta[ithetap]
                    dQpdlprime[itheta,ithetap] = Sc[ithetap]
        # Integral over ithetap
        Qp[itheta] = np.dot(Sc,dldtheta)
        if derivatives:
            dQpdp1[itheta,:] = np.dot(dScdp1.T,dldtheta)
    return Qp, dQpdp1, dQpdp2, dQpdtau2, dQpdlprime

@jit(nopython=True,cache=True)
def proximity_derivatives_func(theta,zeta,nfp,xm_sensitivity,xn_sensitivity,\
                                  dQpdp1,dQpdp2,dQpdtau2,dQpdlprime,\
                                  dRdtheta,dZdtheta,lprime):

    mnmax_sensitivity = len(xm_sensitivity)
    nzeta = len(dRdtheta[:,0])
    ntheta = len(dRdtheta[0,:])
    
    dQpdrmnc = np.zeros((nzeta,mnmax_sensitivity,ntheta))
    dQpdzmns = np.zeros((nzeta,mnmax_sensitivity,ntheta))
    for izeta in range(nzeta):
        for imn in range(mnmax_sensitivity):
            angle = xm_sensitivity[imn]*theta[izeta,:] \
                - nfp*xn_sensitivity[imn]*zeta[izeta,:]
            dRdrmnc = np.cos(angle)
            dZdzmns = np.sin(angle)
            d2Rdthetadrmnc = -xm_sensitivity[imn]*np.sin(angle)
            d2Zdthetadzmns = xm_sensitivity[imn]*np.cos(angle)
            dlprimedrmnc = dRdtheta[izeta,:]*d2Rdthetadrmnc/lprime[izeta,:]
            dlprimedzmns = dZdtheta[izeta,:]*d2Zdthetadzmns/lprime[izeta,:]
            dtauRdrmnc = d2Rdthetadrmnc/lprime[izeta,:] \
                - dRdtheta[izeta,:]*dlprimedrmnc/lprime[izeta,:]**2
            dtauRdzmns = - dRdtheta[izeta,:]*dlprimedzmns/lprime[izeta,:]**2
            dtauZdrmnc = - dZdtheta[izeta,:]*dlprimedrmnc/lprime[izeta,:]**2
            dtauZdzmns = d2Zdthetadzmns/lprime[izeta,:] \
                - dZdtheta[izeta,:]*dlprimedzmns/lprime[izeta,:]**2
            for itheta in range(ntheta):
                dQpdrmnc[izeta,imn,itheta] = dQpdp1[izeta,itheta,0]*dRdrmnc[itheta] \
                    + np.sum(dQpdp2[izeta,itheta,:,0]*dRdrmnc) \
                    + np.sum(dQpdtau2[izeta,itheta,:,0]*dtauRdrmnc) \
                    + np.sum(dQpdtau2[izeta,itheta,:,1]*dtauZdrmnc) \
                    + np.dot(dQpdlprime[izeta,itheta,:],dlprimedrmnc)
                dQpdzmns[izeta,imn,itheta] = dQpdp1[izeta,itheta,1]*dZdzmns[itheta] \
                    + np.sum(dQpdp2[izeta,itheta,:,1]*dZdzmns) \
                    + np.sum(dQpdtau2[izeta,itheta,:,0]*dtauRdzmns) \
                    + np.sum(dQpdtau2[izeta,itheta,:,1]*dtauZdzmns) \
                    + np.dot(dQpdlprime[izeta,itheta,:],dlprimedzmns)
    return dQpdrmnc, dQpdzmns

    
@jit(nopython=True,cache=True)
def self_contact(p1,p2,tau2):
    if (np.all(p1 == p2)):
        return 1e12
    norm = np.linalg.norm(p1-p2)
    normal_distance_ratio = normalDistanceRatio(p1,p2,tau2)
    if (normal_distance_ratio > 0):
        return norm/normal_distance_ratio
    else:
        return 1e12
    
@jit(nopython=True,cache=True)
def normalDistanceRatio(p1,p2,tau2):
    norm = np.linalg.norm(p1-p2)
    return 1-((p2-p1).dot(tau2))**2/norm**2

@jit(nopython=True,cache=True)
def normalDistanceRatio_gradient(p1,p2,tau2):
    norm = np.linalg.norm(p1-p2)
    dnormalDistanceRatiodp1 = -2*((p2-p1).dot(tau2)/norm)*(-tau2/norm + ((p2-p1).dot(tau2)/norm**3)*(p2-p1))
    dnormalDistanceRatiodp2 = -2*((p2-p1).dot(tau2)/norm)*( tau2/norm + ((p2-p1).dot(tau2)/norm**3)*(p1-p2))
    dnormalDistanceRatiodtau2 = -2*((p2-p1).dot(tau2)/norm)*((p2-p1)/norm)
    return dnormalDistanceRatiodp1,dnormalDistanceRatiodp2,dnormalDistanceRatiodtau2
  
# Given 2 points p1 and p2 and tangent at p2, compute self-contact function [(26) in Walker]
@jit(nopython=True,cache=True)
def self_contact_exp(p1,p2,tau2,min_curvature_radius=0.2,exp_weight=0.1):
    norm = np.linalg.norm(p1-p2)
    normal_distance_ratio = normalDistanceRatio(p1,p2,tau2)
    if (normal_distance_ratio > 0):
        return np.exp(-(norm/normal_distance_ratio-min_curvature_radius)/exp_weight)
    else:
        return 0.

@jit(nopython=True,cache=True)
def self_contact_exp_gradient(p1,p2,tau2,min_curvature_radius=0.2,exp_weight=0.01):
    norm = np.linalg.norm(p1-p2)
    N = normalDistanceRatio(p1,p2,tau2) # norm distance ratio squared
    [dnormalDistanceRatiodp1,dnormalDistanceRatiodp2,dnormalDistanceRatiodtau2] \
        = normalDistanceRatio_gradient(p1,p2,tau2)
    if (N != 0):
        dScdp1 = (p1-p2)/(norm*N) - norm*dnormalDistanceRatiodp1/(N*N)
        dScdp2 = (p2-p1)/(norm*N) - norm*dnormalDistanceRatiodp2/(N*N)
        dScdtau2 = - norm*dnormalDistanceRatiodtau2/(N*N)
        return -self_contact_exp(p1,p2,tau2,min_curvature_radius,exp_weight)*dScdp1/exp_weight,\
            -self_contact_exp(p1,p2,tau2,min_curvature_radius,exp_weight)*dScdp2/exp_weight,\
            -self_contact_exp(p1,p2,tau2,min_curvature_radius,exp_weight)*dScdtau2/exp_weight
    else:
        return np.array([0.,0.]), np.array([0.,0.]), np.array([0.,0.])
    
@jit(nopython=True,cache=True)
def self_intersect(x,y):
    assert(x.ndim==1 and y.ndim==1 and len(x)==len(y))
    # Repeat last point
    x = np.append(x,x[0])
    y = np.append(y,y[0])
  
    npoints = len(x)
    for i in range(npoints-1):
        p1 = [x[i],y[i]]
        p2 = [x[i+1],y[i+1]]
        # Compare with all line segments that do not contain x[i] or x[i+1]
        for j in range(i+2,npoints-1):
            p3 = [x[j],y[j]]
            p4 = [x[j+1],y[j+1]]
            # Ignore if any points are in common with p1
            if (p1==p3 or p2==p3 or p1==p4 or p2==p4):
                continue
            if (segment_intersect(np.array(p1),np.array(p2),np.array(p3),\
                                  np.array(p4))):
                return True
    return False

@jit(nopython=True,cache=True)
def segment_intersect(p1,p2,p3,p4):
    """
    Check if segment defined by (p1, p2) intersections segment defined by (p2, p4)
    
    Args:
        p1 
    
    """
    l1 = p3 - p1
    l2 = p2 - p1
    l3 = p4 - p1
  
    # Check for intersection of bounding box 
    b1 = [min(p1[0],p2[0]),min(p1[1],p2[1])]
    b2 = [max(p1[0],p2[0]),max(p1[1],p2[1])]
    b3 = [min(p3[0],p4[0]),min(p3[1],p4[1])]
    b4 = [max(p3[0],p4[0]),max(p3[1],p4[1])]
  
    boundingBoxIntersect = ((b1[0] <= b4[0]) and (b2[0] >= b3[0]) 
        and (b1[1] <= b4[1]) and (b2[1] >= b3[1]))
    return (((l1[0]*l2[1]-l1[1]*l2[0])*(l3[0]*l2[1]-l3[1]*l2[0]) < 0) and 
        boundingBoxIntersect)

@jit(nopython=True,cache=True)
def surface_intersect(R,Z):  
    nzeta = len(R[:,0])
    for izeta in range(nzeta):
        if (self_intersect(R[izeta,:],Z[izeta,:])):
            return izeta
    return -1
        
