import unittest
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')

from vmec_input import VmecInput
from finite_differences import finite_difference_derivative

class Test(unittest.TestCase):

    vmecInput = VmecInput('input.w7x',ntheta=5,nzeta=5)
    
    def test_update_namelist(self):
        """
        Test that nfp is correctly updated in the namelist
        """
        nfp_init = self.vmecInput.nfp
        self.vmecInput.nfp = 20
        self.vmecInput.update_namelist()
        self.assertEqual(self.vmecInput.nfp, self.vmecInput.namelist['nfp'])
        self.vmecInput.nfp = nfp_init
        self.vmecInput.update_namelist()
        self.assertEqual(self.vmecInput.nfp, self.vmecInput.namelist['nfp'])
        self.assertEqual(nfp_init, self.vmecInput.nfp)
        
    def test_update_modes(self):
        """
        Test that number of modes, arrays (xm, xn, rbc, zbs), and length
        (mnmax) are updated correctly
        """        
        mpol = self.vmecInput.mpol
        ntor = self.vmecInput.ntor
        mnmax = self.vmecInput.mnmax
        xm = np.copy(self.vmecInput.xm)
        xn = np.copy(self.vmecInput.xn)
        rbc = np.copy(self.vmecInput.rbc)
        zbs = np.copy(self.vmecInput.zbs)
        # Adjust modes 
        self.vmecInput.update_modes(2 * mpol, 2 * ntor)
        self.assertEqual(np.max(self.vmecInput.xm), 2 * mpol - 1)
        self.assertEqual(np.max(self.vmecInput.xn), 2 * ntor)
        self.assertEqual(len(self.vmecInput.rbc), self.vmecInput.mnmax)
        self.assertEqual(len(self.vmecInput.zbs), self.vmecInput.mnmax)
        self.assertEqual(len(self.vmecInput.xm), self.vmecInput.mnmax)
        self.assertEqual(len(self.vmecInput.xn), self.vmecInput.mnmax)
        # Return to initial values
        self.vmecInput.update_modes(mpol, ntor)
        self.assertEqual(mnmax, self.vmecInput.mnmax)
        self.assertTrue(np.all(xm == self.vmecInput.xm))
        self.assertTrue(np.all(xn == self.vmecInput.xn))
        self.assertTrue(np.all(rbc == self.vmecInput.rbc))
        self.assertTrue(np.all(zbs == self.vmecInput.zbs))
        
    def test_update_grids(self):
        """
        Check that grids update correctly and reset back to initial size. 
        """
        # Save original items
        ntheta = self.vmecInput.ntheta
        nzeta = self.vmecInput.nzeta
        thetas = np.copy(self.vmecInput.thetas)
        zetas = np.copy(self.vmecInput.zetas)
        thetas_2d = np.copy(self.vmecInput.thetas_2d)
        zetas_2d = np.copy(self.vmecInput.zetas_2d)
        dtheta = np.copy(self.vmecInput.dtheta)
        dzeta = np.copy(self.vmecInput.dzeta)
        zetas_full = np.copy(self.vmecInput.zetas_full)
        thetas_2d_full = np.copy(self.vmecInput.thetas_2d_full)
        zetas_2d_full = np.copy(self.vmecInput.zetas_2d_full)
        nfp = self.vmecInput.nfp
        # Update grids
        self.vmecInput.update_grids(2*ntheta, 2*nzeta)
        self.assertEqual(self.vmecInput.ntheta, 2*ntheta)
        self.assertEqual(self.vmecInput.nzeta, 2*nzeta)
        self.assertEqual(len(self.vmecInput.thetas), 2*ntheta)
        self.assertEqual(len(self.vmecInput.zetas), 2*nzeta)
        self.assertEqual(dtheta*ntheta, 2*self.vmecInput.dtheta*ntheta)
        self.assertEqual(dzeta*nzeta, 2*self.vmecInput.dzeta*nzeta)
        self.assertEqual(len(self.vmecInput.zetas_full), \
                         nfp * nzeta * 2)
        self.assertEqual(len(self.vmecInput.thetas_2d_full[:,0]), nzeta * 2 * nfp)
        self.assertEqual(len(self.vmecInput.zetas_2d_full[:,0]), nzeta * 2 * nfp)
        self.assertEqual(len(self.vmecInput.thetas_2d_full[0,:]), ntheta * 2)
        self.assertEqual(len(self.vmecInput.zetas_2d_full[0,:]), ntheta * 2)
        # Reset
        self.vmecInput.update_grids(ntheta, nzeta)
        self.assertEqual(self.vmecInput.ntheta, ntheta)
        self.assertEqual(self.vmecInput.nzeta, nzeta)
        self.assertTrue(np.all(self.vmecInput.thetas == thetas))
        self.assertTrue(np.all(self.vmecInput.zetas == zetas))
        self.assertEqual(dtheta, self.vmecInput.dtheta)
        self.assertEqual(dzeta, self.vmecInput.dzeta)
        self.assertTrue(np.all(self.vmecInput.zetas_full == zetas_full))
        self.assertTrue(np.all(self.vmecInput.thetas_2d_full == thetas_2d_full))
        self.assertTrue(np.all(self.vmecInput.zetas_2d_full == zetas_2d_full))
        
    def test_read_boundary_input(self):
        """
        Check that every non-zero element in rbc_array is in rbc (and for zbs)
        """
        rbc_array = np.array(self.vmecInput.namelist["rbc"]).T
        zbs_array = np.array(self.vmecInput.namelist["zbs"]).T
        
        [rbc, zbs] = self.vmecInput.read_boundary_input()
        for im in range(len(rbc_array[0,:])):
            for jn in range(len(rbc_array[:,0])):
                if (rbc_array[jn,im] == 0):
                    self.assertTrue(np.any(rbc == rbc_array[jn,im]))
        for im in range(len(zbs_array[0,:])):
            for jn in range(len(zbs_array[:,0])):
                if (zbs_array[jn,im] == 0):
                    self.assertTrue(np.any(zbs == zbs_array[jn,im]))
                    
    def test_area(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Test that a positive float is returned.
        """
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        self.assertRaises(ValueError, self.vmecInput.area, \
            theta = thetas, zeta = zetas)
        area = self.vmecInput.area()
        self.assertIsInstance(area, float)
        self.assertGreater(area, 0)
        
    def test_area_derivatives(self):
        """
        Performs finite difference testing of derivatives.
        """ 
        # Check first 10 modes
        xm_sensitivity = self.vmecInput.xm[0:10]
        xn_sensitivity = self.vmecInput.xn[0:10]
        
        [dareadrmnc, dareadzmns] = self.vmecInput.area_derivatives(\
                                                xm_sensitivity,xn_sensitivity)

        def temp_fun_rbc(im, rbc):
            rbc_init = self.vmecInput.rbc[im]
            self.vmecInput.rbc[im] = rbc
            area = self.vmecInput.area()
            self.vmecInput.rbc[im] = rbc_init
            return area

        def temp_fun_zbs(im, zbs):
            zbs_init = self.vmecInput.zbs[im]
            self.vmecInput.zbs[im] = zbs
            area = self.vmecInput.area()
            self.vmecInput.zbs[im] = zbs_init
            return area

        for im in range(len(xm_sensitivity)):
            this_rbc = self.vmecInput.rbc[im]
            this_zbs = self.vmecInput.zbs[im]
            dareadrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc), epsilon=1e-8, \
                method='centered')
            dareadzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs), epsilon=1e-8, \
                method='centered')
            self.assertAlmostEqual(dareadrmnc_fd, dareadrmnc[im], places=5)
            self.assertAlmostEqual(dareadzmns_fd, dareadzmns[im], places=5)
        
    def test_volume(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Test that a positive float is returned.
        """
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        self.assertRaises(ValueError, self.vmecInput.volume, \
            theta = thetas, zeta = zetas)
        volume = self.vmecInput.volume()
        self.assertIsInstance(volume, float)
        self.assertGreater(volume, 0)
        
    def test_volume_derivatives(self):
        """
        Perform finite-difference testing of derivatives.
        """
        xm_sensitivity = self.vmecInput.xm[0:10]
        xn_sensitivity = self.vmecInput.xn[0:10]
        
        [dvolumedrmnc, dvolumedzmns] = self.vmecInput.volume_derivatives(\
                                                xm_sensitivity,xn_sensitivity)

        def temp_fun_rbc(im, rbc):
            rbc_init = self.vmecInput.rbc[im]
            self.vmecInput.rbc[im] = rbc
            volume = self.vmecInput.volume()
            self.vmecInput.rbc[im] = rbc_init
            return volume

        def temp_fun_zbs(im, zbs):
            zbs_init = self.vmecInput.zbs[im]
            self.vmecInput.zbs[im] = zbs
            volume = self.vmecInput.volume()
            self.vmecInput.zbs[im] = zbs_init
            return volume

        for im in range(len(xm_sensitivity)):
            this_rbc = self.vmecInput.rbc[im]
            this_zbs = self.vmecInput.zbs[im]
            dvolumedrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc), epsilon=1e-8, \
                method='centered')
            dvolumedzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs), epsilon=1e-8, \
                method='centered')
            self.assertAlmostEqual(dvolumedrmnc_fd, dvolumedrmnc[im], places=5)
            self.assertAlmostEqual(dvolumedzmns_fd, dvolumedzmns[im], places=5)
        
    def test_normal(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Check that shape of ouput matches input.
        """
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        self.assertRaises(ValueError, self.vmecInput.normal, \
            theta = thetas, zeta = zetas)
        [Nx, Ny, Nz] = self.vmecInput.normal()
        self.assertTrue(np.all(np.shape(Nx) == np.shape(thetas)))
        self.assertTrue(np.all(np.shape(Ny) == np.shape(thetas)))
        self.assertTrue(np.all(np.shape(Nz) == np.shape(thetas)))
        
    def test_position_first_derivatives(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Check that shape of ouput matches input.
        Perform finite-difference testing of derivatives.
        """
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        self.assertRaises(ValueError, self.vmecInput.position_first_derivatives, \
            theta = thetas, zeta = zetas)
        [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, \
                dRdtheta, dRdzeta] = self.vmecInput.position_first_derivatives()
#         self.assertTrue(np.all(np.shape(dxdtheta) == np.shape(thetas)))
#         self.assertTrue(np.all(np.shape(dxdzeta) == np.shape(thetas)))
#         self.assertTrue(np.all(np.shape(dydtheta) == np.shape(thetas)))   
#         self.assertTrue(np.all(np.shape(dydzeta == np.shape(thetas))))
#         self.assertTrue(np.all(np.shape(dzdtheta == np.shape(thetas))))
#         self.assertTrue(np.all(np.shape(dzdzeta == np.shape(thetas))))
#         self.assertTrue(np.all(np.shape(dRdtheta == np.shape(thetas))))
#         self.assertTrue(np.all(np.shape(dRdzeta == np.shape(thetas))))
                
        for itheta in range(0,self.vmecInput.ntheta,5):
            for izeta in range(0,self.vmecInput.nzeta,5):
                this_zeta = self.vmecInput.zetas[izeta]
                this_theta = self.vmecInput.thetas[itheta]
                dxdzeta_fd = finite_difference_derivative(this_zeta,\
                    lambda zeta : self.vmecInput.position(this_theta,zeta)[0], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dxdzeta_fd, dxdzeta[izeta,itheta], 
                                       places=5)
                dydzeta_fd = finite_difference_derivative(this_zeta,\
                    lambda zeta : self.vmecInput.position(this_theta,zeta)[1], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dydzeta_fd, dydzeta[izeta,itheta], 
                                       places=5)
                dzdzeta_fd = finite_difference_derivative(this_zeta,\
                    lambda zeta : self.vmecInput.position(this_theta,zeta)[2], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dzdzeta_fd, dzdzeta[izeta,itheta], 
                                       places=5)
                dRdzeta_fd = finite_difference_derivative(this_zeta,\
                    lambda zeta : self.vmecInput.position(this_theta,zeta)[3], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dRdzeta_fd, dRdzeta[izeta,itheta], 
                                       places=5)
                dxdtheta_fd = finite_difference_derivative(this_theta,\
                    lambda theta : self.vmecInput.position(theta,this_zeta)[0], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dxdtheta_fd, dxdtheta[izeta,itheta], 
                                       places=5)
                dydtheta_fd = finite_difference_derivative(this_theta,\
                    lambda theta : self.vmecInput.position(theta,this_zeta)[1], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dydtheta_fd, dydtheta[izeta,itheta], 
                                       places=5)
                dzdtheta_fd = finite_difference_derivative(this_theta,\
                    lambda theta : self.vmecInput.position(theta,this_zeta)[2], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dzdtheta_fd, dzdtheta[izeta,itheta], 
                                       places=5)
                dRdtheta_fd = finite_difference_derivative(this_theta,\
                    lambda theta : self.vmecInput.position(theta,this_zeta)[3], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(dRdtheta_fd, dRdtheta[izeta,itheta], 
                                       places=5) 
                
    def test_position_first_derivatives_surface(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Check that shape of ouput matches input.
        Performs finite difference testing of derivatives.
        """ 
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
#         zetas = np.delete(zetas,0,0)
        
        epsilon = 1e-3

        # Check first 10 modes
        xm_sensitivity = self.vmecInput.xm[0:10]
        xn_sensitivity = self.vmecInput.xn[0:10]

#         self.assertRaises(ValueError, \
#                           self.vmecInput.position_first_derivatives_surface, 
#                           xm_sensitivity, xn_sensitivity, \
#                           theta = thetas, zeta = zetas)
        [d2Rdthetadrmnc, d2Xdthetadrmnc, d2Ydthetadrmnc, d2Zdthetadzmns, \
            d2Rdzetadrmnc, d2Xdzetadrmnc, d2Ydzetadrmnc, d2Zdzetadzmns] = \
                self.vmecInput.position_first_derivatives_surface(xm_sensitivity, \
                                                           xn_sensitivity)

        def temp_fun_rbc(im, rbc):
            rbc_init = self.vmecInput.rbc[im]
            self.vmecInput.rbc[im] = rbc
            [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, \
                dRdtheta, dRdzeta] \
                = self.vmecInput.position_first_derivatives()
            self.vmecInput.rbc[im] = rbc_init
            return dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, \
                dRdtheta, dRdzeta

        def temp_fun_zbs(im, zbs):
            zbs_init = self.vmecInput.zbs[im]
            self.vmecInput.zbs[im] = zbs
            [dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, \
                dRdtheta, dRdzeta] \
                = self.vmecInput.position_first_derivatives()
            self.vmecInput.zbs[im] = zbs_init
            return dxdtheta, dxdzeta, dydtheta, dydzeta, dzdtheta, dzdzeta, \
                dRdtheta, dRdzeta

        for im in range(len(xm_sensitivity)):
            this_rbc = self.vmecInput.rbc[im]
            this_zbs = self.vmecInput.zbs[im]
            d2Xdthetadrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[0], epsilon=epsilon, \
                method='centered')
            d2Xdzetadrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[1], epsilon=epsilon, \
                method='centered')
            d2Ydthetadrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[2], epsilon=epsilon, \
                method='centered')
            d2Ydzetadrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[3], epsilon=epsilon, \
                method='centered')
            d2Zdthetadzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs)[4], epsilon=epsilon, \
                method='centered')
            d2Zdzetadzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs)[5], epsilon=epsilon, \
                method='centered')
            d2Rdthetadrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[6], epsilon=epsilon, \
                method='centered')
            d2Rdzetadrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[7], epsilon=epsilon, \
                method='centered')
            for izeta in range(self.vmecInput.nzeta):
                for itheta in range(self.vmecInput.ntheta):
                    self.assertAlmostEqual(d2Xdthetadrmnc_fd[izeta, itheta], \
                        d2Xdthetadrmnc[im,izeta,itheta], places=4)
                    self.assertAlmostEqual(d2Xdzetadrmnc_fd[izeta, itheta], \
                        d2Xdzetadrmnc[im,izeta,itheta], places=4)     
                    self.assertAlmostEqual(d2Ydthetadrmnc_fd[izeta, itheta], \
                        d2Ydthetadrmnc[im,izeta,itheta], places=4)        
                    self.assertAlmostEqual(d2Ydzetadrmnc_fd[izeta, itheta], \
                        d2Ydzetadrmnc[im,izeta,itheta], places=4)     
                    self.assertAlmostEqual(d2Zdthetadzmns_fd[izeta, itheta], \
                        d2Zdthetadzmns[im,izeta,itheta], places=4)  
                    self.assertAlmostEqual(d2Zdzetadzmns_fd[izeta, itheta], \
                        d2Zdzetadzmns[im,izeta,itheta], places=4)
                    self.assertAlmostEqual(d2Rdthetadrmnc_fd[izeta, itheta], \
                        d2Rdthetadrmnc[im,izeta,itheta], places=4) 
                    self.assertAlmostEqual(d2Rdzetadrmnc_fd[izeta, itheta], \
                        d2Rdzetadrmnc[im,izeta,itheta], places=4) 
                
    def test_position_second_derivatives(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Check that shape of ouput matches input.
        Perform finite-difference testing of derivatives.
        """
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        self.assertRaises(ValueError, self.vmecInput.position_second_derivatives, \
            theta = thetas, zeta = zetas)
        [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, \
                d2ydthetadzeta, d2zdtheta2, d2zdzeta2, d2zdthetadzeta, \
               d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta] \
                = self.vmecInput.position_second_derivatives()
#         self.assertTrue(np.all(np.shape(d2xdtheta2) == np.shape(thetas)))
#         self.assertTrue(np.all(np.shape(d2xdzeta2) == np.shape(thetas)))
#         self.assertTrue(np.all(np.shape(d2xdthetadzeta) == np.shape(thetas)))   
#         self.assertTrue(np.all(np.shape(d2ydtheta2 == np.shape(thetas))))
#         self.assertTrue(np.all(np.shape(d2ydzeta2 == np.shape(thetas))))
#         self.assertTrue(np.all(np.shape(d2ydthetadzeta == np.shape(thetas))))
#         self.assertTrue(np.all(np.shape(d2Zdtheta2 == np.shape(thetas))))
#         self.assertTrue(np.all(np.shape(d2Zdzeta2 == np.shape(thetas))))
#         self.assertTrue(np.all(np.shape(d2Zdthetadzeta == np.shape(thetas))))
                
        for itheta in range(0,self.vmecInput.ntheta,5):
            for izeta in range(0,self.vmecInput.nzeta,5):
                this_zeta = self.vmecInput.zetas[izeta]
                this_theta = self.vmecInput.thetas[itheta]
                d2xdtheta2_fd = finite_difference_derivative(this_theta,\
                    lambda theta : self.vmecInput.position_first_derivatives(theta,this_zeta)[0], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2xdtheta2_fd, d2xdtheta2[izeta,itheta], 
                                       places=5)
                d2xdzeta2_fd = finite_difference_derivative(this_zeta,\
                    lambda zeta : self.vmecInput.position_first_derivatives(this_theta,zeta)[1], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2xdzeta2_fd, d2xdzeta2[izeta,itheta], 
                                       places=5)
                d2xdthetadzeta_fd = finite_difference_derivative(this_zeta,\
                    lambda zeta : self.vmecInput.position_first_derivatives(this_theta,zeta)[0], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2xdthetadzeta_fd, d2xdthetadzeta[izeta,itheta], 
                                       places=5)                
                d2ydtheta2_fd = finite_difference_derivative(this_theta,\
                    lambda theta : self.vmecInput.position_first_derivatives(theta,this_zeta)[2], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2ydtheta2_fd, d2ydtheta2[izeta,itheta], 
                                       places=5)
                d2ydzeta2_fd = finite_difference_derivative(this_zeta,\
                    lambda zeta : self.vmecInput.position_first_derivatives(this_theta,zeta)[3], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2ydzeta2_fd, d2ydzeta2[izeta,itheta], 
                                       places=5)
                d2ydthetadzeta_fd = finite_difference_derivative(this_theta,\
                    lambda theta : self.vmecInput.position_first_derivatives(theta,this_zeta)[3], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2ydthetadzeta_fd, d2ydthetadzeta[izeta,itheta], 
                                       places=5)
                d2zdtheta2_fd = finite_difference_derivative(this_theta,\
                    lambda theta : self.vmecInput.position_first_derivatives(theta,this_zeta)[4], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2zdtheta2_fd, d2zdtheta2[izeta,itheta], 
                                       places=5)
                d2zdzeta2_fd = finite_difference_derivative(this_zeta,\
                    lambda zeta : self.vmecInput.position_first_derivatives(this_theta,zeta)[5], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2zdzeta2_fd, d2zdzeta2[izeta,itheta], 
                                       places=5)     
                d2zdthetadzeta_fd = finite_difference_derivative(this_theta,\
                    lambda theta : self.vmecInput.position_first_derivatives(theta,this_zeta)[5], \
                     epsilon=1e-5 ,method='centered')
                self.assertAlmostEqual(d2zdthetadzeta_fd, d2zdthetadzeta[izeta,itheta], 
                                       places=5)     

    def test_position_second_derivatives_surface(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Check that shape of ouput matches input.
        Performs finite difference testing of derivatives.
        """ 
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        
        # Check first 10 modes
        xm_sensitivity = self.vmecInput.xm[0:10]
        xn_sensitivity = self.vmecInput.xn[0:10]
        
        self.assertRaises(ValueError, \
                          self.vmecInput.position_second_derivatives_surface, 
                          xm_sensitivity, xn_sensitivity, \
                          theta = thetas, zeta = zetas)
        [d3Xdtheta2drmnc, d3Ydtheta2drmnc, d3Zdtheta2dzmns, \
               d3Xdzeta2drmnc, d3Ydzeta2drmnc, d3Zdzeta2dzmns, \
               d3Xdthetadzetadrmnc, d3Ydthetadzetadrmnc, \
               d3Zdthetadzetadzmns] = \
                self.vmecInput.position_second_derivatives_surface(xm_sensitivity, \
                                                           xn_sensitivity)

        def temp_fun_rbc(im, rbc):
            rbc_init = self.vmecInput.rbc[im]
            self.vmecInput.rbc[im] = rbc
            [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, \
                d2ydthetadzeta, d2zdtheta2, d2zdzeta2, d2zdthetadzeta, \
               d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta] \
                = self.vmecInput.position_second_derivatives()
            self.vmecInput.rbc[im] = rbc_init
            return d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, \
                d2ydthetadzeta, d2zdtheta2, d2zdzeta2, d2zdthetadzeta
        
        def temp_fun_zbs(im, zbs):
            zbs_init = self.vmecInput.zbs[im]
            self.vmecInput.zbs[im] = zbs
            [d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, \
                d2ydthetadzeta, d2zdtheta2, d2zdzeta2, d2zdthetadzeta, \
               d2Rdtheta2, d2Rdzeta2, d2Rdthetadzeta] \
                = self.vmecInput.position_second_derivatives()
            self.vmecInput.zbs[im] = zbs_init
            return d2xdtheta2, d2xdzeta2, d2xdthetadzeta, d2ydtheta2, d2ydzeta2, \
                d2ydthetadzeta, d2zdtheta2, d2zdzeta2, d2zdthetadzeta
        
        for im in range(len(xm_sensitivity)):
            this_rbc = self.vmecInput.rbc[im]
            this_zbs = self.vmecInput.zbs[im]
            d3Xdtheta2drmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[0], epsilon=1e-5, \
                method='centered')
            d3Ydtheta2drmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[3], epsilon=1e-5, \
                method='centered')
            d3Zdtheta2dzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs)[6], epsilon=1e-5, \
                method='centered')
            d3Xdzeta2drmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[1], epsilon=1e-5, \
                method='centered')
            d3Ydzeta2drmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[4], epsilon=1e-5, \
                method='centered')
            d3Zdzeta2dzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs)[7], epsilon=1e-5, \
                method='centered')
            d3Xdthetadzetadrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[2], epsilon=1e-5, \
                method='centered')
            d3Ydthetadzetadrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[5], epsilon=1e-5, \
                method='centered')
            d3Zdthetadzetadzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs)[8], epsilon=1e-5, \
                method='centered')
            for izeta in range(self.vmecInput.nzeta):
                for itheta in range(self.vmecInput.ntheta):
                    self.assertAlmostEqual(d3Xdtheta2drmnc_fd[izeta, itheta], \
                        d3Xdtheta2drmnc[im,izeta,itheta], places=4)
                    self.assertAlmostEqual(d3Ydtheta2drmnc_fd[izeta, itheta], \
                        d3Ydtheta2drmnc[im,izeta,itheta], places=4)     
                    self.assertAlmostEqual(d3Zdtheta2dzmns_fd[izeta, itheta], \
                        d3Zdtheta2dzmns[im,izeta,itheta], places=4)        
                    self.assertAlmostEqual(d3Xdzeta2drmnc_fd[izeta, itheta], \
                        d3Xdzeta2drmnc[im,izeta,itheta], places=4)     
                    self.assertAlmostEqual(d3Ydzeta2drmnc_fd[izeta, itheta], \
                        d3Ydzeta2drmnc[im,izeta,itheta], places=4)  
                    self.assertAlmostEqual(d3Zdzeta2dzmns_fd[izeta, itheta], \
                        d3Zdzeta2dzmns[im,izeta,itheta], places=4)
                    self.assertAlmostEqual(d3Xdthetadzetadrmnc_fd[izeta, itheta], \
                        d3Xdthetadzetadrmnc[im,izeta,itheta], places=4) 
                    self.assertAlmostEqual(d3Ydthetadzetadrmnc_fd[izeta, itheta], \
                        d3Ydthetadzetadrmnc[im,izeta,itheta], places=4) 
                    self.assertAlmostEqual(d3Zdthetadzetadzmns_fd[izeta, itheta], \
                        d3Zdthetadzetadzmns[im,izeta,itheta], places=4) 

    
    def test_jacobian(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Check that shape of ouput matches input.
        """
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        self.assertRaises(ValueError, self.vmecInput.jacobian, \
            theta = thetas, zeta = zetas)
        norm_normal = self.vmecInput.jacobian()
        self.assertTrue(np.all(np.shape(thetas) == np.shape(norm_normal)))
        
    def test_jacobian_derivatives(self):
        """
        Performs finite difference testing of derivatives.
        """ 
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        
        # Check first 10 modes
        xm_sensitivity = self.vmecInput.xm[0:10]
        xn_sensitivity = self.vmecInput.xn[0:10]
        
        [djacobiandrmnc, djacobiandzmns] = \
                self.vmecInput.jacobian_derivatives(xm_sensitivity, \
                                                           xn_sensitivity)

        def temp_fun_rbc(im, rbc):
            rbc_init = self.vmecInput.rbc[im]
            self.vmecInput.rbc[im] = rbc
            jacobian = self.vmecInput.jacobian()
            self.vmecInput.rbc[im] = rbc_init
            return jacobian
        
        def temp_fun_zbs(im, zbs):
            zbs_init = self.vmecInput.zbs[im]
            self.vmecInput.zbs[im] = zbs
            jacobian = self.vmecInput.jacobian()
            self.vmecInput.zbs[im] = zbs_init
            return jacobian
        
        for im in range(len(xm_sensitivity)):
            this_rbc = self.vmecInput.rbc[im]
            this_zbs = self.vmecInput.zbs[im]
            djacobiandrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc), epsilon=1e-5, \
                method='centered')
            djacobiandzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs), epsilon=1e-5, \
                method='centered')
            for izeta in range(self.vmecInput.nzeta):
                for itheta in range(self.vmecInput.ntheta):
                    self.assertAlmostEqual(djacobiandrmnc_fd[izeta, itheta], \
                        djacobiandrmnc[im,izeta,itheta], places=4)
                    self.assertAlmostEqual(djacobiandzmns_fd[izeta, itheta], \
                        djacobiandzmns[im,izeta,itheta], places=4) 
                    
    def test_normal_derivatives(self):
        """
        Performs finite difference testing of derivatives.
        """ 
        
        # Check first 10 modes
        xm_sensitivity = self.vmecInput.xm[0:10]
        xn_sensitivity = self.vmecInput.xn[0:10]
        
        [dNxdrmnc, dNxdzmns, dNydrmnc, dNydzmns, dNzdrmnc] = \
            self.vmecInput.normal_derivatives(xm_sensitivity,xn_sensitivity)

        def temp_fun_rbc(im, rbc):
            rbc_init = self.vmecInput.rbc[im]
            self.vmecInput.rbc[im] = rbc
            Nx, Ny, Nz = self.vmecInput.normal()
            self.vmecInput.rbc[im] = rbc_init
            return Nx, Ny, Nz 
        
        def temp_fun_zbs(im, zbs):
            zbs_init = self.vmecInput.zbs[im]
            self.vmecInput.zbs[im] = zbs
            Nx, Ny, Nz = self.vmecInput.normal()
            self.vmecInput.zbs[im] = zbs_init
            return Nx, Ny, Nz
        
        for im in range(len(xm_sensitivity)):
            this_rbc = self.vmecInput.rbc[im]
            this_zbs = self.vmecInput.zbs[im]
            dNxdrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[0], epsilon=1e-5, \
                method='centered')
            dNxdzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs)[0], epsilon=1e-5, \
                method='centered')
            dNydrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[1], epsilon=1e-5, \
                method='centered')
            dNydzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs)[1], epsilon=1e-5, \
                method='centered')
            dNzdrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[2], epsilon=1e-5, \
                method='centered')
            for izeta in range(self.vmecInput.nzeta):
                for itheta in range(self.vmecInput.ntheta):
                    self.assertAlmostEqual(dNxdrmnc_fd[izeta, itheta], \
                        dNxdrmnc[im,izeta,itheta], places=5)
                    self.assertAlmostEqual(dNxdzmns_fd[izeta, itheta], \
                        dNxdzmns[im,izeta,itheta], places=5)     
                    self.assertAlmostEqual(dNydrmnc_fd[izeta, itheta], \
                        dNydrmnc[im,izeta,itheta], places=5)   
                    self.assertAlmostEqual(dNydzmns_fd[izeta, itheta], \
                        dNydzmns[im,izeta,itheta], places=5)     
                    self.assertAlmostEqual(dNzdrmnc_fd[izeta, itheta], \
                        dNzdrmnc[im,izeta,itheta], places=5) 
                    
    def test_suface_curvature_derivatives(self):
        """
        Performs finite difference testing of derivatives.
        """ 
        
        # Check first 10 modes
        xm_sensitivity = self.vmecInput.xm[0:10]
        xn_sensitivity = self.vmecInput.xn[0:10]
        epsilon = 1e-6
        
        [dHdrmnc, dHdzmns, dKdrmnc, dKdzmns, dkappa1drmnc, dkappa1dzmns, \
            dkappa2drmnc, dkappa2dzmns] = \
            self.vmecInput.surface_curvature_derivatives(xm_sensitivity,xn_sensitivity)

        def temp_fun_rbc(im, rbc):
            rbc_init = self.vmecInput.rbc[im]
            self.vmecInput.rbc[im] = rbc
            H, K, kappa1, kappa2 = self.vmecInput.surface_curvature()
            self.vmecInput.rbc[im] = rbc_init
            return H, K, kappa1, kappa2
        
        def temp_fun_zbs(im, zbs):
            zbs_init = self.vmecInput.zbs[im]
            self.vmecInput.zbs[im] = zbs
            H, K, kappa1, kappa2 = self.vmecInput.surface_curvature()
            self.vmecInput.zbs[im] = zbs_init
            return H, K, kappa1, kappa2
        
        for im in range(len(xm_sensitivity)):
            this_rbc = self.vmecInput.rbc[im]
            this_zbs = self.vmecInput.zbs[im]
            dHdrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[0], epsilon=epsilon, \
                method='centered')
            dHdzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs)[0], epsilon=epsilon, \
                method='centered')
            dKdrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[1], epsilon=epsilon, \
                method='centered')
            dKdzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs)[1], epsilon=epsilon, \
                method='centered')
            dkappa1drmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[2], epsilon=epsilon, \
                method='centered')
            dkappa1dzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs)[2], epsilon=epsilon, \
                method='centered')
            dkappa2drmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[3], epsilon=epsilon, \
                method='centered')
            dkappa2dzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs)[3], epsilon=epsilon, \
                method='centered')
            for izeta in range(self.vmecInput.nzeta):
                for itheta in range(self.vmecInput.ntheta):
                    self.assertAlmostEqual(dHdrmnc_fd[izeta, itheta], \
                        dHdrmnc[im,izeta,itheta], places=5)
                    self.assertAlmostEqual(dHdzmns_fd[izeta, itheta], \
                        dHdzmns[im,izeta,itheta], places=5)     
                    self.assertAlmostEqual(dKdrmnc_fd[izeta, itheta], \
                        dKdrmnc[im,izeta,itheta], places=5)   
                    self.assertAlmostEqual(dKdzmns_fd[izeta, itheta], \
                        dKdzmns[im,izeta,itheta], places=5)     
                    self.assertAlmostEqual(dkappa1drmnc_fd[izeta, itheta], \
                        dkappa1drmnc[im,izeta,itheta], places=5) 
                    self.assertAlmostEqual(dkappa1dzmns_fd[izeta, itheta], \
                        dkappa1dzmns[im,izeta,itheta], places=5) 
                    self.assertAlmostEqual(dkappa2drmnc_fd[izeta, itheta], \
                        dkappa2drmnc[im,izeta,itheta], places=5)                     
                    self.assertAlmostEqual(dkappa2dzmns_fd[izeta, itheta], \
                        dkappa2dzmns[im,izeta,itheta], places=5) 
                    
    def test_normalized_jacobian(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Check that shape of ouput matches input.
        Checks that normalized jacobian integrates to 1
        """
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        self.assertRaises(ValueError, self.vmecInput.normalized_jacobian, \
            theta = thetas, zeta = zetas)
        normalized_jacobian = self.vmecInput.normalized_jacobian()
        self.assertTrue(np.all(np.shape(thetas) == \
                               np.shape(normalized_jacobian)))

        normalized_jacobian_sum = np.sum(normalized_jacobian) \
            / np.sum(np.size(normalized_jacobian))
        self.assertAlmostEqual(normalized_jacobian_sum, 1)
        
    def test_normalized_jacobian_derivatives(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Check that shape of ouput matches input.
        Performs finite difference testing of derivatives.
        """ 
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
        
        # Check first 10 modes
        xm_sensitivity = self.vmecInput.xm[0:10]
        xn_sensitivity = self.vmecInput.xn[0:10]
        
        self.assertRaises(ValueError, \
                          self.vmecInput.normalized_jacobian_derivatives, 
                          xm_sensitivity, xn_sensitivity, \
                          theta = thetas, zeta = zetas)
        [dnormalized_jacobiandrbc, dnormalized_jacobiandzbs] = \
            self.vmecInput.normalized_jacobian_derivatives(xm_sensitivity, \
                                                           xn_sensitivity)

        self.assertTrue(np.all(np.shape(thetas) == \
                          np.shape(np.squeeze(dnormalized_jacobiandrbc[1,:,:]))))
        self.assertTrue(np.all(np.shape(thetas) == \
                          np.shape(np.squeeze(dnormalized_jacobiandzbs[1,:,:]))))

        def temp_fun_rbc(im, rbc):
            rbc_init = self.vmecInput.rbc[im]
            self.vmecInput.rbc[im] = rbc
            normalized_jacobian = self.vmecInput.normalized_jacobian()
            self.vmecInput.rbc[im] = rbc_init
            return normalized_jacobian
        
        def temp_fun_zbs(im, zbs):
            zbs_init = self.vmecInput.zbs[im]
            self.vmecInput.zbs[im] = zbs
            normalized_jacobian = self.vmecInput.normalized_jacobian()
            self.vmecInput.zbs[im] = zbs_init
            return normalized_jacobian
        
        for im in range(len(xm_sensitivity)):
            this_rbc = self.vmecInput.rbc[im]
            this_zbs = self.vmecInput.zbs[im]
            dnormalized_jacobiandrbc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc), epsilon=1e-5, \
                method='centered')
            dnormalized_jacobiandzbs_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs), epsilon=1e-5, \
                method='centered')
            for izeta in range(self.vmecInput.nzeta):
                for itheta in range(self.vmecInput.ntheta):
                    self.assertAlmostEqual(dnormalized_jacobiandrbc_fd[izeta, itheta], \
                        dnormalized_jacobiandrbc[im,izeta,itheta], places=5)
                    self.assertAlmostEqual(dnormalized_jacobiandzbs_fd[izeta, itheta], \
                        dnormalized_jacobiandzbs[im,izeta,itheta], places=5)
                    
    def test_position_derivatives_surface(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Check that shape of ouput matches input.
        Performs finite difference testing of derivatives.
        """ 
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)

        # Check first 10 modes
        xm_sensitivity = self.vmecInput.xm[0:10]
        xn_sensitivity = self.vmecInput.xn[0:10]

        self.assertRaises(ValueError, \
                          self.vmecInput.position_derivatives_surface, 
                          xm_sensitivity, xn_sensitivity, \
                          theta = thetas, zeta = zetas)
        [dXdrmnc, dYdrmnc, dZdzmns, dRdrmnc] = \
            self.vmecInput.position_derivatives_surface(xm_sensitivity, \
                                                           xn_sensitivity)

        self.assertTrue(np.all(np.shape(thetas) == \
                          np.shape(np.squeeze(dXdrmnc[1,:,:]))))
        self.assertTrue(np.all(np.shape(thetas) == \
                          np.shape(np.squeeze(dYdrmnc[1,:,:]))))
        self.assertTrue(np.all(np.shape(thetas) == \
                          np.shape(np.squeeze(dZdzmns[1,:,:]))))
        self.assertTrue(np.all(np.shape(thetas) == \
                          np.shape(np.squeeze(dRdrmnc[1,:,:]))))

        def temp_fun_rbc(im, rbc):
            rbc_init = self.vmecInput.rbc[im]
            self.vmecInput.rbc[im] = rbc
            [X, Y, Z, R] = self.vmecInput.position()
            self.vmecInput.rbc[im] = rbc_init
            return X, Y, Z, R

        def temp_fun_zbs(im, zbs):
            zbs_init = self.vmecInput.zbs[im]
            self.vmecInput.zbs[im] = zbs
            [X, Y, Z, R] = self.vmecInput.position()
            self.vmecInput.zbs[im] = zbs_init
            return X, Y, Z, R

        for im in range(len(xm_sensitivity)):
            this_rbc = self.vmecInput.rbc[im]
            this_zbs = self.vmecInput.zbs[im]
            dXdrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[0], epsilon=1e-5, \
                method='centered')
            dYdrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[1], epsilon=1e-5, \
                method='centered')
            dRdrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[3], epsilon=1e-5, \
                method='centered')
            dZdzmns_fd = finite_difference_derivative(this_rbc,\
                lambda zbs : temp_fun_zbs(im,zbs)[2], epsilon=1e-5, \
                method='centered')

            for izeta in range(self.vmecInput.nzeta):
                for itheta in range(self.vmecInput.ntheta):
                    self.assertAlmostEqual(dXdrmnc_fd[izeta, itheta], \
                        dXdrmnc[im,izeta,itheta], places=5)
                    self.assertAlmostEqual(dYdrmnc_fd[izeta, itheta], \
                        dYdrmnc[im,izeta,itheta], places=5)
                    self.assertAlmostEqual(dRdrmnc_fd[izeta, itheta], \
                        dRdrmnc[im,izeta,itheta], places=5)
                    self.assertAlmostEqual(dZdzmns_fd[izeta, itheta], \
                        dZdzmns[im,izeta,itheta], places=5)
                    
    def test_radius_derivatives(self):
        """
        Performs finite difference testing of derivatives.
        """ 
        # Check first 10 modes
        xm_sensitivity = self.vmecInput.xm[0:10]
        xn_sensitivity = self.vmecInput.xn[0:10]

        [dRdrmnc, dRdzmns] = \
            self.vmecInput.radius_derivatives(xm_sensitivity, xn_sensitivity)

        def temp_fun_rbc(im, rbc):
            rbc_init = self.vmecInput.rbc[im]
            self.vmecInput.rbc[im] = rbc
            [X, Y, Z, R] = self.vmecInput.position()
            self.vmecInput.rbc[im] = rbc_init
            return R

        def temp_fun_zbs(im, zbs):
            zbs_init = self.vmecInput.zbs[im]
            self.vmecInput.zbs[im] = zbs
            [X, Y, Z, R] = self.vmecInput.position()
            self.vmecInput.zbs[im] = zbs_init
            return R

        for im in range(len(xm_sensitivity)):
            this_rbc = self.vmecInput.rbc[im]
            this_zbs = self.vmecInput.zbs[im]
            dRdrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc), epsilon=1e-5, \
                method='centered')

            for izeta in range(self.vmecInput.nzeta):
                for itheta in range(self.vmecInput.ntheta):
                    self.assertAlmostEqual(dRdrmnc_fd[izeta, itheta], \
                        dRdrmnc[im,izeta,itheta], places=5)
            
    def test_summed_radius_constraint_derivatives(self):
        """
        Performs finite difference testing of derivatives.
        """ 
        # Check first 10 modes
        xm_sensitivity = self.vmecInput.xm[0:10]
        xn_sensitivity = self.vmecInput.xn[0:10]
        
        [dconstraintdrmnc, dconstraintdzmns] = self.vmecInput.summed_radius_constraint_derivatives(\
                                                xm_sensitivity,xn_sensitivity,exp_weight=1e3)

        def temp_fun_rbc(im, rbc):
            rbc_init = self.vmecInput.rbc[im]
            self.vmecInput.rbc[im] = rbc
            constraint = self.vmecInput.summed_radius_constraint(exp_weight=1e3)
            self.vmecInput.rbc[im] = rbc_init
            return constraint

        def temp_fun_zbs(im, zbs):
            zbs_init = self.vmecInput.zbs[im]
            self.vmecInput.zbs[im] = zbs
            constraint = self.vmecInput.summed_radius_constraint(exp_weight=1e3)
            self.vmecInput.zbs[im] = zbs_init
            return constraint

        for im in range(len(xm_sensitivity)):
            this_rbc = self.vmecInput.rbc[im]
            this_zbs = self.vmecInput.zbs[im]
            dconstraintdrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc), epsilon=1e-8, \
                method='centered')
            dconstraintdzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs), epsilon=1e-8, \
                method='centered')
            self.assertAlmostEqual(dconstraintdrmnc_fd, dconstraintdrmnc[im], places=5)
            self.assertAlmostEqual(dconstraintdzmns_fd, dconstraintdzmns[im], places=5)

    def test_summed_proximity_derivatives(self):
        # Check first 10 modes
        xm_sensitivity = self.vmecInput.xm[0:10]
        xn_sensitivity = self.vmecInput.xn[0:10]
        
        [dproximitydrmnc, dproximitydzmns] = self.vmecInput.summed_proximity_derivatives(\
                                                xm_sensitivity,xn_sensitivity,exp_weight=1e3)
        
        def temp_fun_rbc(im, rbc):
            rbc_init = self.vmecInput.rbc[im]
            self.vmecInput.rbc[im] = rbc
            proximity = self.vmecInput.summed_proximity(exp_weight=1e3)
            self.vmecInput.rbc[im] = rbc_init
            return proximity

        def temp_fun_zbs(im, zbs):
            zbs_init = self.vmecInput.zbs[im]
            self.vmecInput.zbs[im] = zbs
            proximity = self.vmecInput.summed_proximity(exp_weight=1e3)
            self.vmecInput.zbs[im] = zbs_init
            return proximity

        for im in range(len(xm_sensitivity)):
            this_rbc = self.vmecInput.rbc[im]
            this_zbs = self.vmecInput.zbs[im]
            dproximitydrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc), epsilon=1e-8, \
                method='centered')
            dproximitydzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs), epsilon=1e-8, \
                method='centered')
            self.assertAlmostEqual(dproximitydrmnc_fd, dproximitydrmnc[im], places=5)
            self.assertAlmostEqual(dproximitydzmns_fd, dproximitydzmns[im], places=5) 
            
    def test_surface_curvature_metric_derivatives(self):
        # Check first 10 modes
        xm_sensitivity = self.vmecInput.xm[0:10]
        xn_sensitivity = self.vmecInput.xn[0:10]
        epsilon = 1e-5
        
        [dmetric1drmnc, dmetric1dzmns, dmetric2drmnc, dmetric2dzmns] = \
                self.vmecInput.surface_curvature_metric_integrated_derivatives(\
                                    xm_sensitivity, xn_sensitivity)
        
        def temp_fun_rbc(im, rbc):
            rbc_init = self.vmecInput.rbc[im]
            self.vmecInput.rbc[im] = rbc
            metric = self.vmecInput.surface_curvature_metric_integrated()
            self.vmecInput.rbc[im] = rbc_init
            return metric

        def temp_fun_zbs(im, zbs):
            zbs_init = self.vmecInput.zbs[im]
            self.vmecInput.zbs[im] = zbs
            metric = self.vmecInput.surface_curvature_metric_integrated()
            self.vmecInput.zbs[im] = zbs_init
            return metric

        for im in range(len(xm_sensitivity)):
            this_rbc = self.vmecInput.rbc[im]
            this_zbs = self.vmecInput.zbs[im]
            dmetric1drmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[0], epsilon=epsilon, \
                method='centered')
            dmetric1dzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs)[0], epsilon=epsilon, \
                method='centered')
            dmetric2drmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc)[1], epsilon=epsilon, \
                method='centered')
            dmetric2dzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs)[1], epsilon=epsilon, \
                method='centered')
            self.assertAlmostEqual(dmetric1drmnc_fd, dmetric1drmnc[im], places=5)
            self.assertAlmostEqual(dmetric2dzmns_fd, dmetric2dzmns[im], places=5)     
            
    def test_surface_curvature_metric_derivatives(self):
        # Check first 10 modes
        xm_sensitivity = self.vmecInput.xm[0:10]
        xn_sensitivity = self.vmecInput.xn[0:10]
        epsilon = 1e-5
        
        [dmetricdrmnc, dmetricdzmns] = self.vmecInput.surface_curvature_metric_derivatives(\
                                                xm_sensitivity,xn_sensitivity,exp_weight=1e4)
        
        def temp_fun_rbc(im, rbc):
            rbc_init = self.vmecInput.rbc[im]
            self.vmecInput.rbc[im] = rbc
            metric = self.vmecInput.surface_curvature_metric(exp_weight=1e4)
            self.vmecInput.rbc[im] = rbc_init
            return metric

        def temp_fun_zbs(im, zbs):
            zbs_init = self.vmecInput.zbs[im]
            self.vmecInput.zbs[im] = zbs
            metric = self.vmecInput.surface_curvature_metric(exp_weight=1e4)
            self.vmecInput.zbs[im] = zbs_init
            return metric

        for im in range(len(xm_sensitivity)):
            this_rbc = self.vmecInput.rbc[im]
            this_zbs = self.vmecInput.zbs[im]
            dmetricdrmnc_fd = finite_difference_derivative(this_rbc,\
                lambda rbc : temp_fun_rbc(im,rbc), epsilon=epsilon, \
                method='centered')
            dmetricdzmns_fd = finite_difference_derivative(this_zbs,\
                lambda zbs : temp_fun_zbs(im,zbs), epsilon=epsilon, \
                method='centered')
            self.assertAlmostEqual(dmetricdrmnc_fd, dmetricdrmnc[im], places=5)
            self.assertAlmostEqual(dmetricdzmns_fd, dmetricdzmns[im], places=5)     
                    
    def test_position(self):
        """
        Check that ValueError is raised if incorrect shapes are passed.
        Check that shape of ouput matches input.
        Test that X and Y are consistent with R and that Z averages to 0
        """         
        thetas = self.vmecInput.thetas_2d
        zetas = self.vmecInput.zetas_2d
        zetas = np.delete(zetas,0,0)
                
        self.assertRaises(ValueError, \
                          self.vmecInput.position, 
                          theta = thetas, zeta = zetas)
        
        thetas = self.vmecInput.thetas_2d_full
        zetas = self.vmecInput.zetas_2d_full

        [X, Y, Z, R] = self.vmecInput.position(theta = thetas,zeta = zetas)
        self.assertTrue(np.all(np.shape(X) == np.shape(thetas)))
        self.assertTrue(np.all(np.shape(Y) == np.shape(thetas)))
        self.assertTrue(np.all(np.shape(Z) == np.shape(thetas)))
        self.assertTrue(np.all(np.shape(R) == np.shape(thetas)))

        Z_int = np.sum(Z) * self.vmecInput.dzeta * self.vmecInput.dtheta 
        self.assertAlmostEqual(Z_int, 0, places=5)
        
        for itheta in range(self.vmecInput.ntheta):
            for izeta in range(self.vmecInput.nzeta):
                self.assertAlmostEqual(X[izeta,itheta] ** 2 + \
                                       Y[izeta,itheta] ** 2, \
                                       R[izeta,itheta] ** 2, places = 10)        
        
if __name__ == '__main__':
    unittest.main()