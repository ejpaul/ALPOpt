import unittest
import sys
import numpy as np
import numbers
import scipy.optimize

sys.path.append('..')

from finite_differences import finite_difference_derivative_random, \
    finite_difference_derivative


class Test(unittest.TestCase):
    
    def test_finite_difference_derivative_random(self):
        """
        Check that ValueError is raised if incorrect method if passed
        Evaluate derivative of rosenbrock function with centered difference
        Compare with analytic derivative
        Perform same comparison with forward difference
        Evaluate derivative of vector-valued function (f[x]=x*x)
        Check that call with specified unitvec returns same derivative
        """
        x = 2*np.ones(10)
        self.assertRaises(ValueError,finite_difference_derivative_random,\
                x,scipy.optimize.rosen,epsilon=1e-5,method='centeredd')
        dfdx, unitvec = finite_difference_derivative_random(x,scipy.optimize.rosen,
                                                  epsilon=1e-5,method='centered')
        dfdx_analytic = np.vdot(unitvec,scipy.optimize.rosen_der(x))
        self.assertAlmostEqual(dfdx_analytic,dfdx,places=5)
        dfdx, unitvec = finite_difference_derivative_random(x,scipy.optimize.rosen,
                                                  epsilon=1e-6,method='forward')
        dfdx_analytic = np.vdot(unitvec,scipy.optimize.rosen_der(x))
        self.assertAlmostEqual(dfdx_analytic,dfdx,places=2)
        
        dfdx, unitvec = finite_difference_derivative_random(x, lambda x : x * x,
                                                  epsilon=1e-5,method='centered')
        dfdx_analytic = np.dot(2 * np.diag(x),unitvec)
        self.assertTrue(np.allclose(dfdx,dfdx_analytic,atol=1e-5))
        dfdx_random, unitvec = finite_difference_derivative_random(x, lambda x : x * x,
                                epsilon=1e-5,method='centered',unitvec=unitvec)
        self.assertTrue(np.allclose(dfdx,dfdx_random,atol=1e-5))

    def test_finite_difference_derivative(self):    
        """
        Check that ValueError is raised if incorrect method if passed
        Evaluate derivative of rosenbrock function with centered difference
        Compare with analytic derivative
        Check that random derivative is matched in unitvec direction
        Perform comparison with analytic derivatives using forward difference
        Evaluate derivative of vector-valued function (f[x]=x*x)
        """
        x = 2*np.ones(10)
        self.assertRaises(ValueError,finite_difference_derivative,\
                x,scipy.optimize.rosen,epsilon=1e-5,method='centeredd')
        dfdx = finite_difference_derivative(x,scipy.optimize.rosen,
                                                  epsilon=1e-5,method='centered')
        dfdx_analytic = scipy.optimize.rosen_der(x)
        self.assertTrue(np.allclose(dfdx_analytic,dfdx,atol=1e-5))
        
        dfdx_random, unitvec = finite_difference_derivative_random(x,scipy.optimize.rosen,
                                                 epsilon=1e-5,method='centered')
        self.assertAlmostEqual(dfdx_random,np.dot(unitvec,dfdx),places=6)
        
        dfdx = finite_difference_derivative(x,scipy.optimize.rosen,
                                                    epsilon=1e-6,method='forward')
        self.assertTrue(np.allclose(dfdx_analytic,dfdx,atol=1e-3))
        
        dfdx = finite_difference_derivative(x, lambda x : x * x,
                                                 epsilon=1e-5,method='centered')
        dfdx_analytic = 2 * np.diag(x)
        self.assertTrue(np.allclose(dfdx_analytic,dfdx,atol=1e-5))
        
        