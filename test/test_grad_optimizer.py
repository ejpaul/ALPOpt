import unittest
import sys
import numpy as np
import numbers
import scipy.optimize

sys.path.append('..')

from grad_optimizer import GradOptimizer

class Test(unittest.TestCase):

    gradOptimizer = GradOptimizer(nparameters=10)
    
    def test_add_objective(self):
        """
        Add rosenbrock objective. 
        Check that nobjectives is incremented.
        Check that TypeError is raised if incorrect types are passed.
        """
        self.gradOptimizer.add_objective(scipy.optimize.rosen, \
                                         scipy.optimize.rosen_der,1)
        self.assertEqual(self.gradOptimizer.nobjectives, 1)
        self.assertRaises(TypeError,self.gradOptimizer.add_objective,1,\
                          scipy.optimize.rosen,3)
        self.assertRaises(TypeError,self.gradOptimizer.add_objective,\
                         scipy.optimize.rosen,2,3)
        self.assertRaises(TypeError,self.gradOptimizer.add_objective,\
                         scipy.optimize.rosen,scipy.optimize.rosen_der,'a')
        
    def test_add_bound(self):
        """
        Check that TypeError/ValueError is raised if incorrect args are passed.
        Add bound objective. Check that bound_constrained is set to true.
        Check that bound_constraints_min/max are of the correct length
        """
        self.assertRaises(TypeError,self.gradOptimizer.add_bound,\
                                       's','min')
        self.assertRaises(ValueError,self.gradOptimizer.add_bound,\
                         [1 for i in range(11)],'min')
        self.assertRaises(TypeError,self.gradOptimizer.add_bound,\
                         0,0)
        self.assertRaises(ValueError,self.gradOptimizer.add_bound,\
                         0,'minn')
        
        self.gradOptimizer.add_bound(0,'min')
        self.assertTrue(self.gradOptimizer.bound_constrained)
        self.assertEqual(len(self.gradOptimizer.bound_constraints_min),\
                        self.gradOptimizer.nparameters)
        self.assertEqual(len(self.gradOptimizer.bound_constraints_max),0)
        self.gradOptimizer.add_bound(1,'max')
        self.assertTrue(self.gradOptimizer.bound_constrained)
        self.assertEqual(len(self.gradOptimizer.bound_constraints_min),\
                        self.gradOptimizer.nparameters)
        self.assertEqual(len(self.gradOptimizer.bound_constraints_max),\
                        self.gradOptimizer.nparameters)
        
    def test_ineq(self):
        """
        Check that TypeError is raised if incorrect args are passed.
        Add ineq. constraint.
        Check that ineq_constrained), n_ineq_constraints, 
         self.gradOptimizer.ineq_constraints,
         self.gradOptimizer.ineq_constraints_grad are set appropriately
        """
        self.assertRaises(TypeError,self.gradOptimizer.add_ineq,\
                         1,scipy.optimize.rosen)
        self.assertRaises(TypeError,self.gradOptimizer.add_ineq,\
                         scipy.optimize.rosen,1)
        self.gradOptimizer.add_ineq(np.sin,np.cos)
        self.assertTrue(self.gradOptimizer.ineq_constrained)
        self.assertEqual(self.gradOptimizer.n_ineq_constraints,1)
        self.assertEqual(len(self.gradOptimizer.ineq_constraints),1)
        self.assertEqual(len(self.gradOptimizer.ineq_constraints_grad),1)
        
    def test_eq(self):
        """
        Check that TypeError is raised if incorrect args are passed.
        Add equality constraint
        Check that eq_constrained, n_eq_constraints, eq_constraints, 
         eq_constraints_grad are set appropriately
        """
        self.assertRaises(TypeError,self.gradOptimizer.add_eq,\
                         1,scipy.optimize.rosen)
        self.assertRaises(TypeError,self.gradOptimizer.add_eq,\
                         scipy.optimize.rosen,1)
        self.gradOptimizer.add_eq(np.sin,np.cos)
        self.assertTrue(self.gradOptimizer.eq_constrained)
        self.assertEqual(self.gradOptimizer.n_eq_constraints,1)
        self.assertEqual(len(self.gradOptimizer.eq_constraints),1)
        self.assertEqual(len(self.gradOptimizer.eq_constraints_grad),1)
        
    def test_objective_fun(self):
        """
        Check that ValueError is raised if incorrect length is passed
        Add cubic function
        Call objectives_fun
        Check that output is of correct type 
        Check that neval_objectives, objectives_hist, parameters_hist are
         set appropriately
        Call objectives_fun again
        Check that neval_objectives, objectives_hist, parameters_hist are
         set appropriately
        Add objective and check that RuntimeError is raised 
        Remove the additional objective
        """
        self.assertRaises(ValueError,self.gradOptimizer.objectives_fun,\
                          [1 for i in range(11)])
        # Add objective 
        self.gradOptimizer.add_objective(lambda x : np.sum(np.array(x) ** 3), \
                                         lambda x : 3 * np.array(x) ** 2, 1)        
        objective = self.gradOptimizer.objectives_fun([0 for i in range(10)])
        self.assertTrue(isinstance(objective,numbers.Number))
        
        self.assertEqual(self.gradOptimizer.neval_objectives,1)
        self.assertEqual(len(self.gradOptimizer.objectives_hist[:,0]),\
                         self.gradOptimizer.neval_objectives)
        self.assertEqual(len(self.gradOptimizer.objectives_hist[0,:]),\
                         self.gradOptimizer.nobjectives)
        self.assertEqual(len(self.gradOptimizer.objective_hist),1)
        self.assertEqual(len(self.gradOptimizer.parameters_hist[:,0]),\
                         self.gradOptimizer.neval_objectives)
        self.assertEqual(len(self.gradOptimizer.parameters_hist[0,:]),\
                        self.gradOptimizer.nparameters)
        self.assertTrue(np.allclose([0 for i in range(10)],\
                                    self.gradOptimizer.parameters_hist))
        # Call again
        self.gradOptimizer.objectives_fun([1 for i in range(10)])
        self.assertEqual(self.gradOptimizer.neval_objectives,2)
        self.assertEqual(len(self.gradOptimizer.objectives_hist[:,0]),\
                         self.gradOptimizer.neval_objectives)
        self.assertEqual(len(self.gradOptimizer.objectives_hist[0,:]),\
                         self.gradOptimizer.nobjectives)
        self.assertEqual(len(self.gradOptimizer.objective_hist),2)
        self.assertEqual(len(self.gradOptimizer.parameters_hist[:,0]),\
                        self.gradOptimizer.neval_objectives)
        self.assertEqual(len(self.gradOptimizer.parameters_hist[0,:]),\
                        self.gradOptimizer.nparameters)
        self.assertTrue(np.allclose([1 for i in range(10)],\
                                   self.gradOptimizer.parameters_hist[1,:]))
        # Add objective and check that RuntimeError is raised
        self.gradOptimizer.add_objective(lambda x : np.sum(np.array(x) ** 2), \
                                         lambda x : 2 * np.array(x), 2)
        self.assertRaises(RuntimeError,\
                          self.gradOptimizer.objectives_fun,\
                                                        [0 for i in range(10)])
        # Remove extra item
        self.gradOptimizer.objectives.pop()
        self.gradOptimizer.objectives_grad.pop()
        self.gradOptimizer.objective_weights.pop()
        self.gradOptimizer.nobjectives -= 1
                                             
    def test_objectives_grad_fun(self):
        """
        Check that ValueError is raised if incorrect length is passed
        Call objectives_grad_fun
        Check that output is of correct shape and type
        Check that neval_objectives_grad and objectives_grad_norm_hist are
            set appropriately
        Call objectives_grad_fun again
        Check that output is of correct shape and type
        Check that neval_objectives_grad and objectives_grad_norm_hist are
            set appropriately
        Check that gradients are equal to each other 
        """
        self.assertRaises(ValueError,self.gradOptimizer.objectives_grad_fun,\
                          [1 for i in range(11)])
        
        grad = self.gradOptimizer.objectives_grad_fun([0 for i in range(10)])
        self.assertTrue(isinstance(grad, np.ndarray))
        self.assertEqual(grad.ndim, 1)
        self.assertEqual(len(grad),self.gradOptimizer.nparameters)
        self.assertEqual(self.gradOptimizer.neval_objectives_grad,1)
        self.assertEqual(len(self.gradOptimizer.objectives_grad_norm_hist),1)
        grad2 = self.gradOptimizer.objectives_grad_fun([0 for i in range(10)])
        self.assertTrue(isinstance(grad2, np.ndarray))
        self.assertEqual(grad2.ndim, 1)
        self.assertEqual(len(grad2),self.gradOptimizer.nparameters)
        self.assertEqual(self.gradOptimizer.neval_objectives_grad,2)
        self.assertEqual(len(self.gradOptimizer.objectives_grad_norm_hist),2)
        self.assertTrue(np.all(grad2 == grad))
        
    def test_ineq_fun(self):
        """
        Check that ValueError is raised if incorrect length of input is passed
        Add inequality constraint with cubic function
        Call ineq_fun
        Check that output is of appropriate type
        Check that neval_ineq_constraints and ineq_constraints_hist are set
            appropriately
        Call ineq_fun again
        Check that output is of appropriate type
        Check that neval_ineq_constraints and ineq_constraints_hist are set
            appropriately
        Add another inequality constraint and check that RuntimeError is raised
        """
        self.assertRaises(ValueError,self.gradOptimizer.ineq_fun,\
                          [1 for i in range(11)])
        self.gradOptimizer.reset_all()
        # Add inequality constraint
        self.gradOptimizer.add_ineq(lambda x : np.sum(np.array(x) ** 3), \
                                         lambda x : 3 * np.array(x) **2)        
        ineq_value = self.gradOptimizer.ineq_fun([0 for i in range(10)])
        self.assertTrue(isinstance(ineq_value,np.ndarray))
        self.assertEqual(self.gradOptimizer.neval_ineq_constraints,1)
        self.assertEqual(len(self.gradOptimizer.ineq_constraints_hist[:,0]),\
                         self.gradOptimizer.neval_ineq_constraints)
        
        self.assertEqual(len(self.gradOptimizer.ineq_constraints_hist[0,:]),\
                         self.gradOptimizer.n_ineq_constraints)
        self.assertEqual(self.gradOptimizer.ineq_constraints_hist[0,:],\
                        ineq_value)

        # Call again
        ineq_value = self.gradOptimizer.ineq_fun([1 for i in range(10)])
        self.assertEqual(self.gradOptimizer.neval_ineq_constraints,2)
        self.assertEqual(len(self.gradOptimizer.ineq_constraints_hist[:,0]),\
                         self.gradOptimizer.neval_ineq_constraints)
        self.assertEqual(len(self.gradOptimizer.ineq_constraints_hist[0,:]),\
                         self.gradOptimizer.n_ineq_constraints)
        self.assertEqual(len(self.gradOptimizer.ineq_constraints_hist),2)
        self.assertEqual(self.gradOptimizer.ineq_constraints_hist[1,:],\
                         ineq_value)
        # Add constraint and check that RuntimeError is raised
        self.gradOptimizer.add_ineq(lambda x : np.sum(np.array(x) ** 2), \
                                         lambda x : 2 * np.array(x))
        self.assertRaises(RuntimeError,\
                          self.gradOptimizer.ineq_fun,[0 for i in range(10)])
        self.gradOptimizer.reset_all()
        
    def test_ineq_grad_fun(self):
        """
        Check that ValueError is raised if incorrect shape is passed
        Add inequality constraint with cubic function
        Call ineq_grad_fun
        Check for correct output type
        Check that neval_ineq_constraints_grad, ineq_constraints_grad_norm_his, 
         n_ineq_constraints are set appropriately
        """
        self.assertRaises(ValueError,self.gradOptimizer.ineq_grad_fun,\
                          [1 for i in range(11)])
        self.gradOptimizer.reset_all()
        # Add inequality constraint
        self.gradOptimizer.add_ineq(lambda x : np.sum(np.array(x) ** 3), \
                                         lambda x : 3 * np.array(x) **2)        
        ineq_grad = self.gradOptimizer.ineq_grad_fun([0 for i in range(10)])
        self.assertTrue(isinstance(ineq_grad,np.ndarray))
        self.assertEqual(self.gradOptimizer.neval_ineq_constraints_grad,1)
        self.assertEqual(len(self.gradOptimizer.ineq_constraints_grad_norm_hist[:,0]),\
                         self.gradOptimizer.neval_ineq_constraints_grad)
        
        self.assertEqual(len(self.gradOptimizer.ineq_constraints_grad_norm_hist[0,:]),\
                         self.gradOptimizer.n_ineq_constraints)
        self.assertEqual(self.gradOptimizer.ineq_constraints_grad_norm_hist[0,:],\
                        scipy.linalg.norm(ineq_grad))
        
    def test_optimize(self):
        """
        Check that ValueError is raised if incorrect shape is passed
        Check that TyepError is raised if incorrect type of ftolAbs, ftolRel,
            xtolAbs, xtolRel
        Check that ValueError is raised if incorrect packaged or algorithm
        Add rosenbrock function objective
        Optimize using nlopt TNEWTON
        Check that optimum matches expected value
        Perform same test with scipy CG
        Add rosenbrock function objective with bound constraint (>= 0)
        Optimize with nlopt SLSQP 
        Check that optimum matches expected value
        Perform same test with scipy L-BFGS-B
        Remove previous objectives, add linear objective with quadratic constraint
        Optimize using nlopt TNEWTON
        Check that optimum matches expected value
        Perform same test with nlopt trust-constr
        Remove previous objectives, add equality constraints: 
            min -x1*x2 s.t. sum(x) = 10 
        Optimize with nlopt SLSQP 
        Check that optimum matches expected value
        Optimize with scipy SLSQP, nlopt TNEWTON, and scipy trust-constr
        Add 2 inequality constraints 
            min x1 + x2 s.t. 2 - x1^2 - x2^2 >= 0 and x2 >= 0
        Optimize with nlopt SLSQP and scipy SLSQP
        Check that optimum matches expected value
        Add 2 equality constraints
            min x1 + x2 s.t. 2 - x1^2 - x2^2 = 0 and x2 = 0
        Check that optimum matches expected value
        """
        self.assertRaises(ValueError,self.gradOptimizer.optimize,\
                          [1 for i in range(11)])
        self.assertRaises(TypeError,self.gradOptimizer.optimize,\
                         [1 for i in range(10)],ftolAbs='s')
        self.assertRaises(TypeError,self.gradOptimizer.optimize,\
                         [1 for i in range(10)],ftolRel='s')
        self.assertRaises(TypeError,self.gradOptimizer.optimize,\
                         [1 for i in range(10)],xtolAbs='s')
        self.assertRaises(TypeError,self.gradOptimizer.optimize,\
                         [1 for i in range(10)],xtolRel='s')
        self.assertRaises(ValueError,self.gradOptimizer.optimize,\
                         [1 for i in range(10)],package='nloppt')
        # Check that ValueError raised if incorrect method for nlopt
        self.assertRaises(ValueError,self.gradOptimizer.optimize,\
                         [1 for i in range(10)],package='nlopt',\
                          method='MMAN')
        self.assertRaises(ValueError,self.gradOptimizer.optimize,\
                         [1 for i in range(10)],package='nlopt',\
                         method='CG')
        # Check that ValueError raised if incorrect method for scipy
        self.assertRaises(ValueError,self.gradOptimizer.optimize,\
                         [1 for i in range(10)],package='scipy',\
                          method='CGG')
        self.assertRaises(ValueError,self.gradOptimizer.optimize,\
                         [1 for i in range(10)],package='nlopt',\
                         method='BFGS')

        self.gradOptimizer.reset_all()
                
        # Test unconstrained Rosenbrock function
        self.gradOptimizer.add_objective(scipy.optimize.rosen, \
                                         scipy.optimize.rosen_der,1)
        x = 20*np.ones(10)
        # Test nlopt
        [xopt,fopt,result] = \
            self.gradOptimizer.optimize(x,package='nlopt',method='TNEWTON')
        self.assertAlmostEqual(fopt,0)
        self.assertTrue(np.allclose(xopt,1))
        # Test scipy
        [xopt,fopt,result] = \
            self.gradOptimizer.optimize(x,package='scipy',method='CG')
        self.assertAlmostEqual(fopt,0)
        self.assertTrue(np.allclose(xopt,1,atol=1e-3))

        
        # Test bound constrained Rosenbrock function
        self.gradOptimizer.reset_all()
        self.gradOptimizer.add_objective(scipy.optimize.rosen, \
                                         scipy.optimize.rosen_der,1)
        self.gradOptimizer.add_bound(0,'min')
        x = 2*np.ones(10)
        # Test nlopt
        [xopt,fopt,result] = \
            self.gradOptimizer.optimize(x,package='nlopt',method='SLSQP')
        self.assertAlmostEqual(fopt,0,places=5)
        self.assertTrue(np.allclose(xopt,1,atol=1e-2))
        # Test scipy
        [xopt,fopt,result] = \
            self.gradOptimizer.optimize(x,package='scipy',method='L-BFGS-B')
        self.assertAlmostEqual(fopt,0,places=5)
        self.assertTrue(np.allclose(xopt,1,atol=1e-2))
        
        # Test inequality constrained function
        # min sum(x) s.t. sum(x**2) < 2 
        self.gradOptimizer.reset_all()
        self.gradOptimizer.add_objective(lambda x : np.sum(x), \
                                        lambda x : np.ones(np.size(x)),1)
        self.gradOptimizer.add_ineq(lambda x : 2 - np.sum(x ** 2), \
                                   lambda x : - 2 * x)
        x = np.zeros(10)
        # Test nlopt
        [xopt,fopt,result] = \
            self.gradOptimizer.optimize(x,package='nlopt',method='SLSQP')
        self.assertAlmostEqual(fopt,-10*np.sqrt(1/5),places=8)
        self.assertTrue(np.allclose(xopt,-np.sqrt(1/5),atol=1e-6))
        # Test scipy
        [xopt,fopt,result] = \
            self.gradOptimizer.optimize(x,package='scipy',method='SLSQP')
        self.assertAlmostEqual(fopt,-10*np.sqrt(1/5),places=5)
        self.assertTrue(np.allclose(xopt,-np.sqrt(1/5),atol=1e-6))

        # Test inequality constraint
        self.gradOptimizer.reset_all()
        self.gradOptimizer.add_objective(lambda x : np.sum(x), \
                                        lambda x : np.ones(np.size(x)),1)
        self.gradOptimizer.add_ineq(lambda x : 2 - np.sum(x ** 2), \
                                   lambda x : - 2 * x)
        x = 0.25*np.ones(10)
        # Test nlopt
        [xopt,fopt,result] = \
            self.gradOptimizer.optimize(x,package='nlopt',method='TNEWTON')
        self.assertAlmostEqual(fopt,-10*np.sqrt(1/5),places=8)
        self.assertTrue(np.allclose(xopt,-np.sqrt(1/5),atol=1e-6))
        # Test scipy
        [xopt,fopt,result] = \
            self.gradOptimizer.optimize(x,package='scipy',method='trust-constr',\
                                       tol=1e-10)
        self.assertAlmostEqual(fopt,-10*np.sqrt(1/5),places=2)
        self.assertTrue(np.allclose(xopt,-np.sqrt(1/5),atol=1e-3))
        
        # Test equality constrained problem 
        # min -x1*x2 s.t. sum(x) = 10 
        self.gradOptimizer.reset_all()
        self.gradOptimizer.nparameters = 2
        self.gradOptimizer.add_objective(lambda x : -x[0]*x[1], \
                                        lambda x : -np.array([x[1],x[0]]), 1)
        self.gradOptimizer.add_eq(lambda x : np.sum(x) - 10, \
                                  lambda x : [1,1])
        x = [3,7] # Initial condition satisfies constraints
        # Test nlopt
        [xopt,fopt,result] = \
            self.gradOptimizer.optimize(x,package='nlopt',method='SLSQP')
        self.assertAlmostEqual(fopt,-25,places=8)
        self.assertTrue(np.allclose(xopt,5,atol=1e-5))
        # Test scipy
        [xopt,fopt,result] = \
            self.gradOptimizer.optimize(x,package='scipy',method='SLSQP')
        self.assertAlmostEqual(fopt,-25,places=8)
        self.assertTrue(np.allclose(xopt,5,atol=1e-5))
        
        # Test same problem with TNEWTON/trust-constr
        self.gradOptimizer.reset_all()
        self.gradOptimizer.nparameters = 2
        self.gradOptimizer.add_objective(lambda x : -x[0]*x[1], \
                                        lambda x : -np.array([x[1],x[0]]), 1)
        self.gradOptimizer.add_eq(lambda x : np.sum(x) - 10, \
                                  lambda x : [1,1])
        x = [3,7] # Initial condition satisfies constraints
        # Test nlopt
        [xopt,fopt,result] = \
            self.gradOptimizer.optimize(x,package='nlopt',method='TNEWTON')
        self.assertAlmostEqual(fopt,-25,places=8)
        self.assertTrue(np.allclose(xopt,5,atol=1e-5))
        # Test scipy
        [xopt,fopt,result] = \
            self.gradOptimizer.optimize(x,package='scipy',method='trust-constr')
        self.assertAlmostEqual(fopt,-25,places=7)
        self.assertTrue(np.allclose(xopt,5,atol=1e-4))
        
        # 2 inequality constraints 
        # min x1 + x2 s.t. 2 - x1^2 - x2^2 >= 0 and x2 >= 0
        self.gradOptimizer.reset_all()
        self.gradOptimizer.add_objective(lambda x : np.sum(x), \
                                         lambda x : np.ones(np.size(x)),1)
        self.gradOptimizer.add_ineq(lambda x : 2 - np.sum(x ** 2), \
                                    lambda x : - 2 * x)
        self.gradOptimizer.add_ineq(lambda x : x[1], lambda x : [0, 1])
        x = np.zeros(2)
        # Test nlopt
        [xopt,fopt,result] = \
            self.gradOptimizer.optimize(x,package='nlopt',method='SLSQP')
        self.assertAlmostEqual(fopt,-np.sqrt(2),places=8)
        self.assertAlmostEqual(xopt[0],-np.sqrt(2),places=8)
        self.assertAlmostEqual(xopt[1],0,places=8)
        # Test scipy
        [xopt,fopt,result] = \
            self.gradOptimizer.optimize(x,package='scipy',method='SLSQP')
        self.assertAlmostEqual(fopt,-np.sqrt(2),places=5)
        self.assertAlmostEqual(xopt[0],-np.sqrt(2),places=5)
        self.assertAlmostEqual(xopt[1],0,places=5)
        
        # 2 equality constraints
        # min x1 + x2 s.t. 2 - x1^2 - x2^2 = 0 and x2 = 0
        self.gradOptimizer.reset_all()
        self.gradOptimizer.nparameters = 3
        self.gradOptimizer.add_objective(lambda x : -x[0]*x[1], \
                                        lambda x : -np.array([x[1],x[0],0]), 1)
        self.gradOptimizer.add_eq(lambda x : np.sum(x) - 10, \
                                  lambda x : np.array([1,1,1]))
        self.gradOptimizer.add_eq(lambda x : x[2], lambda x : np.array([0,0,1]))
        x = [3,7,0] # Initial condition satisfies constraints
        # Test nlopt
        [xopt,fopt,result] = \
            self.gradOptimizer.optimize(x,package='nlopt',method='SLSQP')
        self.assertAlmostEqual(fopt,-25,places=8)
        self.assertAlmostEqual(xopt[0],5,places=8)
        self.assertAlmostEqual(xopt[1],5,places=8)
        self.assertAlmostEqual(xopt[2],0,places=8)
        # Test scipy
        [xopt,fopt,result] = \
            self.gradOptimizer.optimize(x,package='scipy',method='SLSQP')
        self.assertAlmostEqual(fopt,-25,places=8)
        self.assertAlmostEqual(xopt[0],5,places=8)
        self.assertAlmostEqual(xopt[1],5,places=8)
        self.assertAlmostEqual(xopt[2],0,places=8)        
        
