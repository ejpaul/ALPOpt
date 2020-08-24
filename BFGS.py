import scipy.linalg
import numpy as np

class BFGS:
    def __init__(self):
        # Hyperparameters
        self.epsilon = 1e-6 # tolerance in function gradient
        self.beta = 1 # Scaling for initial Hessian 
        self.alpha_max = 1 # Maximum step length
        self.alpha_scale = 0.1 # Scaling for initial step length
        self.alpha_tol = 1e-12 # 1-alpha >= alpha_tol
        self.rho_backtrack = 0.9 
        self.alpha0 = 1.0 # Initial step size for backtrack
        self.c1 = 1e-4
        self.c2 = 0.9
        self.test = True
        self.norm_grad_hist = []
        self.f_hist = []
        self.alpha_hist = []
        self.niter = 0
        
    def reset_hist(self):
        self.f_hist = []
        self.alpha_hist = []
        self.norm_grad_hist = []

    def BFGS_opt(self,x0,objective_fun,objective_grad_fun):
        """
        Args:
            x ()
            objective_fun ()
            objective_grad_fun ()
        """
        nparameters = len(x0)
        # Initialize Hessian as constant multiplied by identity
        identity = np.identity(nparameters)
        H = self.beta*identity
        # Compute initial function value and gradient
        grad_f0 = objective_grad_fun(x0)
        norm_grad_f0 = scipy.linalg.norm(grad_f0)
        f0 = objective_fun(x0)
        
        x = x0
        f = f0
        grad_f = grad_f0
        norm_grad_f = norm_grad_f0
        while (norm_grad_f > self.epsilon):
            self.niter += 1
            print("norm_grad_f : "+str(norm_grad_f))
            self.norm_grad_hist.append(norm_grad_f)
            self.f_hist.append(f)
            # Compute search direction
            p = - np.dot(H,grad_f)
            # Perform line search
            print('Performing line search')
            [alpha,f_new,grad_f_new] = self.line_search_wolfe(x,\
                                       objective_fun,objective_grad_fun,f,grad_f,p)
            self.alpha_hist.append(alpha)
            # Update Hessian matrix
            s = alpha*p
            y = grad_f_new - grad_f
            rho = 1/np.dot(y,s)
            matrixl = identity - rho*np.outer(s,y)
            matrixr = identity - rho*np.outer(y,s)
            H = np.dot(matrixl,np.dot(H,matrixr)) + rho*np.outer(s,s)
            
            # Update iterate
            x += s
            # Update gradient
            grad_f = grad_f_new
            norm_grad_f = scipy.linalg.norm(grad_f)
            # Update function value
            f = f_new
            
        print('Obtained optimum at: ' + str(x))
        print('Norm of gradient: ' + str(norm_grad_f))
        print('Function value: ' + str(f))
        return x
    
    def line_search_backtrack(self,x0,objective_fun,objective_grad_fun,f0,gradf0,p):
        """
        f0 = initial objective function value
        gradf0 = initial objective function gradient
        
        Algorith 3.1 from Nocedal & Wright
        
        """        
        # Initialize step size
        alpha = self.alpha0
        f_new = objective_fun(x0 + alpha*p)
        
        # Test Armijo condition
        while (f_new > f0 + self.c1 * alpha * np.dot(gradf0,p)):
            # Backtrack
            alpha *= self.rho_backtrack
            f_new = objective_fun(x0 + alpha*p)

        gradf_new = objective_grad_fun(x0 + alpha*p)
        return alpha, f_new, gradf_new        
            
    def line_search_wolfe(self,x0,objective_fun,objective_grad_fun,f0,gradf0,p):
        """
        f0 = initial objective function value
        gradf0 = initial objective function gradient
        
        Algorith 3.5-3.6 from Nocedal & Wright
        
        """
        # phi(alpha) = f(x + alpha * p)
        # phi'(alpha) = f'(x + alpha * p) * p
        phiprime0 = np.dot(gradf0,p)
        
        # Check that p is a descent direction
        if (self.test):
            assert(phiprime0 < 0)
        
        alpha_factor = 0.99
        # Previous step size
        alpha_old = 0
        # Current step size - choose alpha1 in (0,alpha_max)
        alpha = self.alpha_scale*self.alpha_max
        i = 1
        f_old = f0
        while (True):
            # Evaluate function
            x = x0 + alpha * p
            f = objective_fun(x)
            
            # Sufficient decrease condition (N&W 3.7a)
            cond1 = f > (f0 + self.c1 * alpha * phiprime0)
            cond2 = (f >= f_old) and (i > 1)
            if (cond1 or cond2):
                return self.zoom(alpha_old,alpha,x0,objective_fun,objective_grad_fun,f0,gradf0,p)
            gradf = objective_grad_fun(x)
            phiprime = np.dot(gradf,p)
            # Curvature condition (N&W 3.7b)
            if (np.abs(phiprime) <= - self.c2 * phiprime0):
                return alpha, f, gradf
            if (phiprime >= 0):
                return self.zoom(alpha,alpha_old,x0,objective_fun,objective_grad_fun,f0,gradf0,p)            
            alpha_old = alpha
            # Choose alpha in (alpha_old,alpha_max)
            alpha = 0.5*(alpha_old + self.alpha_max)
            # Make sure alpha does not get too close to 1 or 0
            if (1-alpha <= self.alpha_tol or alpha <= self.alpha_tol):
                break 

            i += 1
            
        # If line search failed, try backtrack
        return self.line_search_backtrack(x0,objective_fun,objective_grad_fun,\
                                          f0,gradf0,p)

            
    def zoom(self,alpha_low,alpha_high,x0,objective_fun,objective_grad_fun,f0,\
             gradf0,p):
        """
        (1) The interval bounded by alpha_low and alpha_high contains step 
        lengths that satisfy the strong Wolfe conditions.
        (2) alpha_low is, among all step lengths generated so far that satisfy 
        the sufficient condition, the one giving the smallest function value.
        (3) alpha_high is chosen so that phi'(alpha_low) * (alpha_high - alpha_low) < 0 
        """
        # phi(alpha) = f(x + alpha * p)
        # phi'(alpha) = f'(x + alpha * p) * p
        phiprime0 = np.dot(gradf0,p)
        x_low = x0 + alpha_low * p
        f_low = objective_fun(x_low)
        
        # Assumption 1 - One of these three must be satisfied
        x_high = x0 + alpha_high * p
        f_high = objective_fun(x_high)
        cond1 = f_high > f0 + self.c1 * alpha_high * phiprime0
        cond2 = f_high >= f_low
        cond3 = np.dot(objective_grad_fun(x_high),p) >= 0 
        if (self.test):
            assert(cond1 or cond2 or cond3)
            # Assumption 2 - alpha_low gives smallest function value
            assert(f_low < f_high)
            # Assumption 3 - phi'(alpha_low) * (alpha_high - alpha_low) < 0 
            assert(np.dot(objective_grad_fun(x_low),p) \
                      * (alpha_high - alpha_low) < 0)
        
        while (True):
            # Interpolate to find trial step length in [alpha_low,alpha_high]
            alpha = 0.5*(alpha_low + alpha_high)
            x = x0 + alpha * p
            f = objective_fun(x)
            cond1 = (f > f0 + self.c1 * alpha * phiprime0)
            cond2 = f >= f_low
            if (cond1 or cond2):
                alpha_high = alpha
            else:
                gradf = objective_grad_fun(x)
                phiprime = np.dot(gradf,p)
                if (np.abs(phiprime) <= - self.c2 * phiprime0):
                    return alpha, f, gradf
                if (phiprime * (alpha_high - alpha_low) >= 0):
                    alpha_high = alpha_low
                    x_low = x0 + alpha_low * p
                    f_low = objective_fun(x_low)  
                alpha_low = alpha
                
            # Assumption 1 - One of these three must be satisfied
            x_high = x0 + alpha_high * p
            f_high = objective_fun(x_high)
            x_low = x0 + alpha_low * p
            f_low = objective_fun(x_low)
            cond1 = f_high > f0 + self.c1 * alpha_high * phiprime0
            cond2 = f_high >= f_low
            cond3 = np.dot(objective_grad_fun(x_high),p) >= 0 
            if (self.test):
                assert(cond1 or cond2 or cond3)
                # Assumption 2 - alpha_low gives smallest function value
                assert(f_low < f_high)
                # Assumption 3 - phi'(alpha_low) * (alpha_high - alpha_low) < 0 
                assert(np.dot(objective_grad_fun(x_low),p) \
                          * (alpha_high - alpha_low) < 0)

