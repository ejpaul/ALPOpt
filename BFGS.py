import scipy.linalg
import numpy as np

class BFGS:
    def __init__(self,epsilon=1e-12,beta=1,alpha_max=1,alpha_scale=0.1,
                alpha_tol=1e-12,rho_backtrack=0.9,alpha0=1.0,c1=1e-4,c2=0.9,
                test=False,line_search='backtrack',ftol=1e-12,xtol=1e-12,
                niter_max=1e3):
        # Hyperparameters
        self.epsilon = epsilon # tolerance in function gradient
        self.beta = beta # Scaling for initial Hessian 
        self.alpha_max = alpha_max # Maximum step length
        self.alpha_scale = alpha_scale # Scaling for initial step length
        self.alpha_tol = alpha_tol # 1-alpha >= alpha_tol
        self.rho_backtrack = rho_backtrack
        self.alpha0 = alpha0 # Initial step size for backtrack
        self.c1 = c1
        self.c2 = c2
        self.test = test
        self.line_search = line_search
        self.ftol = ftol
        self.xtol = xtol
        self.niter_max = niter_max
        
        self.niter = 0
        self.norm_grad_hist = []
        self.f_hist = []
        self.alpha_hist = []
        self.exit_string = ['gtol achieved','ftol achieved','xtol achieved','niter_max achieved']
        
    def reset_hist(self):
        self.f_hist = []
        self.alpha_hist = []
        self.norm_grad_hist = []

    def BFGS_opt(self,x0,objective_fun,objective_grad_fun,cond=None):
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
        exit_code = 0
        while (norm_grad_f > self.epsilon):
            self.niter += 1
            print("f : "+str(f))
            print("norm_grad_f : "+str(norm_grad_f))
            self.norm_grad_hist.append(norm_grad_f)
            self.f_hist.append(f)
            # Compute search direction
            p = -np.dot(H,grad_f)
            # Check that this is a descent direction
            if (np.dot(p,grad_f)>=0):
                print('Descent direction not obtained. Using steepest descent.')
                p = -grad_f
            # Perform line search
            print('Performing line search')
            if (self.line_search == 'backtrack'):
                 [alpha,f_new,grad_f_new] = self.line_search_backtrack_NR(x,\
                                       objective_fun,objective_grad_fun,f,grad_f,p)               
            elif (self.line_search == 'wolfe'):
                [alpha,f_new,grad_f_new] = self.line_search_wolfe(x,\
                                       objective_fun,objective_grad_fun,f,grad_f,p)
            elif (self.line_search == 'scipy'):
                if (self.niter > 1):
                    [alpha,fc,gc,f_new,f,grad_dot_p] = \
                        scipy.optimize.line_search(objective_fun, objective_grad_fun, x,\
                                           p, gfk=grad_f, old_fval=f, \
                                           old_old_fval=self.f_hist[-2],maxiter=10000)
                    print(alpha)
                    grad_f_new = objective_grad_fun(x + alpha * p)
                else:
                    [alpha,fc,gc,f_new,f,grad_dot_p] = \
                        scipy.optimize.line_search(objective_fun, objective_grad_fun, x,\
                                           p, gfk=grad_f, old_fval=f,maxiter=10000) 
                    print(alpha)
                    grad_f_new = objective_grad_fun(x + alpha * p)
            # Check termination criteria
            if (self.niter > 1):
                if ((np.abs(f_new - f) < self.ftol)):
                    exit_code = 1
                    f = f_new
                    grad_f = grad_f_new
                    norm_grad_f = scipy.linalg.norm(grad_f)
                    break
                if (scipy.linalg.norm(alpha*p) < self.xtol):
                    exit_code = 2
                    f = f_new
                    grad_f = grad_f_new
                    norm_grad_f = scipy.linalg.norm(grad_f)
                    break 
            if (self.niter > self.niter_max):
                exit_code = 3
                f = f_new
                grad_f = grad_f_new
                norm_grad_f = scipy.linalg.norm(grad_f)
                break                 
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
        print('Exit code: '+str(exit_code))
        print(self.exit_string[exit_code])
        # Change this to determine outcome of optimization
        return x, f, exit_code
    
    def line_search_backtrack_NR(self,x0,objective_fun,objective_grad_fun,f0,gradf0,p):
        """
        Numerical Recipes p. 479
        """
        exit_code = 0
        # Normalize search direction
        norm_p = scipy.linalg.norm(p)
        if (norm_p > 1):
            p = p/norm_p
            
        slope = np.dot(gradf0,p)
        if (slope >= 0):
            print('Roundoff problem in line search.')
            sys.exit(0)
        # Compute alpha_min
        alpha_min = 0
        for i in range(len(p)):
            temp = np.abs(p[i])/max(abs(x0[i]),1)
            if (temp > alpha_min):
                alpha_min = temp
        alpha_min = self.xtol/alpha_min
        
        rhs2 = 0
        alpha2 = 0
        alpha = 1.0 # Try full Newton step first
        while True:
            x = x0 + alpha * p
            f = objective_fun(x)
            # Convergence on xtol
            if (alpha < alpha_min):
                gradf_new = objective_grad_fun(x)
                print('Line search terminated due to xtol')
                print('alpha_min = '+str(alpha_min))
                return alpha, f, gradf_new
            # Sufficient decrease condition
            elif (f <= f0 + self.c1 * alpha * slope):
                gradf_new = objective_grad_fun(x)
                print('Line search terminated due to sufficient decrease')
                return alpha, f, gradf_new
            # Backtrack
            else:
                # First time
                if (alpha == 1.0):
                    tmplam = - slope/(2.0*(f-f0-slope))
                else:
                    rhs1 = f - f0 - alpha * slope
                    rhs2 = f2 - f0 - alpha2 * slope
                    a = (rhs1/(alpha**2) - rhs2/(alpha2**2))/(alpha-alpha2)
                    b = (-alpha2*rhs1/(alpha**2) + alpha*rhs2/(alpha2**2))/(alpha-alpha2)
                    if (a == 0):
                        tmplam = -slope/(2.0*b)
                    else:
                        disc = b**2 - 3.0*a*slope
                        if (disc < 0):
                            tmplam = 0.5*alpha
                        elif (b <= 0):
                            tmplam = (-b + np.sqrt(disc))/(3.0*a)
                        else:
                            tmplam = -slope/(b + np.sqrt(disc))
                        if (tmplam > 0.5*alpha):
                            tmplam = 0.5*alpha # lambda <= 0.5*lambda1
                alpha2 = alpha
                f2 = f
                alpha = max(tmplam,0.1*alpha) # lambda >= 0.1*lambda1
                print('tmplam = '+str(tmplam))
                print('alpha = '+str(alpha))
    
    def line_search_backtrack(self,x0,objective_fun,objective_grad_fun,f0,gradf0,p,cond=None):
        """
        f0 = initial objective function value
        gradf0 = initial objective function gradient
        
        Algorith 3.1 from Nocedal & Wright
        
        """        
        # Default condition 
        if cond is None:
            def cond(x):
                return True
            
        # Initialize step size
        alpha = self.alpha0
        f_new = objective_fun(x0 + alpha*p)
        
        # Test Armijo condition and user-specified condition
        armijo = f_new <= f0 + self.c1 * alpha * np.dot(gradf0,p)
        user_cond = cond(x0)
        while ((user_cond and armijo) == False):
            # Backtrack
            alpha *= self.rho_backtrack
            f_new = objective_fun(x0 + alpha*p)
            
            armijo = f_new <= f0 + self.c1 * alpha * np.dot(gradf0,p)
            user_cond = cond(x0 + alpha*p)

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

