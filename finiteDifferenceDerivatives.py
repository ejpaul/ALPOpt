import numpy as np
import numbers

class finiteDifference:
  def __init__(self,method,epsilon,function,args):
    self.method = method
    self.epsilon = epsilon
    self.function = function
    self.args = args
    
  def evaluate(self,x):
    if (self.args is not None):
      return self.function(x,*self.args).copy()
    else:
      return self.function(x).copy()
    
  def evaluateEpsilon(self,x,epsilon):
    x_epsilon = np.copy(x)+epsilon
    return self.evaluate(x_epsilon) 

# Approximates finite difference derivative with an N-point stencil with step size epsilon
def finiteDifferenceDerivative(x,function,args=None,epsilon=1e-2,method='forward'):
  assert method in ['forward','centered'], "method passed to finiteDifferenceDerivative must be 'forward' or 'centered'."
  
  if isinstance(x,numbers.Number):
    x = np.array([x])
  assert(isinstance(x,(list,np.ndarray)))
  if isinstance(x,list):
    x = np.array(x)
  assert(x.ndim==1)
  
  finiteDifferenceObject = finiteDifference(method,epsilon,function,args)
  # Call function once to get size of output
  test_function = finiteDifferenceObject.evaluate(x)
  dims = [len(x)]
  dims.extend(list(np.shape(test_function)))
  dfdx = np.zeros(dims)
  for i in range(np.size(x)):
    step = np.zeros(np.shape(x))
    step[i] = epsilon
    if (method=='centered'):
      function_r = finiteDifferenceObject.evaluateEpsilon(x,step)
      function_l = finiteDifferenceObject.evaluateEpsilon(x,-step)
      if (dims==()):
        dfdx[i] = (function_r-function_l)/(2*epsilon)
      else:
        dfdx[i,...] = (function_r-function_l)/(2*epsilon)
    if (method=='forward'):
      function_r = finiteDifferenceObject.evaluateEpsilon(x,step)
      function_l = test_function
      if (dims==()):
        dfdx[i] = (function_r-function_l)/(epsilon)
      else:
        dfdx[i,...] = (function_r-function_l)/(epsilon)
  return dfdx
