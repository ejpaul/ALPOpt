
  """
  A class used for gradient-based optimization
  
  ...
  Attributes
  ---------
  objective (function) : objective functions, f[x]
    def f(x):
      ...
      return objective_value
  inequalityConstraints (function or list of functions) : list of inequality constraints, g_i[x] >= 0
    def g(x):
      ...
      return constraint_value
  equalityConstraints (function or list of functions) : list of equality constraints, h_i[x] = 0

  """

class gradBasedOptimizer:
  def __init__(self,Nparameters,objectives,objectiveWeights,objectivesGradient,inequalityConstraints=None,\
      equalityConstraints=None,inequalityConstraintsGradient=None,equalityConstraintsGradient=None,\
      boundConstraintsMin=None,boundConstraintsMax=None,name='optimization')
    
    # Type checking
    assert (isinstance(Nparameters,int) and Nparameters>0)
    
    assert (isinstance(objectives,(list,np.ndarray)) or objectives.type == types.FunctionType)
    if (isinstance(objectives,(list,np.ndarray))):
      for i in range(len(objectives)):
        assert objectives[i].type == types.FunctionType
      self.NObjectives = len(objectives)
      assert (isinstance(objectiveWeights,(list,np.ndarray)))
      assert len(objectiveWeights) == len(objectives)
      for i in range(len(objectiveWeights)):
        assert isinstance(objectiveWeights[i],numbers.Number)
      assert (isinstance(objectivesGradient,(list,np.ndarray)))
      assert len(objectivesGradient) == len(objectives)
      for i in range(len(objectivesGradient)):
        assert objectivesGradient[i].type == types.FunctionType
    else:
      self.NObjectives = 1
      assert (isinstance(objectiveWeights,numbers.Number))
      assert (objectivesGradient.type == types.FunctionType)

    if (inequalityConstraints is not None):
      assert (isinstance(inequalityConstraints,(list,np.ndarray)) or inequalityConstraints.type == types.FunctionType)
      assert inequalityConstraintsGradient is not None
      if (isinstance(inequalityConstraints,(list,np.ndarray))):
        for i in range(len(inequalityConstraints)):
          assert inequalityConstraints[i].type == types.FunctionType
        assert (isinstance(inequalityConstraintsGradient,(list,np.ndarray)))
        assert len(inequalityConstraints)==len(inequalityConstraintsGradient)
        for i in range(len(inequalityConstraintsGradient,(list,np.ndarray))):
          assert inequalityConstaintsGradient[i].type == types.FunctionType
        self.NinequalityConstraints = len(inequalityConstraints)
      else:
        assert (inequalityConstraintsGradient.type == types.FunctionType)
        self.NinequalityConstraints = 1
      self.inequalityConstrained = True

    if (equalityConstraints is not None):      
      assert (isinstance(equalityConstraints,(list,np.ndarray)) or equalityConstraints.type == types.FunctionType)
      assert (equalityConstraintsGradient is not None)
      if (isinstance(equalityConstraints,(list,np.ndarray))):
        for i in range(len(equalityConstraints)):
          assert equalityConstraints[i].type == types.FunctionType
        assert (isinstance(equalityConstraintsGradient,(list,np.ndarray))
        assert(len(equalityConstraints)==len(equalityConstraintsGradient))
        for i in range(len(equalityConstraintsGradient)):
          assert equalityConstraintsGradient[i].type == types.FunctionType
        self.NequalityConstraints = len(equalityConstraints)
      else:
        assert (equalityConstraintsGradient.type == types.FunctionType)
        self.NequalityConstraints = 1
      self.equalityConstrained = True

    if (boundConstraintsMin is not None):
      assert (boundConstraintsMin.type==numbers.Number or isinstance(boundConstraintsMin,(list,np.array)))
      if (isinstance(boundConstraintsMin,(list,np.array))):
        assert len(boundConstraintsMin) == Nparameters
      self.boundConstrained = True
    if (boundConstraintsMax is not None):
      assert (boundConstraintsMax.type==numbers.Number or isinstance(boundConstraintsMax,(list,np.array)))
      if (isinstance(boundConstraintsMax,(list,np.array))):
        assert len(boundConstraintsMax) == Nparameters
      self.boundConstrained = True
      
    self.Nparameters = Nparameters
    self.objectives = objectives
    self.objectivesWeight = objectivesWeight
    self.objectivesGradient = objectivesGradient
    self.inequalityConstraints = inequalityConstraints
    self.equalityConstraints = equalityConstraints
    self.inequalityConstraintsGradient = inequalityConstraintsGradient
    self.equalityConstraintsGradient = equalityConstraintsGradient
    
    self.name = name
    self.objectivesHistory = []
    self.objectivesGradientNormHistory = []
    self.objectiveHistory = []
    
    self.nevalObjectives = 0
    self.nevalObjectiveGradient = 0 
    if (self.inequalityConstrainted):
      self.inequalityConstraintsHistory = []
      self.inequalityConstraintsGradientNormHistory = []
      self.nevalInequalityConstraints = 0
      self.nevalInequalityConstraintsGradient = 0
    if (self.equalityConstrained):
      self.equalityConstraintsHistory = []
      self.equalityConstraintsGradientNormHistory = []
      self.nevalEqualityConstraints = 0
      self.nevalEqualityConstraintsGradient = 0                

    self.nlopt_methods = ('MMA','SLSQP','CCSAQ','LBFGS','TNEWTON','TNEWTON_PRECOND_RESTART',\
        'TNEWTON_PRECOND','TNEWTON_RESTART','VAR2','VAR1','StOGO','STOGO_RAND',\
        'MLSL','MLSL_LDS')
    self.nlopt_methods_inequalityConstrained = ('MMA','SLSQP','CCSAQ')
    self.nlopt_methods_equalityConstrained = ('SLSQP','CCSAQ')
    self.nlopt_methods_boundConstrained = ('SLSQP')
    self.nlopt_dict = {'MMA': nlopt.LD_MMA, 'SLSQP': nlopt.LD_SLSQP, 'CCSAQ': nlopt.LD_CCSAQ,\
      'LBFGS': nlopt.LD_LBFGS, 'TNEWTON': nlopt.LD_TNEWTON, 'TNEWTON_PRECOND_RESTART': nlopt.LD_TNEWTON_PRECOND_RESTART,\
      'TNEWTON_PRECOND': nlopt.LD_TNEWTON_PRECOND, 'TNEWTON_RESTART': nlopt.LD_TNEWTON_RESTART,\
      'VAR2': nlopt.LD_VAR2, 'VAR1': nlopt.LD_VAR1, 'SToGO': nlopt.GD_STOGO, \
      'STOGO_RAND': nlopt.GD_STOGO_RAND}
    
    self.scipy_methods = ('CG','BFGS','Newton-CG','L-BFGS-B','TNC','SLSQP','dogleg','trust-ncg')
    self.scipy_methods_equalityConstrained = ('SLSQP')
    self.scipy_methods_equalityConstrained = ('SLSQP')
    self.scipy_methods_boundConstrained = ('L-BFGS-B','TNC','SLSQP')
    
    
  def optimize(x,module='nlopt',method='CCSAQ',ftolAbs,ftolRel,xtolAbs,xtolRel):
    assert (module in ('nlopt','scipy'))
    if (module == 'nlopt'):
      assert (method in self.nlopt_methods)
      if (self.inequalityConstrained):
        assert method in self.nlopt_methods_inequalityConstrained
      if (self.equalityConstrained):
        assert method in self.nlopt_methods_equalityConstrained
      if (self.boundConstrained):
        assert method in self.nlopt_methods_boundConstrained
      self.nlopt_optimize(method,ftolAbs,ftolRel,xtolAbs,xtolRel)
    if (module == 'scipy'):
      assert (method in self.scipy_methods)
      if (self.inequalityConstrained):
        assert method in self.scipy_methods_inequalityConstrained
      if (self.equalityConstrained):
        assert method in self.scipy_methods_equalityConstrained
      if (self.boundConstrained):
        assert method in self.scipy_methods_boundConstrained
      self.scipy_optimize(method,ftolAbs,ftolRel,xtolAbs,xtolRel)
                            
  def nlopt_optimize(self,method='BFGS',ftolAbs=1e-4,ftolRel=1e-4,xtolAbs=1e-4,\
    xtolRel=1e-4,maxeval=10000,maxtime=10000):
    assert (isinstance(maxeval,int))
    assert (isinstance(maxtime,int))
    assert (isinstance(ftolAbs,float))
    assert (isinstance(ftolRel,float))
    assert (isinstance(xtolAbs,float))
    assert (isinstance(xtolRel,float))
    opt = nlopt.opt(self.nlopt_dict[method],self.Nparameters)
    opt.set_ftol_abs(ftolAbs)
    opt.set_ftol_rel(ftolRel)
    opt.set_xtol_abs(xtolAbs)
    opt.set_xtol_rel(xtolRel)
    opt.set_maxeval(maxeval)
    opt.set_maxtime(maxtime)
    opt.set_min_objective(self.nlopt_function)
    if (self.equalityConstrained):
      if (self.NequalityConstraints == 1):
        for i in range(self.NequalityConstraints):
          opt.set_ 
      
    opt.add_equality_constraint(nlopt_optimize_object.nlopt_constraint_proximity,0)

  def scipy_optimize(self,method,ftolAbs,ftolRel,xtolAbs,xtolRel):
    opt = nlopt.opt(self.nlopt_dict[method],self.Nparameters)
    opt.set_ftol_abs(1e-4)
    opt.set_ftol_rel(1e-4)
    
  boundary_opt = np.copy(vmecOptimizationObject.boundary_opt)
  opt.set_min_objective(nlopt_optimize_object.nlopt_function)
  tol = np.zeros(vmecOptimizationObject.nzeta*vmecOptimizationObject.ntheta)
  opt.add_inequality_constraint(nlopt_optimize_object.nlopt_constraint_proximity,0)
                
  def objectivesFun(self,x):
    if self.Nobjectives > 1:
      objectiveValues = np.zeros(self.Nobjectives)
      for i in range(self.Nobjectives):
        objectiveValues[i] = self.objectives[i](x)
    else
      objectiveValues = self.objectives(x)
    if (self.feval_objectives == 1):
      self.objectivesHistory = objectiveValues
    else:
      self.objectivesHistory = np.vstack(self.objectivesHistory,objectiveValues)
    objective = np.dot(np.array(self.objectiveWeights),np.array(objectiveValues))
    self.objectiveHistory.append(objective)
    self.nevalObjectives += 1
    return objective
                
  def objectivesGradientFun(self,x):
    objectivesGradientValue = np.zeros(self.Nparameters,self.Nobjectives)
    for i in range(self.Nobjectives):
      objectivesGradientValue[:,i] = self.objectivesGradient[i](x)
    
    objectiveGradient = np.matmul(np.array(objectivesGradientValue),np.array(self.objectiveWeights))
    gradientNorm = scipy.linalg.norm(objectiveGradient)
    self.nevalObjectivesGradient += 1
    return objectiveGradient
              
  
