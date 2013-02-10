__docformat__ = "restructuredtext en"
from numpy import *
from oologfcn import *
from oographics import Graphics
from setDefaultIterFuncs import setDefaultIterFuncs, denyingStopFuncs
from nonLinFuncs import nonLinFuncs
from residuals import residuals
from ooIter import ooIter

#from Point import Point currently lead to bug
from openopt.kernel.Point import Point

from iterPrint import ooTextOutput
from ooMisc import setNonLinFuncsNumber, assignScript, norm
from nonOptMisc import isspmatrix, scipyInstalled, scipyAbsentMsg, csr_matrix, Vstack, Hstack, EmptyClass, isPyPy, oosolver
from copy import copy as Copy
try:
    from DerApproximator import check_d1
    DerApproximatorIsInstalled = True
except:
    DerApproximatorIsInstalled = False

ProbDefaults = {'diffInt': 1.5e-8,  'xtol': 1e-6,  'noise': 0}
from runProbSolver import runProbSolver
import GUI
from fdmisc import setStartVectorAndTranslators


class user:
    def __init__(self):
        pass

class oomatrix:
    def __init__(self):
        pass
    def matMultVec(self, x, y):
        return dot(x, y) if not isspmatrix(x) else x._mul_sparse_matrix(csr_matrix(y.reshape((y.size, 1)))).A.flatten() 
    def matmult(self, x, y):
        return dot(x, y)
        #return asarray(x) ** asarray(y)
    def dotmult(self, x, y):
        return x * y
        #return asarray(x) * asarray(y)

class autocreate:
    def __init__(self): pass

class baseProblem(oomatrix, residuals, ooTextOutput):
    isObjFunValueASingleNumber = True
    manage = GUI.manage # GUI func
    #_useGUIManager = False # TODO: implement it
    prepared = False
    _baseProblemIsPrepared = False
    
    name = 'unnamed'
    state = 'init'# other: paused, running etc
    castFrom = '' # used by converters qp2nlp etc
    nonStopMsg = ''
    xlabel = 'time'
    plot = False # draw picture or not
    show = True # use command pylab.show() after solver finish or not

    iter = 0
    cpuTimeElapsed = 0.
    TimeElapsed = 0.
    isFinished = False
    invertObjFunc = False # True for goal = 'max' or 'maximum'
    nProc = 1 # number of processors to use

    lastPrintedIter = -1
    
    iterObjFunTextFormat = '%0.3e'
    finalObjFunTextFormat = '%0.8g'
    debug = 0
    
    iprint = 10
    #if iprint<0 -- no output
    #if iprint==0 -- final output only

    maxDistributionSize = 0 # used in stochastic problems

    maxIter = 1000
    maxFunEvals = 10000 # TODO: move it to NinLinProblem class?
    maxCPUTime = inf
    maxTime = inf
    maxLineSearch = 500 # TODO: move it to NinLinProblem class?
    xtol = ProbDefaults['xtol'] # TODO: move it to NinLinProblem class?
    gtol = 1e-6 # TODO: move it to NinLinProblem class?
    ftol = 1e-6
    contol = 1e-6
    
    fTol = None

    minIter = 0
    minFunEvals = 0
    minCPUTime = 0.0
    minTime = 0.0
    
    storeIterPoints = False 

    userStop = False # becomes True is stopped by user
    
    useSparse = 'auto' # involve sparse matrices: 'auto' (autoselect, premature) | True | False
    useAttachedConstraints = False

    x0 = None
    isFDmodel = False # OO kernel set it to True if oovars/oofuns are used

    noise = ProbDefaults['noise'] # TODO: move it to NinLinProblem class?

    showFeas = False
    useScaledResidualOutput = False

    hasLogicalConstraints = False
    
    # A * x <= b inequalities
    A = None
    b = None

    # Aeq * x = b equalities
    Aeq = None
    beq = None
    
    scale = None

    goal = None# should be redefined by child class
    # possible values: 'maximum', 'min', 'max', 'minimum', 'minimax' etc
    showGoal = False# can be redefined by child class, used for text & graphic output

    color = 'b' # blue, color for plotting
    specifier = '-'# simple line for plotting
    plotOnlyCurrentMinimum = False # some classes like GLP change the default to True
    xlim = (nan,  nan)
    ylim = (nan,  nan)
    legend = ''

    fixedVars = None
    freeVars = None
    
    istop = 0
    
    maxSolutions = 1 # used in interalg and mb other solvers

    fEnough = -inf # if value less than fEnough will be obtained
    # and all constraints no greater than contol
    # then solver will be stopped.
    # this param is handled in iterfcn of OpenOpt kernel
    # so it may be ignored with some solvers not closely connected to OO kernel

    fOpt = None # optimal value, if known
    implicitBounds = inf
    
    convex = 'unknown' # used in interalg
    _linear_objective = False # used in interalg

    def __init__(self, *args, **kwargs):
        # TODO: add the field to ALL classes
        self.err = ooerr
        self.warn = oowarn
        self.info = ooinfo
        self.hint = oohint
        self.pWarn = ooPWarn
        self.disp = oodisp
        self.data4TextOutput = ['objFunVal', 'log10(maxResidual)']
        self.nEvals = {}
        
        
        if hasattr(self, 'expectedArgs'): 
            if len(self.expectedArgs)<len(args):
                self.err('Too much arguments for '+self.probType +': '+ str(len(args)) +' are got, at most '+ str(len(self.expectedArgs)) + ' were expected')
            for i, arg in enumerate(args):
                setattr(self, self.expectedArgs[i], arg)
        self.norm = norm
        self.denyingStopFuncs = denyingStopFuncs()
        self.iterfcn = lambda *args, **kwargs: ooIter(self, *args, **kwargs)# this parameter is only for OpenOpt developers, not common users
        self.graphics = Graphics()
        self.user = user()
        self.F = lambda x: self.objFuncMultiple2Single(self.objFunc(x)) # TODO: should be changes for LP, MILP, QP classes!

        self.point = lambda *args,  **kwargs: Point(self, *args,  **kwargs)

        self.timeElapsedForPlotting = [0.]
        self.cpuTimeElapsedForPlotting = [0.]
        #user can redirect these ones, as well as debugmsg
        self.debugmsg = lambda msg: oodebugmsg(self,  msg)
        
        self.constraints = [] # used in isFDmodel

        self.callback = [] # user-defined callback function(s)
        
        self.solverParams = autocreate()

        self.userProvided = autocreate()

        self.special = autocreate()

        self.intVars = [] # for problems like MILP
        self.binVars = [] # for problems like MILP
        self.optionalData = []#string names of optional data like 'c', 'h', 'Aeq' etc
        
        if self.allowedGoals is not None: # None in EIG
            if 'min' in self.allowedGoals:
                self.minimize = lambda *args, **kwargs: minimize(self, *args, **kwargs)
            if 'max' in self.allowedGoals:
                self.maximize = lambda *args, **kwargs: maximize(self, *args, **kwargs)
                
        assignScript(self, kwargs)

    def __finalize__(self):
        if self.isFDmodel:
            self.xf = self._vector2point(self.xf)

    def objFunc(self, x):
        return self.f(x) # is overdetermined in LP, QP, LLSP etc classes

    def __isFiniteBoxBounded__(self): # TODO: make this function 'lazy'
        return all(isfinite(self.ub)) and all(isfinite(self.lb))

    def __isNoMoreThanBoxBounded__(self): # TODO: make this function 'lazy'
        return self.b.size ==0 and self.beq.size==0 and (self._baseClassName == 'Matrix' or (not self.userProvided.c and not self.userProvided.h))

#    def __1stBetterThan2nd__(self,  f1, f2,  r1=None,  r2=None):
#        if self.isUC:
#            #TODO: check for goal = max/maximum
#            return f1 < f2
#        else:#then r1, r2 should be defined
#            return (r1 < r2 and  self.contol < r2) or (((r1 <= self.contol and r2 <=  self.contol) or r1==r2) and f1 < f2)
#
#    def __1stCertainlyBetterThan2ndTakingIntoAcoountNoise__(self,   f1, f2,  r1=None,  r2=None):
#        if self.isUC:
#            #TODO: check for goalType = max
#            return f1 + self.noise < f2 - self.noise
#        else:
#            #return (r1 + self.noise < r2 - self.noise and  self.contol < r2) or \
#            return (r1 < r2  and  self.contol < r2) or \
#            (((r1 <= self.contol and r2 <=  self.contol) or r1==r2) and f1 + self.noise < f2 - self.noise)


    def solve(self, *args, **kwargs):
        return runProbSolver(self, *args, **kwargs)
        
    def _solve(self, *args, **kwargs):
        self.debug = True
        return self.solve(*args, **kwargs)
    
    def objFuncMultiple2Single(self, f):
        #this function can be overdetermined by child class
        if asfarray(f).size != 1: self.err('unexpected f size. The function should be redefined in OO child class, inform OO developers')
        return f

    def inspire(self, newProb, sameConstraints=True):
        # fills some fields of new prob with old prob values
        newProb.castFrom = self.probType

        #TODO: hold it in single place

        fieldsToAssert = ['contol', 'xtol', 'ftol', 'gtol', 'iprint', 'maxIter', 'maxTime', 'maxCPUTime','fEnough', 'goal', 'color', 'debug', 'maxFunEvals', 'xlabel']
        # TODO: boolVars, intVars
        if sameConstraints: fieldsToAssert+= ['lb', 'ub', 'A', 'Aeq', 'b', 'beq']

        for key in fieldsToAssert:
            if hasattr(self, key): setattr(newProb, key, getattr(self, key))


        # note: because of 'userProvided' from prev line
        #self self.userProvided is same to newProb.userProvided
        
#        for key in ['f','df', 'd2f']:
#                if hasattr(self.userProvided, key) and getattr(self.userProvided, key):
#                    setattr(newProb, key, getattr(self.user, key))
        
        Arr = ['f', 'df']
        if sameConstraints:
            Arr += ['c','dc','h','dh','d2c','d2h']
        
        for key in Arr:
            if hasattr(self.userProvided, key):
                if getattr(self.userProvided, key):
                    #setattr(newProb, key, getattr(self.user, key))
                    setattr(newProb, key, getattr(self, key)) if self.isFDmodel else setattr(newProb, key, getattr(self.user, key))
                else:
                    setattr(newProb, key, None)
                        
    FuncDesignerSign = 'f'

    def _isFDmodel(self):
        try:
            #from FuncDesigner.ooFun import oofun
            from FuncDesigner import ooarray, oofun
        except ImportError:
            return False
        fds = getattr(self, self.FuncDesignerSign, None)
        if fds is None:
            return False        
        if isinstance(fds, (oofun, ooarray)):
            return True
        if isinstance(fds, dict):
            return True if isinstance(list(fds.keys())[0], (oofun, ooarray)) else False
        if isinstance(fds, (list, tuple, ndarray)):
            if isinstance(fds[0], (oofun, ooarray)):
                return True
            elif isinstance(fds[0], (list, tuple, ndarray)):
                return isinstance(fds[0][0], (oofun, ooarray))
        return False
    
    # Base class method
    def _prepare(self): 
        if self._baseProblemIsPrepared: return
        if self.useSparse == 0:
            self.useSparse = False
        elif self.useSparse == 1:
            self.useSparse = True
        if self.useSparse == 'auto' and not scipyInstalled:
            self.useSparse = False
        if self.useSparse == True and not scipyInstalled:
            self.err("You can't set useSparse=True without scipy installed")
        if self._isFDmodel():
            self.isFDmodel = True
            self._FD = EmptyClass()
            self._FD.nonBoxConsWithTolShift = []
            self._FD.nonBoxCons = []
            from FuncDesigner import _getAllAttachedConstraints, _getDiffVarsID, ooarray, oopoint, oofun#, _Stochastic
            self._FDVarsID = _getDiffVarsID()
            
            probDep = set()
            updateDep = lambda Dep, elem: [updateDep(Dep, f) for f in elem] if isinstance(elem, (tuple, list, set, ndarray))\
            else Dep.update(elem._getDep()) if isinstance(elem, oofun) else None
            
            if self.probType in ['SLE', 'NLSP', 'SNLE', 'LLSP']:
                equations = self.C if self.probType in ('SLE', 'LLSP') else self.f
                F = equations
                updateDep(probDep, equations)
                ConstraintTags = [(elem if not isinstance(elem, (list, tuple, ndarray)) else elem[0]).isConstraint for elem in equations]
                cond_all_oofuns_but_not_cons = not any(ConstraintTags) 
                cond_cons = all(ConstraintTags) 
                if not cond_all_oofuns_but_not_cons and not cond_cons:
                    raise OpenOptException('for FuncDesigner SLE/SNLE constructors args must be either all-equalities or all-oofuns')            
                if self.fTol is not None:
                    fTol = min((self.ftol, self.fTol))
                    self.warn('''
                    both ftol and fTol are passed to the SNLE;
                    minimal value of the pair will be used (%0.1e);
                    also, you can modify each personal tolerance for equation, e.g. 
                    equations = [(sin(x)+cos(y)=-0.5)(tol = 0.001), ...]
                    ''' % fTol)
                else:
                    fTol = self.ftol
                self.fTol = self.ftol = fTol
                appender = lambda arg: [appender(elem) for elem in arg] if isinstance(arg, (ndarray, list, tuple, set))\
                else ((arg.oofun*(fTol/arg.tol) if arg.tol != 0 else arg.oofun) if arg.isConstraint else arg)
                EQs = []
                for eq in equations:
                    rr = appender(eq)
                    if type(rr) == list:
                        EQs += rr
                    else:
                        EQs.append(rr)
                #EQs = [((elem.oofun*(fTol/elem.tol) if elem.tol != 0 else elem.oofun) if elem.isConstraint else elem) for elem in equations]
                if self.probType in ('SLE', 'LLSP'): self.C = EQs
                elif self.probType in ('NLSP', 'SNLE'): self.f = EQs
                else: raise OpenOptException('bug in OO kernel')
            else:
                F = [self.f]
                updateDep(probDep, self.f)
            updateDep(probDep, self.constraints)
            
            # TODO: implement it
            
#            startPointVars = set(self.x0.keys())
#            D = startPointVars.difference(probDep)
#            if len(D):
#                print('values for variables %s are missing in start point' % D)
#            D2 = probDep.difference(startPointVars)
#            if len(D2):
#                self.x0 = dict([(key, self.x0[key]) for key in D2])

            for fn in ['lb', 'ub', 'A', 'Aeq', 'b', 'beq']:
                if not hasattr(self, fn): continue
                val = getattr(self, fn)
                if val is not None and any(isfinite(val)):
                    self.err('while using oovars providing lb, ub, A, Aeq for whole prob is forbidden, use for each oovar instead')
                    
            if not isinstance(self.x0, dict):
                self.err('Unexpected start point type: ooPoint or Python dict expected, '+ str(type(self.x0)) + ' obtained')
            
            x0 = self.x0.copy()

            tmp = []
            for key, val in x0.items():
                if not isinstance(key, (list, tuple, ndarray)):
                    tmp.append((key, val))
                else: # can be only ooarray although
                    val = atleast_1d(val)
                    if len(key) != val.size:
                        self.err('''
                        for the sake of possible bugs prevention lenght of oovars array 
                        must be equal to lenght of its start point value, 
                        assignments like x = oovars(m); startPoint[x] = 0 are forbidden, 
                        use startPoint[x] = [0]*m or np.zeros(m) instead''')
                    for i in range(val.size):
                        tmp.append((key[i], val[i]))
            Tmp = dict(tmp)
            
            if isinstance(self.fixedVars, dict):
                for key, val in self.fixedVars.items():
                    if isinstance(key, (list, tuple, ndarray)): # can be only ooarray although
                        if len(key) != len(val):
                            self.err('''
                            for the sake of possible bugs prevention lenght of oovars array 
                            must be equal to lenght of its start point value, 
                            assignments like x = oovars(m); fixedVars[x] = 0 are forbidden, 
                            use fixedVars[x] = [0]*m or np.zeros(m) instead''')
                        for i in range(len(val)):
                            Tmp[key[i]] = val[i]
                    else:
                        Tmp[key] = val
                self.fixedVars = set(self.fixedVars.keys())
            # mb other operations will speedup it?
            if self.probType != 'ODE':
                Keys = set(Tmp.keys()).difference(probDep)
                for key in Keys:
                    Tmp.pop(key)

            self.x0 = Tmp
            self._categoricalVars = set()
            for key, val in self.x0.items():
                #if key.domain is not None and key.domain is not bool and key.domain is not 'bool':
                if type(val) in (str, string_):
                    self._categoricalVars.add(key)
                    key.formAuxDomain()
                    self.x0[key] = searchsorted(key.aux_domain, val, 'left')
                elif key.domain is not None and key.domain is not bool and key.domain is not 'bool' and key.domain is not int and val not in key.domain:
                    self.x0[key] = key.domain[0]
            
            self.x0 = oopoint(self.x0)
            self.x0.maxDistributionSize = self.maxDistributionSize
            
            if self.probType in ['LP', 'MILP'] and self.f.getOrder(self.freeVars, self.fixedVars) > 1:
                self.err('for LP/MILP objective function has to be linear, while this one ("%s") is not' % self.f.name)

            setStartVectorAndTranslators(self)
            
            if self.fixedVars is None or (self.freeVars is not None and len(self.freeVars)<len(self.fixedVars)):
                D_kwargs = {'Vars':self.freeVars}
            else:
                D_kwargs = {'fixedVars':self.fixedVars}
            D_kwargs['useSparse'] = self.useSparse
            D_kwargs['fixedVarsScheduleID'] = self._FDVarsID
            D_kwargs['exactShape'] = True
            
            self._D_kwargs = D_kwargs
            
            variableTolerancesDict = dict([(v, v.tol) for v in self._freeVars])
            self.variableTolerances = self._point2vector(variableTolerancesDict)
            
            #Z = self._vector2point(zeros(self.n))
            if len(self._fixedVars) < len(self._freeVars) and 'isdisjoint' in dir(set()):
                areFixed = lambda dep: dep.issubset(self._fixedVars)
                #isFixed = lambda v: v in self._fixedVars
                Z = dict([(v, zeros_like(self._x0[v]) if v not in self._fixedVars else self._x0[v]) for v in self._x0.keys()])
            else:
                areFixed = lambda dep: dep.isdisjoint(self._freeVars)
                #isFixed = lambda v: v not in self._freeVars
                Z = dict([(v, zeros_like(self._x0[v]) if v in self._freeVars else self._x0[v]) for v in self._x0.keys()])
            Z = oopoint(Z, maxDistributionSize = self.maxDistributionSize)
           
            #p.isFixed = isFixed
            lb, ub = -inf*ones(self.n), inf*ones(self.n)

            # TODO: get rid of start c, h = None, use [] instead
            A, b, Aeq, beq = [], [], [], []
            
            if type(self.constraints) not in (list, tuple, set):
                self.constraints = [self.constraints]
            oovD = self._oovarsIndDict
            LB = {}
            UB = {}
            
            """                                    gather attached constraints                                    """
            
            C = list(self.constraints)
            self.constraints = set(self.constraints)
            for v in self._x0.keys():
                if not array_equal(v.lb, -inf):
                    self.constraints.add(v >= v.lb)
                if not array_equal(v.ub, inf):
                    self.constraints.add(v <= v.ub)
            
            if hasattr(self, 'f'):
                if type(self.f) in [list, tuple, set]:
                    C += list(self.f)
                else: # self.f is oofun
                    C.append(self.f)
            
            if self.useAttachedConstraints: 
                self.constraints.update(_getAllAttachedConstraints(C))
                
            FF = self.constraints.copy()
            for _F in F:
                if isinstance(_F, (tuple, list, ndarray, set)):
                    FF.update(_F)
                else:
                    FF.add(_F)

            unvectorizableFuncs = set()
            
            #unvectorizableVariables = set([var for var, val in self._x0.items() if isinstance(val, _Stochastic) or asarray(val).size > 1])
            
            # TODO: use this
            unvectorizableVariables = set([])
            
            # temporary replacement:
            #unvectorizableVariables = set([var for var, val in self._x0.items() if asarray(val).size > 1])
            
            
            cond = False
            #debug
#            unvectorizableVariables = set(self._x0.keys())
#            hasVectorizableFuncs = False
#            cond = True
            #debug end

            if 1 and isPyPy:
                hasVectorizableFuncs = False
                unvectorizableFuncs = FF
            else:
                hasVectorizableFuncs = False
                if len(unvectorizableVariables) != 0:
                    for ff in FF:
                        _dep = ff._getDep()
                        if cond or len(_dep & unvectorizableVariables) != 0:
                            unvectorizableFuncs.add(ff)
                        else:
                            hasVectorizableFuncs = True
                else:
                    hasVectorizableFuncs = True
            
            self.unvectorizableFuncs = unvectorizableFuncs
            self.hasVectorizableFuncs = hasVectorizableFuncs
            
            for v in self._freeVars:
                d = v.domain
                if d is bool or d is 'bool':
                    self.constraints.update([v>0, v<1])
                elif d is not None and d is not int and d is not 'int':
                    # TODO: mb add integer domains?
                    v.domain = array(list(d))
                    v.domain.sort()
                    self.constraints.update([v >= v.domain[0], v <= v.domain[-1]])
                    if hasattr(v, 'aux_domain'):
                         self.constraints.add(v - (len(v.aux_domain)-1)<=0)
                    
#            for v in self._categoricalVars:
#                if isFixed(v):
#                    ind = searchsorted(v.aux_domain, p._x0[v], 'left')
#                    if v.aux_domain

            """                                         handling constraints                                         """
            StartPointVars = set(self._x0.keys())
            self.dictOfFixedFuncs = {}
            from FuncDesigner import broadcast
            if self.probType in ['SLE', 'NLSP', 'SNLE', 'LLSP']:
                for eq in equations:
                    broadcast(formDictOfFixedFuncs, eq, self.useAttachedConstraints, self.dictOfFixedFuncs, areFixed, self._x0)
            else:
                broadcast(formDictOfFixedFuncs, self.f, self.useAttachedConstraints, self.dictOfFixedFuncs, areFixed, self._x0)

            if oosolver(self.solver).useLinePoints:
                self._firstLinePointDict = {}
                self._secondLinePointDict = {}
                self._currLinePointDict = {}
            inplaceLinearRender = oosolver(self.solver).__name__ == 'interalg'
            
            if inplaceLinearRender and hasattr(self, 'f'):
                D_kwargs2 = D_kwargs.copy()
                D_kwargs2['useSparse'] = False
                if type(self.f) in [list, tuple, set]:
                    ff = []
                    for f in self.f:
                        if f.getOrder(self.freeVars, self.fixedVars) < 2:
                            D = f.D(Z, **D_kwargs2)
                            f2 = linear_render(f, D, Z)
                            ff.append(f2)
                        else:
                            ff.append(f)
                    self.f = ff
                else: # self.f is oofun
                    if self.f.getOrder(self.freeVars, self.fixedVars) < 2:
                        D = self.f.D(Z, **D_kwargs2)
                        self.f = linear_render(self.f, D, Z)
                        if self.isObjFunValueASingleNumber:
                            self._linear_objective = True
                            self._linear_objective_factor = self._pointDerivative2array(D).flatten()
                            self._linear_objective_scalar = self.f(Z)
                                
            handleConstraint_args = (StartPointVars, areFixed, oovD, A, b, Aeq, beq, Z, D_kwargs, LB, UB, inplaceLinearRender)
            for c in self.constraints:
                if isinstance(c, ooarray):
                    for elem in c: 
                        self.handleConstraint(elem, *handleConstraint_args) 
                elif c is True:
                    continue
                elif c is False:
                    self.err('one of elements from constraints is "False", solution is impossible')
                elif not hasattr(c, 'isConstraint'): 
                    self.err('The type ' + str(type(c)) + ' is inappropriate for problem constraints')
                else:
                    self.handleConstraint(c, *handleConstraint_args)
                    
            if len(b) != 0:
                self.A, self.b = Vstack(A), Hstack(b)
                if hasattr(self.b, 'toarray'): self.b = self.b.toarray()
            if len(beq) != 0:
                self.Aeq, self.beq = Vstack(Aeq), Hstack(beq)
                if hasattr(self.beq, 'toarray'): self.beq = self.beq.toarray()
            for vName, vVal in LB.items():
                inds = oovD[vName]
                lb[inds[0]:inds[1]] = vVal
            for vName, vVal in UB.items():
                inds = oovD[vName]
                ub[inds[0]:inds[1]] = vVal
            self.lb, self.ub = lb, ub
        else: # not FuncDesigner
            if self.fixedVars is not None or self.freeVars is not None:
                self.err('fixedVars and freeVars are valid for optimization of FuncDesigner models only')
        if self.x0 is None: 
            arr = ['lb', 'ub']
            if self.probType in ['LP', 'MILP', 'QP', 'SOCP', 'SDP']: arr.append('f')
            if self.probType in ['LLSP', 'LLAVP', 'LUNP']: arr.append('D')
            for fn in arr:
                if not hasattr(self, fn): continue
                fv = asarray(getattr(self, fn))
                if any(isfinite(fv)):
                    self.x0 = zeros(fv.size)
                    break
        self.x0 = ravel(self.x0)
        
        if not hasattr(self, 'n'): self.n = self.x0.size
        if not hasattr(self, 'lb'): self.lb = -inf * ones(self.n)
        if not hasattr(self, 'ub'): self.ub =  inf * ones(self.n)        

        for fn in ('A', 'Aeq'):
            fv = getattr(self, fn)
            if fv is not None:
                #afv = asfarray(fv) if not isspmatrix(fv) else fv.toarray() # TODO: omit casting to dense matrix
                afv = asfarray(fv)  if type(fv) in [list, tuple] else fv
                if len(afv.shape) > 1:
                    if afv.shape[1] != self.n:
                        self.err('incorrect ' + fn + ' size')
                else:
                    if afv.shape != () and afv.shape[0] == self.n: afv = afv.reshape(1, self.n)
                setattr(self, fn, afv)
            else:
                setattr(self, fn, asfarray([]).reshape(0, self.n))
                
        nA, nAeq = prod(self.A.shape), prod(self.Aeq.shape) 
        SizeThreshold = 2 ** 15
        if scipyInstalled:
            from scipy.sparse import csc_matrix
            if isspmatrix(self.A) or (nA > SizeThreshold  and flatnonzero(self.A).size < 0.25*nA):
                self._A = csc_matrix(self.A)
            if isspmatrix(self.Aeq) or (nAeq > SizeThreshold and flatnonzero(self.Aeq).size < 0.25*nAeq):
                self._Aeq = csc_matrix(self.Aeq)
            
        elif nA > SizeThreshold or nAeq > SizeThreshold:
            self.pWarn(scipyAbsentMsg)
            
        self._baseProblemIsPrepared = True

    def handleConstraint(self, c, StartPointVars, areFixed, oovD, A, b, Aeq, beq, Z, D_kwargs, LB, UB, inplaceLinearRender):
        #import FuncDesigner as fd
        from FuncDesigner.ooFun import SmoothFDConstraint, BooleanOOFun
        if not isinstance(c, SmoothFDConstraint) and isinstance(c, BooleanOOFun): 
            self.hasLogicalConstraints = True
            #continue
        probtol = self.contol
        f, tol = c.oofun, c.tol
        _lb, _ub = c.lb, c.ub
        #f0, lb_0, ub_0 = f, copy(_lb), copy(_ub)
        lb_0, ub_0 = copy(_lb), copy(_ub)
        Name = f.name
        
        dep = set([f]) if f.is_oovar else f._getDep()
        
        isFixed = areFixed(dep)

        if f.is_oovar and isFixed:  
            if self._x0 is None or f not in self._x0: 
                self.err('your problem has fixed oovar '+ Name + ' but no value for the one in start point is provided')
            return True
            
        if not dep.issubset(StartPointVars):
            self.err('your start point has no enough variables to define constraint ' + c.name)

        if tol < 0:
            if any(_lb  == _ub):
                self.err("You can't use negative tolerance for the equality constraint " + c.name)
            elif any(_lb - tol >= _ub + tol):
                self.err("You can't use negative tolerance for so small gap in constraint" + c.name)

            Shift = (1.0+1e-13)*probtol 
            #######################
            # not inplace modification!!!!!!!!!!!!!
            _lb = _lb + Shift
            _ub = _ub - Shift
            #######################
        
        if tol != 0: self.useScaledResidualOutput = True
        
        # TODO: omit it for interalg
        if tol not in (0, probtol, -probtol):
            scaleFactor = abs(probtol / tol)
            
            f *= scaleFactor
            #c.oofun = f#c.oofun * scaleFactor
            _lb, _ub = _lb * scaleFactor, _ub * scaleFactor
            Contol = tol
            Contol2 = Contol * scaleFactor
        else:
            Contol = asscalar(copy(probtol))
            Contol2 = Contol 
            
        if isFixed:
            # TODO: get rid of self.contol, use separate contols for each constraint
            
            if not c(self._x0, tol=Contol2):
                s = """'constraint "%s" with all-fixed optimization variables it depends on is infeasible in start point, 
                hence the problem is infeasible, maybe you should change start point'""" % c.name
                self.err(s)
            return True

        from FuncDesigner import broadcast
        broadcast(formDictOfFixedFuncs, f, self.useAttachedConstraints, self.dictOfFixedFuncs, areFixed, self._x0)
            #self.dictOfFixedFuncs[f] = f(self.x0)

        f_order = f.getOrder(self.freeVars, self.fixedVars)
        if self.probType in ['LP', 'MILP', 'LLSP', 'LLAVP'] and f_order > 1:
            self.err('for LP/MILP/LLSP/LLAVP all constraints have to be linear, while ' + f.name + ' is not')
        
        if not f.is_oovar and f_order < 2:
            D_kwargs2 = D_kwargs.copy()
            D_kwargs2['useSparse'] = False
            D = f.D(Z, **D_kwargs2)
            if inplaceLinearRender:
                # interalg only
                if any([asarray(val).size > 1 for val in D.values()]):
                    self.err('currently interalg can handle only FuncDesigner.oovars(n), not FuncDesigner.oovar() with size > 1')
                f = linear_render(f, D, Z)
        else:
            D = 0
        
        # TODO: simplify condition of box-bounded oovar detection
        if f.is_oovar:
            inds = oovD[f]
            f_size = inds[1] - inds[0]

            if any(isfinite(_lb)):
                if _lb.size not in (f_size, 1): 
                    self.err('incorrect size of lower box-bound constraint for %s: 1 or %d expected, %d obtained' % (Name, f_size, _lb.size))
                    
                # for PyPy compatibility
                if type(_lb) == ndarray and _lb.size == 1:
                    _lb = _lb.item()
                
                val = array(f_size*[_lb] if type(_lb) == ndarray and _lb.size < f_size else _lb)
                if f not in LB:
                    LB[f] = val
                else:
                    #max((val, LB[f])) doesn't work for arrays
                    if val.size > 1 or LB[f].size > 1:
                        LB[f][val > LB[f]] = val[val > LB[f]] if val.size > 1 else asscalar(val)
                    else:
                        LB[f] = max((val, LB[f]))

            if any(isfinite(_ub)):
                if _ub.size not in (f_size, 1): 
                    self.err('incorrect size of upper box-bound constraint for %s: 1 or %d expected, %d obtained' % (Name, f_size, _ub.size))
                
                # for PyPy compatibility
                if type(_ub) == ndarray and _ub.size == 1:
                    _ub = _ub.item()
                    
                val = array(f_size*[_ub] if type(_ub) == ndarray and _ub.size < f_size else _ub)
                if f not in UB:
                    UB[f] = val
                else:
                    #min((val, UB[f])) doesn't work for arrays
                    if val.size > 1 or UB[f].size > 1:
                        UB[f][val < UB[f]] = val[val < UB[f]] if val.size > 1 else asscalar(val)
                    else:
                        UB[f] = min((val, UB[f]))
                    
        elif _lb == _ub:
            if f_order < 2:
                Aeq.append(self._pointDerivative2array(D))      
                beq.append(-f(Z)+_lb)
            elif self.h is None: self.h = [f-_lb]
            else: self.h.append(f-_lb)
        elif isfinite(_ub):
            if f_order < 2:
                A.append(self._pointDerivative2array(D))                       
                b.append(-f(Z)+_ub)
            elif self.c is None: self.c = [f - _ub]
            else: self.c.append(f - _ub)
        elif isfinite(_lb):
            if f_order < 2:
                A.append(-self._pointDerivative2array(D))                       
                b.append(f(Z) - _lb)                        
            elif self.c is None: self.c = [- f + _lb]
            else: self.c.append(- f + _lb)
        else:
            self.err('inform OpenOpt developers of the bug')
            
        if not f.is_oovar:
            Contol = max((0, Contol2))
            # TODO: handle it more properly, especially  for lb, ub of array type
            # FIXME: name of f0 vs f
#            self._FD.nonBoxConsWithTolShift.append((f0, lb_0 - Contol, ub_0 + Contol))
#            self._FD.nonBoxCons.append((f0, lb_0, ub_0, Contol))
            self._FD.nonBoxConsWithTolShift.append((c, f, _lb - Contol, _ub + Contol))
            self._FD.nonBoxCons.append((c, f, _lb, _ub, Contol))
#            if tol not in (0, probtol, -probtol):
#                print('!', f, _lb, _ub, Contol)
        return False

def formDictOfFixedFuncs(oof, dictOfFixedFuncs, areFixed, startPoint):
    dep = set([oof]) if oof.is_oovar else oof._getDep()
    if areFixed(dep):
        dictOfFixedFuncs[oof] = oof(startPoint)


class MatrixProblem(baseProblem):
    _baseClassName = 'Matrix'
    ftol = 1e-8
    contol = 1e-8
    #obsolete, should be removed
    # still it is used by lpSolve
    # Awhole * x {<= | = | >= } b
    Awhole = None # matrix m x n, n = len(x)
    bwhole = None # vector, size = m x 1
    dwhole = None #vector of descriptors, size = m x 1
    # descriptors dwhole[j] should be :
    # 1 : <Awhole, x> [j] greater (or equal) than bwhole[j]
    # -1 : <Awhole, x> [j] less (or equal) than bwhole[j]
    # 0 : <Awhole, x> [j] = bwhole[j]
    def __init__(self, *args, **kwargs):
        baseProblem.__init__(self, *args, **kwargs)
        self.kernelIterFuncs = setDefaultIterFuncs('Matrix')

    def _Prepare(self):
        if self.prepared == True:
            return
        baseProblem._prepare(self)
        self.prepared = True

    # TODO: move the function to child classes
    def _isUnconstrained(self):
        if  self.b.size !=0 or self.beq.size != 0: 
            return False
        
        # for PyPy compatibility
        if any(atleast_1d(self.lb) != -inf) or any(atleast_1d(self.ub) != inf):
            return False
            
        return True


class Parallel:
    def __init__(self):
        self.f = False# 0 - don't use parallel calclations, 1 - use
        self.c = False
        self.h = False
        #TODO: add paralell func!
        #self.parallel.fun = dfeval

class Args:
    def __init__(self): pass
    f, c, h = (), (), ()

class NonLinProblem(baseProblem, nonLinFuncs, Args):
    _baseClassName = 'NonLin'
    diffInt = ProbDefaults['diffInt']        #finite-difference gradient aproximation step
    #non-linear constraints
    c = None # c(x)<=0
    h = None # h(x)=0
    #lines with |info_user-info_numerical| / (|info_user|+|info_numerical+1e-15) greater than maxViolation will be shown
    maxViolation = 1e-2
    JacobianApproximationStencil = 1
    def __init__(self, *args, **kwargs):
        baseProblem.__init__(self, *args, **kwargs)
        if not hasattr(self, 'args'): self.args = Args()
        self.prevVal = {}
        for fn in ['f', 'c', 'h', 'df', 'dc', 'dh', 'd2f', 'd2c', 'd2h']:
            self.prevVal[fn] = {'key':None, 'val':None}

        self.functype = {}

        #self.isVectoriezed = False

#        self.fPattern = None
#        self.cPattern = None
#        self.hPattern = None
        self.kernelIterFuncs = setDefaultIterFuncs('NonLin')

    def checkdf(self, *args,  **kwargs):
        return self.checkGradient('df', *args,  **kwargs)

    def checkdc(self, *args,  **kwargs):
        return self.checkGradient('dc', *args,  **kwargs)

    def checkdh(self, *args,  **kwargs):
        return self.checkGradient('dh', *args,  **kwargs)
    
    def checkGradient(self, funcType, *args,  **kwargs):
        self._Prepare()
        if not DerApproximatorIsInstalled:
            self.err('To perform gradients check you should have DerApproximator installed, see http://openopt.org/DerApproximator')
        
        if not getattr(self.userProvided, funcType):
            self.warn("you haven't analitical gradient provided for " + funcType[1:] + ', turning derivatives check for it off...')
            return
        if len(args)>0:
            if len(args)>1 or 'x' in kwargs:
                self.err('checkd<func> funcs can have single argument x only (then x should be absent in kwargs )')
            xCheck = asfarray(args[0])
        elif 'x' in kwargs:
            xCheck = asfarray(kwargs['x'])
        else:
            xCheck = asfarray(self.x0)
        
        maxViolation = 0.01
        if 'maxViolation' in kwargs:
            maxViolation = kwargs['maxViolation']
            
        self.disp(funcType + (': checking user-supplied gradient of shape (%d, %d)' % (getattr(self, funcType[1:])(xCheck).size, xCheck.size)))
        self.disp('according to:')
        self.disp('    diffInt = ' + str(self.diffInt)) # TODO: ADD other parameters: allowed epsilon, maxDiffLines etc
        self.disp('    |1 - info_user/info_numerical| < maxViolation = '+ str(maxViolation))        
        
        check_d1(getattr(self, funcType[1:]), getattr(self, funcType), xCheck, **kwargs)
        
        # reset counters that were modified during check derivatives
        self.nEvals[funcType[1:]] = 0
        self.nEvals[funcType] = 0
        
    def _makeCorrectArgs(self):
        argslist = dir(self.args)
        if not ('f' in argslist and 'c' in argslist and 'h' in argslist):
            tmp, self.args = self.args, autocreate()
            self.args.f = self.args.c = self.args.h = tmp
        for j in ('f', 'c', 'h'):
            v = getattr(self.args, j)
            if type(v) != type(()): setattr(self.args, j, (v,))

    def __finalize__(self):
        #BaseProblem.__finalize__(self)
        if self.isFDmodel:
            self.xf = self._vector2point(self.xf)

    def _Prepare(self):
        baseProblem._prepare(self)
        if asarray(self.implicitBounds).size == 1:
            self.implicitBounds = [-self.implicitBounds, self.implicitBounds]
            self.implicitBounds.sort()# for more safety, maybe user-provided value is negative
        if hasattr(self, 'solver'):
            if not self.solver.iterfcnConnected:
                if self.solver.funcForIterFcnConnection == 'f':
                    if not hasattr(self, 'f_iter'):
                        self.f_iter = max((self.n, 4))
                else:
                    if not hasattr(self, 'df_iter'):
                        self.df_iter = True
        
        if self.prepared == True:
            return
            
        
        
        # TODO: simplify it
        self._makeCorrectArgs()
        for s in ('f', 'df', 'd2f', 'c', 'dc', 'd2c', 'h', 'dh', 'd2h'):
            derivativeOrder = len(s)-1
            self.nEvals[Copy(s)] = 0
            Attr = getattr(self, s, None)
            if Attr is not None and (not isinstance(Attr, (list, tuple)) or len(Attr) != 0) :
                setattr(self.userProvided, s, True)

                A = getattr(self,s)

                if type(A) not in [list, tuple]: #TODO: add or ndarray(A)
                    A = (A,)#make tuple
                setattr(self.user, s, A)
            else:
                setattr(self.userProvided, s, False)
            if derivativeOrder == 0:
                setattr(self, s, lambda x, IND=None, userFunctionType= s, ignorePrev=False, getDerivative=False: \
                        self.wrapped_func(x, IND, userFunctionType, ignorePrev, getDerivative))
                
#                setattr(self, s, lambda x, IND=None, userFunctionType= s, ignorePrev=False, getDerivative=False, \
#                        _linePointDescriptor = None: \
#                        self.wrapped_func(x, IND, userFunctionType, ignorePrev, getDerivative, _linePointDescriptor))
            elif derivativeOrder == 1:
                setattr(self, s, lambda x, ind=None, funcType=s[-1], ignorePrev = False, useSparse=self.useSparse:
                        self.wrapped_1st_derivatives(x, ind, funcType, ignorePrev, useSparse))
            elif derivativeOrder == 2:
                setattr(self, s, getattr(self, 'user_'+s))
            else:
                self.err('incorrect non-linear function case')

        self.diffInt = ravel(self.diffInt)
        
        # TODO: mb get rid of the field
        self.vectorDiffInt = self.diffInt.size > 1
        
        if self.scale is not None:
            self.scale = ravel(self.scale)
            if self.vectorDiffInt or self.diffInt[0] != ProbDefaults['diffInt']:
                self.info('using both non-default scale & diffInt is not recommended. diffInt = diffInt/scale will be used')
            self.diffInt = self.diffInt / self.scale
       

        #initialization, getting nf, nc, nh etc:
        for s in ['c', 'h', 'f']:
            if not getattr(self.userProvided, s):
                setattr(self, 'n'+s, 0)
            else:
                setNonLinFuncsNumber(self,  s)
                
        self.prepared = True

    # TODO: move the function to child classes
    def _isUnconstrained(self):
#        s = ((), [], array([]), None)
#        print '1:',all(isinf(self.lb))
#        print self.b.size,self.beq.size
        return self.b.size ==0 and self.beq.size==0 and not self.userProvided.c and not self.userProvided.h \
            and (len(self.lb)==0 or all(isinf(self.lb))) and (len(self.ub)==0 or all(isinf(self.ub)))
    

def minimize(p, *args, **kwargs):
    if 'goal' in kwargs:
        if kwargs['goal'] in ['min', 'minimum']:
            p.warn("you shouldn't pass 'goal' to the function 'minimize'")
        else:
            p.err('ambiguous goal has been requested: function "minimize", goal: %s' %  kwargs['goal'])
    p.goal = 'minimum'
    return runProbSolver(p, *args, **kwargs)

def maximize(p, *args, **kwargs):
    if 'goal' in kwargs:
        if kwargs['goal'] in ['max', 'maximum']:
            p.warn("you shouldn't pass 'goal' to the function 'maximize'")
        else:
            p.err('ambiguous goal has been requested: function "maximize", goal: %s' %  kwargs['goal'])
    p.goal = 'maximum'
    return runProbSolver(p, *args, **kwargs)            


def linear_render(f, D, Z):
    import FuncDesigner as fd
    if f.is_oovar: 
        return f
    ff = f(Z)
    name, tol, _id = f.name, f.tol, f._id
    f = fd.sum([v * (val if type(val) != ndarray or val.ndim < 2 else val.flatten()) for v, val in D.items()]) \
    + (ff if isscalar(ff) or ff.ndim <= 1 else asscalar(ff))
    f.name, f.tol, f._id = name, tol, _id
    return f
