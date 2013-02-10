import numpy as np

#from numpy.linalg import norm, solve#, LinAlgError
from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F, MAX_NON_SUCCESS, IS_NAN_IN_X
from openopt.kernel.baseSolver import *

from openopt.solvers.UkrOpt.interalgMisc import *
from FuncDesigner import sum as fd_sum, abs as fd_abs, oopoint
from ii_engine import *
from interalgCons import processConstraints
from interalgODE import interalg_ODE_routine

from interalgLLR import adjustr4WithDiscreteVariables

bottleneck_is_present = False
try:
    from bottleneck import nanmin
    bottleneck_is_present = True
except ImportError:
    from numpy import nanmin



class interalg(baseSolver):
    __name__ = 'interalg'
    __license__ = "BSD"
    __authors__ = "Dmitrey"
    __alg__ = ""
    __optionalDataThatCanBeHandled__ = ['lb', 'ub', 'c', 'h', 'A', 'Aeq', 'b', 'beq', 'discreteVars']
    iterfcnConnected = True
    fStart = None
    dataType = np.float64
    #maxMem = '150MB'
    maxNodes = 150000
    maxActiveNodes = 150
    sigma = 0.1 # for MOP, unestablished
    
    _requiresBestPointDetection = True
    
    __isIterPointAlwaysFeasible__ = lambda self, p: \
    p.__isNoMoreThanBoxBounded__() or p.probType in ('MOP', 'IP') #and p.probType != 'IP'
    _requiresFiniteBoxBounds = True

    def __init__(self): 
        self.dataHandling = 'auto'
        
    def __solver__(self, p):
        
        isMOP = p.probType == 'MOP'
        if isMOP:
            from interalgMOP import r14MOP
        #isOpt = p.probType in ['NLP', 'NSP', 'GLP', 'MINLP']
        isODE = p.probType == 'ODE'
        isSNLE = p.probType in ('NLSP', 'SNLE')

        if not p.__isFiniteBoxBounded__() and not isODE: 
            p.err('''
            solver %s requires finite lb, ub: 
            lb <= x <= ub 
            (you can use "implicitBoounds")
            ''' % self.__name__)
#        if p.fixedVars is not None:
#            p.err('solver %s cannot handle FuncDesigner problems with some variables declared as fixed' % self.__name__)
        if p.probType in ('LP', 'MILP'):
            p.err("the solver can't handle problems of type " + p.probType)
        if not p.isFDmodel:
            p.err('solver %s can handle only FuncDesigner problems' % self.__name__)
       
        dataType = self.dataType
        if type(dataType) == str:
            if not hasattr(np, dataType):
                p.pWarn('your architecture has no type "%s", float64 will be used instead' % dataType)
                dataType = 'float64'
            dataType = getattr(np, dataType)
            self.dataType = dataType
       
        isIP = p.probType == 'IP'
        if isIP:
            pb = r14IP
            p._F = asarray(0, self.dataType)
            p._residual = 0.0
            f_int = p.user.f[0].interval(p.domain, self.dataType)
            p._r0 = prod(p.ub-p.lb) * (f_int.ub - f_int.lb)
            p._volume = 0.0
            p.kernelIterFuncs.pop(IS_NAN_IN_X)
        elif isMOP:
            pb = r14MOP
        else:
            pb = r14

        
        for val in p._x0.values():
            if isinstance(val,  (list, tuple, np.ndarray)) and len(val) > 1:
                p.pWarn('''
                solver %s currently can handle only single-element variables, 
                use oovars(n) instead of oovar(size=n),
                elseware correct result is not guaranteed
                '''% self.__name__)
                
        vv = list(p._freeVarsList)
        x0 = dict([(v, p._x0[v]) for v in vv])
        
        for val in x0.values():
            if isinstance(val,  (list, tuple, np.ndarray)) and len(val) > 1:
                p.err('''
                solver %s currently can handle only single-element variables, 
                use oovars(n) instead of oovar(size=n)'''% self.__name__)

        point = p.point
        
        p.kernelIterFuncs.pop(SMALL_DELTA_X, None)
        p.kernelIterFuncs.pop(SMALL_DELTA_F, None)
        p.kernelIterFuncs.pop(MAX_NON_SUCCESS, None)
        
        if not bottleneck_is_present and not isODE:
                p.pWarn('''
                installation of Python module "bottleneck" 
                (http://berkeleyanalytics.com/bottleneck,
                available via easy_install, takes several minutes for compilation)
                could speedup the solver %s''' % self.__name__)
        
        n = p.n
        
        maxSolutions = p.maxSolutions
        if maxSolutions == 0: maxSolutions = 10**50
        if maxSolutions != 1 and p.fEnough != -np.inf:
            p.warn('''
            using the solver interalg with non-single solutions mode 
            is not ajusted with fEnough stop criterium yet, it will be omitted
            ''')
            p.kernelIterFuncs.pop(FVAL_IS_ENOUGH)
        
        nNodes = []        
        p.extras['nNodes'] = nNodes
        nActiveNodes = []
        p.extras['nActiveNodes'] = nActiveNodes

        Solutions = Solution()
        Solutions.maxNum = maxSolutions
        Solutions.solutions = []
        Solutions.coords = np.array([]).reshape(0, n)
        p.solutions = Solutions
        
        lb, ub = asarray(p.lb, dataType).copy(), asarray(p.ub, dataType).copy()

        fTol = p.fTol
        if isIP or isODE:
            if p.ftol is None:
                if fTol is not None:
                    p.ftol = fTol
                else:
                    p.err('interalg requires user-supplied ftol (required precision)')
            if fTol is None: fTol = p.ftol
            elif fTol != p.ftol:
                p.err('you have provided both ftol and fTol')

        if fTol is None and not isMOP: # TODO: require tols for MOP
            fTol = 1e-7
            p.warn('solver %s require p.fTol value (required objective function tolerance); 10^-7 will be used' % self.__name__)

        xRecord = 0.5 * (lb + ub)
        adjustr4WithDiscreteVariables(xRecord.reshape(1, -1), p)

        r40 = np.inf
        
        y = lb.reshape(1, -1)
        e = ub.reshape(1, -1)
        r41 = np.inf

        # TODO: maybe rework it, especially for constrained case
        fStart = self.fStart
        
        # TODO: remove it after proper SNLE handling implementation
        if isSNLE:
            r41 = 0.0
#            asdf1 = None
            eqs = [fd_abs(elem) for elem in p.user.f]
            asdf1 = fd_sum(eqs)
            
            # TODO: check it, for reducing calculations
            #C.update([elem == 0 for elem in p.user.f])
        elif isMOP:
            asdf1 = p.user.f
            Solutions.F = []
            if point(p.x0).isFeas(altLinInEq=False):
                Solutions.solutions.append(p.x0.copy())
                Solutions.coords = asarray(Solutions.solutions)
                Solutions.F.append(p.f(p.x0))
                p._solutions = Solutions
        elif not isODE:
            asdf1 = p.user.f[0]
            
            #if p.fOpt is not None:  fOpt = p.fOpt
            if p.goal in ('max', 'maximum'):
                asdf1 = -asdf1
                if p.fOpt is not None:
                    p.fOpt = -p.fOpt
            
            if fStart is not None and fStart < r40: 
                r41 = fStart
                
            for X0 in [point(xRecord), point(p.x0)]:
                if X0.isFeas(altLinInEq=False) and X0.f() < r40:
                    r40 = X0.f()

            if p.isFeas(p.x0):
                tmp = asdf1(p._x0)
                if  tmp < r41:
                    r41 = tmp
                
            if p.fOpt is not None:
                if p.fOpt > r41:
                    p.warn('user-provided fOpt seems to be incorrect, ')
                r41 = p.fOpt

#        if isSNLE:
#            if self.dataHandling == 'raw':
#                p.pWarn('''
#                    this interalg data handling approach ("%s") 
#                    is unimplemented for SNLE yet, dropping to "sorted"'''%self.dataHandling)
#            
#            # handles 'auto' as well
#            self.dataHandling ='sorted'

        domain = oopoint([(v, [p.lb[i], p.ub[i]]) for i,  v in enumerate(vv)], skipArrayCast=True)
        domain.dictOfFixedFuncs = p.dictOfFixedFuncs
        #from FuncDesigner.ooFun import BooleanOOFun, SmoothFDConstraint
        
        if self.dataHandling == 'auto':
            if isIP or isODE:
                self.dataHandling = 'sorted'
            elif isMOP or p.hasLogicalConstraints:
                self.dataHandling = 'raw'
            else:
                r = p.user.f[0].interval(domain, self.dataType)
                M = np.max((np.max(np.atleast_1d(np.abs(r.lb))), np.max(np.atleast_1d(np.abs(r.ub)))))
                for (c, func, lb, ub, tol) in p._FD.nonBoxCons:#[Elem[1] for Elem in p._FD.nonBoxCons]:

                    # !!!!!!!!!!!!!!!!!!!! check it - mb 2nd condition is incorrect
                    #if isinstance(c, BooleanOOFun) and not isinstance(c, SmoothFDConstraint): continue
                    if hasattr(c,'_unnamedBooleanOOFunNumber'):
                        continue
                    
                    r = func.interval(domain, self.dataType)
                    M = np.max((M, np.max(np.atleast_1d(np.abs(r.lb)))))
                    M = np.max((M, np.max(np.atleast_1d(np.abs(r.ub)))))
                self.dataHandling = 'raw' if M < 1e5 else 'sorted'
                    
            #self.dataHandling = 'sorted' if isIP or (p.__isNoMoreThanBoxBounded__() and n < 50) else 'raw'
            
        # TODO: is it required yet?
        if not isMOP and not p.hasLogicalConstraints:
            p._isOnlyBoxBounded = p.__isNoMoreThanBoxBounded__() 
            if isODE or (asdf1.isUncycled and p._isOnlyBoxBounded and np.all(np.isfinite(p.user.f[0].interval(domain).lb))):
                #maxNodes = 1
                self.dataHandling = 'sorted'
                
        if self.dataHandling == 'sorted' and p.hasLogicalConstraints:
            p.warn("interalg: for general logical constraints only dataHandling='raw' mode works")
            self.dataHandling = 'raw'

        self.maxActiveNodes = int(self.maxActiveNodes)
#        if self.maxActiveNodes < 2:
#            p.warn('maxActiveNodes should be at least 2 while you have provided %d. Setting it to 2.' % self.maxActiveNodes)
        self.maxNodes = int(self.maxNodes)

        _in = np.array([], object)
        
        g = np.inf
        C = p._FD.nonBoxConsWithTolShift
        C0 = p._FD.nonBoxCons
#        if isOpt:
#            r = []
#            for (elem, lb, ub, tol) in C0:
#                if tol == 0: tol = p.contol
#                if lb == ub:
#                    r.append(fd_max((fd_abs(elem-lb)-tol, 0)) * (fTol/tol))
#                elif lb == -inf:
#                    r.append(fd_max((0, elem-ub-tol)) * (fTol/tol))
#                elif ub == inf:
#                    r.append(fd_max((0, lb-elem-tol)) * (fTol/tol))
#                else:
#                    p.err('finite box constraints are unimplemented for interalg yet')
            #p._cons_obj = 1e100 * fd_sum(r) if len(r) != 0 else None
            #p._cons_obj = fd_sum(r) if len(r) != 0 else None

        if isSNLE:
            C += [(elem==0, elem, -(elem.tol if elem.tol != 0 else p.ftol), (elem.tol if elem.tol != 0 else p.ftol)) for elem in p.user.f]
            C0 += [(elem==0, elem, 0, 0, (elem.tol if elem.tol != 0 else p.ftol)) for elem in p.user.f]
        
        
        
        # TODO: hanlde fixed variables here
        varTols = p.variableTolerances
        if Solutions.maxNum != 1:
            if not isSNLE:
                p.err('''
                "search several solutions" mode is unimplemented
                for the prob type %s yet''' % p.probType)
            if any(varTols == 0):
                p.err('''
                for the mode "search all solutions" 
                you have to provide all non-zero tolerances 
                for each variable (oovar)
                ''')
            
        pnc = 0
        an = []
        maxNodes = self.maxNodes
        
        # TODO: change for constrained probs
        _s = atleast_1d(inf)
        
        if isODE or (isIP and p.n == 1):
            interalg_ODE_routine(p, self)
            return

        while 1:
            if len(C0) != 0: 
                y, e, nlhc, residual, definiteRange, indT, _s = processConstraints(C0, y, e, _s, p, dataType)
            else:
                nlhc, residual, definiteRange, indT = None, None, True, None
            
            if y.size != 0:
                an, g, fo, _s, Solutions, xRecord, r41, r40 = \
                pb(p, nlhc, residual, definiteRange, y, e, vv, asdf1, C, r40, g, \
                             nNodes, r41, fTol, Solutions, varTols, _in, \
                             dataType, maxNodes, _s, indT, xRecord)
                if _s is None:
                    break
            else:
                an = _in
                fo = 0.0 if isSNLE or isMOP else min((r41, r40 - (fTol if Solutions.maxNum == 1 else 0.0))) 
            pnc = max((len(np.atleast_1d(an)), pnc))
            
            if isIP:
                y, e, _in, _s = \
                    func12(an, self.maxActiveNodes, p, Solutions, vv, varTols, np.inf)
            else:
                y, e, _in, _s = \
                func12(an, self.maxActiveNodes, p, Solutions, vv, varTols, fo)

            nActiveNodes.append(y.shape[0]/2)
            if y.size == 0: 
                if len(Solutions.coords) > 1:
                    p.istop, p.msg = 1001, 'all solutions have been obtained'
                else:
                    p.istop, p.msg = 1000, 'solution has been obtained'
                break            
            ############# End of main cycle ###############
            
        if not isSNLE and not isIP and not isMOP:
            if p._bestPoint.betterThan(p.point(p.xk)):
                p.iterfcn(p._bestPoint)
            else:
                p.iterfcn(p.xk)
        
        ff = p.fk # ff may be not assigned yet
#        ff = p._bestPoint.f()
#        p.xk = p._bestPoint.x
        if isIP: 
            p.xk = np.array([np.nan]*p.n)
            p.rk = p._residual
            p.fk = p._F
        
        isFeas = len(Solutions.F) != 0 if isMOP else p.isFeas(p.xk) if not isIP else p.rk < fTol
        
        if not isFeas and p.istop > 0:
            p.istop, p.msg = -1000, 'no feasible solution has been obtained'
        
        o = asarray([t.o for t in an])
        if o.size != 0:
            g = nanmin([nanmin(o), g])

        if not isMOP:
            p.extras['isRequiredPrecisionReached'] = \
            True if ff - g < fTol and isFeas else False
            # and (k is False or (isSNLE and (p._nObtainedSolutions >= maxSolutions or maxSolutions==1))) 

        if not isMOP and not p.extras['isRequiredPrecisionReached'] and p.istop > 0:
            p.istop = -1
            p.msg = 'required precision is not guarantied'
            
        # TODO: simplify it
        if not isMOP:
            tmp = [nanmin(np.hstack((ff, g, o.flatten()))), np.asscalar(np.array(ff))]
            if p.goal in ['max', 'maximum']: tmp = (-tmp[1], -tmp[0])
            p.extras['extremumBounds'] = tmp if not isIP else 'unimplemented for IP yet'
        
        
        p.solutions = [p._vector2point(s) for s in Solutions.coords] if not isMOP else \
        MOPsolutions([p._vector2point(s) for s in Solutions.coords])
        if isMOP:
            for i, s in enumerate(p.solutions):
                s.useAsMutable = True
                for j, goal in enumerate(p.user.f):
                    s[goal] = Solutions.F[i][j]
                s.useAsMutable = False
            p.solutions.values = np.asarray(Solutions.F)
            p.solutions.coords = Solutions.coords
        if not isMOP and p.maxSolutions == 1: delattr(p, 'solutions')
        if isSNLE and p.maxSolutions != 1:
            for v in p._categoricalVars:
                for elem in r.solutions:
                    elem.useAsMutable = True
                    elem[v] = v.aux_domain[elem[v]]
                    elem.useAsMutable = False
        if p.iprint >= 0 and not isMOP:
#            s = 'Solution with required tolerance %0.1e \n is%s guarantied (obtained precision: %0.1e)' \
#                   %(fTol, '' if p.extras['isRequiredPrecisionReached'] else ' NOT', tmp[1]-tmp[0])
            s = 'Solution with required tolerance %0.1e \n is%s guarantied' \
            %(fTol, '' if p.extras['isRequiredPrecisionReached'] else ' NOT')
            if not isIP and p.maxSolutions == 1:
                s += ' (obtained precision: %0.1e)' % np.abs(tmp[1]-tmp[0])
            if not p.extras['isRequiredPrecisionReached'] and pnc == self.maxNodes: s += '\nincrease maxNodes (current value %d)' % self.maxNodes
            p.info(s)

class Solution:
    pass
    
class MOPsolutions(list):
    pass
    

