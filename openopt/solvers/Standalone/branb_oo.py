from openopt.kernel.baseSolver import baseSolver
from openopt.kernel.ooMisc import isSolved
from openopt.kernel.setDefaultIterFuncs import IS_NAN_IN_X, SMALL_DELTA_X, SMALL_DELTA_F
#from numpy import asarray, inf, ones, nan
from numpy import *
#import openopt
from openopt import NLP, OpenOptException

class branb(baseSolver):
    __name__ = 'branb'
    __license__ = "BSD"
    __authors__ = "Ingar Solberg, Institutt for teknisk kybernetikk, Norges Tekniske Hrgskole, Norway, translated to Python by Dmitrey"
    __homepage__ = ''
    __alg__ = "branch-and-cut (currently the implementation is quite primitive)"
    __info__ = ''
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'discreteVars', 'c', 'h']
    iterfcnConnected = True
    __isIterPointAlwaysFeasible__ = lambda self, p: True
    #eps = 1e-7
    nlpSolver = None

    
    def __init__(self): pass

    def __solver__(self, p):
        if self.nlpSolver is None: p.err('you should explicitely provide parameter nlpSolver (name of NLP solver to use for NLP subproblems)')
        
        # TODO: check it
        # it can be removed in other place during prob preparation
        for key in [IS_NAN_IN_X, SMALL_DELTA_X, SMALL_DELTA_F]:
            if key in p.kernelIterFuncs.keys():
                p.kernelIterFuncs.pop(key)
        
        p.nlpSolver = self.nlpSolver
        startPoint = p.point(p.x0)
        startPoint._f = inf
        fPoint = fminconset(p, startPoint, p)
        p.iterfcn(fPoint)
        p.istop = 1000
        
        
def fminconset(p_current, bestPoint, p):
    p2 = milpTransfer(p)
    p2.lb, p2.ub = p_current.lb, p_current.ub
    
    
    try:
        r = p2.solve(p.nlpSolver)
        curr_NLP_Point = p.point(r.xf)
        curr_NLP_Point._f = r.ff
    except OpenOptException:
        r = None
    
    resultPoint = p.point(bestPoint.x)
    if r is None or r.istop <0 or curr_NLP_Point.f() >= bestPoint.f(): 
        resultPoint._f  = inf
        return resultPoint
    elif r.istop == 0:
        pass# TODO: fix it
    
    # check if all discrete constraints are satisfied
    x = curr_NLP_Point.x
    k = -1
    for i in p.discreteVars.keys():#range(m):	# check x-vector
        # TODO: replace it by "for i, val in dict.itervalues()"
        if not any(abs(x[i] - p.discreteVars[i]) < p.discrtol):
            k=i	# Violation of this set constraint.
            break # Go and split for this x-component

    if k != -1:		# some discrete constraint violated => recursive search is required
        p.debugmsg('k='+str(k)+' x[k]=' + str(x[k]) + ' p.discreteVars[k]=' +str(p.discreteVars[k]))
        Above=where(p.discreteVars[k]>x[k])[0]
        Below=where(p.discreteVars[k]<x[k])[0]
        resultPoint._f = inf
    else:
        if curr_NLP_Point.f() < bestPoint.f():
            bestPoint = curr_NLP_Point
        p.iterfcn(curr_NLP_Point)
        if p.istop:
            if bestPoint.betterThan(curr_NLP_Point):
                p.iterfcn(bestPoint)
            raise isSolved
        return curr_NLP_Point#bestPoint#x, h, exitflag
    

    # Solve first subproblem if it exists
    if Below.size != 0:
        below = Below[-1]# index of largest set element below x(k)
        
        p3 = milpTransfer(p)
        p3.x0 = x.copy()
        
        ub1 = p_current.ub.copy()
        ub1[k] = min((ub1[k], p.discreteVars[k][below]))	# new upper bound on x[k] for 1st subproblem
        ub1[k] = p.discreteVars[k][below]
        p3.ub = ub1
        p3.lb = p_current.lb
        
        Point_B =  fminconset(p3, bestPoint, p)
        resultPoint = Point_B
        if p.discreteConstraintsAreSatisfied(Point_B.x) and Point_B.betterThan(bestPoint):
            bestPoint = Point_B
            #p.iterfcn(bestPoint)

    # Solve second subproblem if it exists
    if Above.size != 0:
        above = Above[0]# index of smallest set element above x(k)
        
        p4 = milpTransfer(p)
        p4.x0 = x.copy()
        
        lb1=p_current.lb.copy()
        lb1[k]=max((lb1[k], p.discreteVars[k][above]))# new upper bound on x[k] for 1st subproblem
        lb1[k] =  p.discreteVars[k][above]
        p4.lb = lb1
        p4.ub = p_current.ub
        
        Point_A =  fminconset(p4, bestPoint, p)
        resultPoint = Point_A
        if p.discreteConstraintsAreSatisfied(Point_A.x) and Point_A.betterThan(bestPoint):
            bestPoint = Point_A        
            #p.iterfcn(bestPoint)
    
    if Below.size != 0 and Above.size != 0:
        if Point_A.f() < Point_B.f():
            resultPoint = Point_A
        else:
            resultPoint = Point_B
            
    return resultPoint#bestPoint

def milpTransfer(originProb):
    newProb = NLP(originProb.f, originProb.x0)
    originProb.inspire(newProb)
    newProb.discreteVars = originProb.discreteVars
    def err(s): # to prevent text output
        raise OpenOptException(s)
    newProb.err = err
    for fn in ['df', 'd2f', 'c', 'dc', 'h', 'dh']:
        if hasattr(originProb, fn) and getattr(originProb.userProvided, fn) or originProb.isFDmodel:
            setattr(newProb, fn, getattr(originProb, fn))
    
    newProb.plot = 0
    newProb.iprint = -1
    newProb.nlpSolver = originProb.nlpSolver 
    return newProb


        
        
        
