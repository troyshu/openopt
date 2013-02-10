__docformat__ = "restructuredtext en"
from numpy import *

#TODO: add stopcases -10,-11,-12, -13
SMALL_DF = 2
SMALL_DELTA_X = 3
SMALL_DELTA_F = 4
FVAL_IS_ENOUGH = 10
MAX_NON_SUCCESS = 11
USER_DEMAND_STOP = 80
BUTTON_ENOUGH_HAS_BEEN_PRESSED = 88
SOLVED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON = 1000



UNDEFINED = 0

IS_NAN_IN_X = -4
IS_LINE_SEARCH_FAILED = -5
IS_MAX_ITER_REACHED = -7
IS_MAX_CPU_TIME_REACHED = -8
IS_MAX_TIME_REACHED = -9
IS_MAX_FUN_EVALS_REACHED = -10
IS_ALL_VARS_FIXED = -11
FAILED_TO_OBTAIN_MOVE_DIRECTION = -13
USER_DEMAND_EXIT = -99

FAILED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON = -1000

def stopcase(arg):
    if hasattr(arg, 'istop'): istop = arg.istop
    else: istop = arg
    
    if istop > 0: return 1
    elif istop in [0, IS_MAX_ITER_REACHED, IS_MAX_CPU_TIME_REACHED, IS_MAX_TIME_REACHED, IS_MAX_FUN_EVALS_REACHED]: return 0
    else: return -1



def setDefaultIterFuncs(className):
    d = dict()

    # positive:
    d[SMALL_DF] = lambda p: small_df(p)
    d[SMALL_DELTA_X] = lambda p: small_deltaX(p)
    d[SMALL_DELTA_F] = lambda p: small_deltaF(p)
    d[FVAL_IS_ENOUGH] = lambda p: isEnough(p)
    #d[11] = lambda p: allVarsAreFixed(p)

    # negative:
    d[IS_NAN_IN_X]  = lambda p: isNanInX(p)
    d[IS_MAX_ITER_REACHED]  = lambda p: isMaxIterReached(p)
    d[IS_MAX_CPU_TIME_REACHED]  = lambda p: isMaxCPUTimeReached(p)
    d[IS_MAX_TIME_REACHED]  = lambda p: isMaxTimeReached(p)
    if className == 'NonLin': d[IS_MAX_FUN_EVALS_REACHED] = lambda p: isMaxFunEvalsReached(p)

    return d

def small_df(p):
    if not hasattr(p, '_df') or p._df is None: return False
    if hasattr(p._df, 'toarray'):p._df = p._df.toarray()
    if p.norm(p._df) >= p.gtol or not all(isfinite(p._df)) or not p.isFeas(p.iterValues.x[-1]): return False
    return (True, '|| gradient F(X[k]) || < gtol')
    #return False if not hasattr(p, 'dF') or p.dF == None or p.norm(p.dF) > p.gtol else True

def small_deltaX(p):
    if p.iter == 0: return False
    diffX = p.iterValues.x[-1] - p.iterValues.x[-2]
    if p.scale is not None: diffX *= p.scale
    if p.norm(diffX) >= p.xtol: return False
    else: return (True, '|| X[k] - X[k-1] || < xtol')
    #r = False if p.norm(p.xk - p.x_prev) > p.xtol else True


def small_deltaF(p):
    if p.iter == 0: return False
    #r = False if p.norm(p.fk - p.f_prev) > p.ftol else True
    if  isnan(p.iterValues.f[-1]) or isnan(p.iterValues.f[-2]) or p.norm(p.iterValues.f[-1] - p.iterValues.f[-2]) >= p.ftol: # or (p.iterValues.r[-1] > p.contol and p.iterValues.r[-1] - p.iterValues.r[-2] < p.contol):
            return False
    else: return (True, '|| F[k] - F[k-1] || < ftol')


def isEnough(p):
#    asscalar(asarray(p.Fk)) was added for compatibility with ironpython
    if p.fEnough > asscalar(asarray(p.Fk)) and p.isFeas(p.xk): #TODO: mb replace by p.rk<p.contol? or other field like p.iterValues.isFeasible?
       return (True, 'fEnough has been reached')
    else: return False

def isNanInX(p):
    if any(isnan(p.xk)):
       return (True, 'NaN in X[k] coords has been obtained')
    else: return False


def isMaxIterReached(p):
    if p.iter >= p.maxIter-1: # iter numeration starts from zero
        return (True, 'Max Iter has been reached')
    else: return False

def isMaxCPUTimeReached(p):
    if p.iterCPUTime[-1] < p.maxCPUTime + p.cpuTimeElapsedForPlotting[-1]:
        return False
    else:
        return (True, 'max CPU time limit has been reached')
    #return p.cpuTimeElapsed >= p.maxCPUTime

def isMaxTimeReached(p):
    if p.currtime - p.timeStart < p.maxTime + p.timeElapsedForPlotting[-1]:
        return False
    else:
        return (True, 'max time limit has been reached')

def isMaxFunEvalsReached(p):
    #if not hasattr(p, 'nFunEvals'): p.warn('no nFunEvals field'); return 0
    if p.nEvals['f'] >= p.maxFunEvals:
        return (True, 'max objfunc evals limit has been reached')
    else:
        return False


####################################################################
def denyingStopFuncs(ProblemGroup=None):
    #mb in future it will be class-dependend,  like Matrix,  Nonlinear etc
    return {isMinIterReached : 'min iter is not reached yet', isMinTimeReached : 'min time is not reached yet', isMinCPUTimeReached:'min cputime is not reached yet', isMinFunEvalsReached:'min objective function evaluations nuber is not reached yet'}

isMinFunEvalsReached = lambda p: p.minFunEvals==0 or ('f' in p.nEvals.keys() and p.nEvals['f'] >= p.minFunEvals)

isMinTimeReached = lambda p: p.currtime - p.timeStart > p.minTime + p.timeElapsedForPlotting[-1]

isMinCPUTimeReached = lambda p: p.iterCPUTime[-1] >= p.minCPUTime + p.cpuTimeElapsedForPlotting[-1]

isMinIterReached = lambda p: p.iter >= p.minIter

