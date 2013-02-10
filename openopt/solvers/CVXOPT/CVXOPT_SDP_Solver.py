from numpy import asarray,  ones, all, isfinite, copy, nan, concatenate, array, asfarray, zeros
from openopt.kernel.ooMisc import WholeRepr2LinConst, xBounds2Matrix
from cvxopt_misc import *
import cvxopt.solvers as cvxopt_solvers
from cvxopt.base import matrix
from openopt.kernel.setDefaultIterFuncs import SOLVED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON,  IS_MAX_ITER_REACHED, IS_MAX_TIME_REACHED, FAILED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON, UNDEFINED

def converter_to_CVXOPT_SDP_Matrices_from_OO_SDP_Class(OO_SDP_Class_2D_Dict_S,  nVars):
    # nVars can be extracted from OO_SDP_Class_2D_Dict_S but it's easier just to pass as param
    a = OO_SDP_Class_2D_Dict_S
    #r = []
    R = {}
    for i, j in a.keys():
        if i not in R.keys():
            R[i] = zeros((nVars, asarray(a[i, 0]).size))
        R[i][j] = asfarray(a[i, j]).flatten()
    r = []
    for i in R.keys():
        r.append(Matrix(R[i].T))
    return r
            
def DictToList(d):
    i = 0
    r = []
    while i in d.keys():
        r.append(matrix(d[i], tc = 'd'))
        i += 1
    return r

def CVXOPT_SDP_Solver(p, solverName):
    if solverName == 'native_CVXOPT_SDP_Solver': solverName = None
    cvxopt_solvers.options['maxiters'] = p.maxIter
    cvxopt_solvers.options['feastol'] = p.contol    
    cvxopt_solvers.options['abstol'] = p.ftol
    if p.iprint <= 0:
        cvxopt_solvers.options['show_progress'] = False
        cvxopt_solvers.options['LPX_K_MSGLEV'] = 0
        #cvxopt_solvers.options['MSK_IPAR_LOG'] = 0
    xBounds2Matrix(p)

    #FIXME: if problem is search for MAXIMUM, not MINIMUM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    f = copy(p.f).reshape(-1,1)

    # CVXOPT have some problems with x0 so currently I decided to avoid using the one

    sol = cvxopt_solvers.sdp(Matrix(p.f), Matrix(p.A), Matrix(p.b), \
                             converter_to_CVXOPT_SDP_Matrices_from_OO_SDP_Class(p.S, p.n), \
                             DictToList(p.d), Matrix(p.Aeq), Matrix(p.beq), solverName)
    p.msg = sol['status']
    if p.msg == 'optimal' :  p.istop = SOLVED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON
    else: p.istop = -100
    if sol['x'] is not None:
        p.xf = asarray(sol['x']).flatten()
        p.ff = sum(p.dotmult(p.f, p.xf))
        #p.duals = concatenate((asarray(sol['y']).flatten(), asarray(sol['z']).flatten()))
    else:
        p.ff = nan
        p.xf = nan*ones(p.n)
