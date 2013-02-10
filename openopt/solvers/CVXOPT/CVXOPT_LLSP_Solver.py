from numpy import asfarray,  ones, all, isfinite, copy, nan, concatenate, dot
from openopt.kernel.ooMisc import WholeRepr2LinConst, xBounds2Matrix, xBounds2cvxoptMatrix
from cvxopt_misc import *
import cvxopt.solvers as cvxopt_solvers

def CVXOPT_LLSP_Solver(p, solverName):
    cvxopt_solvers.options['maxiters'] = p.maxIter
    cvxopt_solvers.options['feastol'] = p.contol    
    cvxopt_solvers.options['abstol'] = p.ftol
    if p.iprint <= 0: 
        cvxopt_solvers.options['show_progress'] = False
        cvxopt_solvers.options['MSK_IPAR_LOG'] = 0
    xBounds2cvxoptMatrix(p)
    
    
    #f = copy(p.f).reshape(-1,1)
    #H = 
    
    sol = cvxopt_solvers.qp(Matrix(p.H), Matrix(p.f), Matrix(p.A), Matrix(p.b), Matrix(p.Aeq), Matrix(p.beq), solverName)
    
    p.msg = sol['status']
    if p.msg == 'optimal' :  p.istop = 1000
    else: p.istop = -100
    
    
    if sol['x'] is not None:
        p.xf = xf = asfarray(sol['x']).flatten()
        p.ff = asfarray(0.5*dot(xf, dot(p.H, xf)) + p.dotmult(p.f, xf).sum()).flatten()
        p.duals = concatenate((asfarray(sol['y']).flatten(), asfarray(sol['z']).flatten()))
    else:
        p.ff = nan
        p.xf = nan*ones(p.n)
