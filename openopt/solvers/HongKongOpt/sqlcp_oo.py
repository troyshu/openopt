from numpy import isfinite, any
from openopt.kernel.baseSolver import *
from sqlcp import sqlcp as SQLCP
from numpy.linalg import LinAlgError

class sqlcp(baseSolver):
    __name__ = 'sqlcp'
    __license__ = "MIT"
    __authors__ = "Enzo Michelangeli"
    __alg__ = "an SQP implementation"
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub']
    iterfcnConnected = True
    #_canHandleScipySparse = True
    QPsolver=None
    __info__ = '''SQP solver. Approximates f in x0 with paraboloid with same gradient and hessian,
    then finds its minimum with a quadratic solver (qlcp by default) and uses it as new point, 
    iterating till changes in x and/or f drop below given limits. 
    Requires the Hessian to be definite positive.
    The Hessian is initially approximated by its principal diagonal, and then
    updated at every step with the BFGS method.
    
    By default it uses QP solver qlcp (license: MIT), however, latter uses LCP solver LCPSolve, that is "free for education" for now. 
    You can use other QP solver via "oosolver('sqlcp', QPsolver='cvxopt_qp')" or any other, including converters.
    
    Copyright (c) 2010 Enzo Michelangeli and IT Vision Ltd
    '''

    def __init__(self): pass
    def __solver__(self, p):
        (A, b) = (p.A, p.b) if p.nb else (None, None)
        (Aeq, beq) = (p.Aeq, p.beq) if p.nbeq else (None, None)
        lb = p.lb if any(isfinite(p.lb)) else None
        ub = p.ub if any(isfinite(p.ub)) else None
        def callback(x):
            p.iterfcn(x)
            return True if p.istop else False
        try:
            SQLCP(p.f, p.x0, df=p.df, A=A , b=b, Aeq=Aeq, beq=beq, lb=p.lb, ub=p.ub, minstep=p.xtol, minfchg=1e-15, qpsolver=self.QPsolver, callback = callback)
        except LinAlgError:
            p.msg = 'linalg error, probably failed to invert Hesse matrix'
            p.istop = -100
            
