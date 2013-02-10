from numpy import eye
def pointProjection(x,  lb, ub, A, b, Aeq, beq):
    from openopt import QP
    # projection of x to set of linear constraints
    n = x.size
    # TODO: INVOLVE SPARSE CVXOPT MATRICES
    p = QP(H = eye(n), f = -x, A = A, b=b, Aeq=Aeq, beq=beq, lb=lb, ub=ub)
    #r = p.solve('cvxopt_qp', iprint = -1)
    r = p.solve('nlp:scipy_slsqp', contol = 1e-8, iprint = -1)
    return r.xf
