from numpy import *
from openopt import *

def pointProjection(x,  lb, ub, A, b, Aeq, beq):
    # projection of x to set of linear constraints
    n = x.size
    # TODO: INVOLVE SPARSE CVXOPT MATRICES
    p = QP(H = eye(n), f = -x, A = A, b=b, Aeq=Aeq, beq=beq, lb=lb, ub=ub)
    r = p.solve('cvxopt_qp')
    return r.xf




if __name__ == '__main__':
    x = array((1, 2, 3))
    lb, ub = None, None
    A = [3, 4, 5]
    b = -15
    Aeq, beq = None, None
    proj = pointProjection(x,  lb, ub, A, b, Aeq, beq)
    print(proj)
