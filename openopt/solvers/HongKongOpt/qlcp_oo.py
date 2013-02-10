from openopt.kernel.baseSolver import baseSolver
from QPSolve import QPSolve

class qlcp(baseSolver):
    __license__ = "MIT"
    __authors__ = "Enzo Michelangeli"
    #_requiresBestPointDetection = True
    
    __name__ = 'qlcp'
    __alg__ = 'Lemke algorithm, using linear complementarity problem'
    #__isIterPointAlwaysFeasible__ = True
    
    __optionalDataThatCanBeHandled__ = ['lb', 'ub', 'A', 'b', 'Aeq', 'beq']
    
    def __init__(self): pass
    def __solver__(self, p):
        # TODO: add QI
        x,  retcode = QPSolve(p.H, p.f, p.A, p.b, p.Aeq, p.beq, p.lb, p.ub)
#Q, e, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None, QI=None
        if retcode[0] == 1:
            p.istop = 1000
            p.xf = x
        else:
            p.istop = -1

        # TODO: istop, msg wrt retcode
        
