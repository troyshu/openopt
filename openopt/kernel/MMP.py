from baseProblem import NonLinProblem
from NLP import NLP
from numpy import max, array, hstack, vstack, zeros, ones, inf, asfarray, atleast_2d

class MMP(NonLinProblem):
    """
    Mini-Max Problem
    """
    _optionalData = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    expectedArgs = ['f', 'x0']
    allowedGoals = ['minimax']
    probType = 'MMP'
    isObjFunValueASingleNumber = False
    showGoal = True
    goal = 'minimax'
    
    def __init__(self, *args, **kwargs):
        NonLinProblem.__init__(self, *args, **kwargs)

    def objFuncMultiple2Single(self, fv):
        return max(fv)

    def mmp2nlp(self, solver, **solver_params):
        f = lambda x: x[-1]
        DF = array([0]*self.n + [1])
        df = lambda x: DF.copy()

        def iterfcn(*args,  **kwargs):
            p2.primalIterFcn(*args,  **kwargs)

            self.xk = p2.xk[:-1].copy()
            self.fk = p2.fk
            self.rk = p2.rk

            self.istop = p2.istop

            if self.istop and p2.rk <= p2.contol:
                self.msg = p2.msg
            self.iterfcn()

        p2 = NLP(f, hstack((self.x0, max(self.f(self.x0)))), df=df, xtol = self.xtol, ftol = self.ftol, gtol = self.gtol,\
        A=hstack((self.A, zeros((len(self.b), 1)))),  b=self.b,  Aeq=hstack((self.Aeq, zeros((len(self.beq), 1)))),  beq=self.beq,  lb=hstack((self.lb, -inf)),  ub=hstack((self.ub, inf)), \
        maxFunEvals = self.maxFunEvals, fEnough = self.fEnough, maxIter=self.maxIter, iprint = -1, \
        maxtime = self.maxTime, maxCPUTime = self.maxCPUTime,  noise = self.noise)

        if self.userProvided.c:
            arr_dc = array([0]*self.nc + [-1]*self.nf).reshape(-1, 1)
            p2.c = lambda x: hstack((self.c(x[:-1]), self.f(x[:-1])-x[-1]))
            p2.dc = lambda x: hstack((vstack((self.dc(x[:-1]), self.df(x[:-1]))), arr_dc))
        else:
            p2.c = lambda x: self.f(x[:-1])-x[-1]
            arr_dc = -ones((self.nf, 1))
            p2.dc = lambda x: hstack((atleast_2d(self.df(x[:-1]),  arr_dc)))
        if self.userProvided.h:
            arr_dh = array([0]*self.nh).reshape(-1, 1)
            p2.h = lambda x: self.h(x[:-1])
            p2.dh = lambda x: hstack((atleast_2d(self.dh(x[:-1])), arr_dh))


        p2.primalIterFcn,  p2.iterfcn = p2.iterfcn, iterfcn

        #p2.checkdc()

        r2 = p2.solve(solver)
        #xf = fsolve(self.f, self.x0, fprime=self.df, xtol = self.xtol, maxfev = self.maxFunEvals)
        xf = r2.xf[:-1]
        self.xk = self.xf = xf
        self.fk = self.ff = max(self.f(xf))

