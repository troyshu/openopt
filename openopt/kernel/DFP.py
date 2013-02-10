from baseProblem import NonLinProblem
from ooMisc import assignScript
from numpy import sum, dot, asfarray, atleast_2d, array, zeros
import NLP

class DFP(NonLinProblem):
    _optionalData = ['lb', 'ub', 'A', 'Aeq', 'b', 'beq', 'c', 'h']
    probType = 'DFP'
    expectedArgs = ['f', 'x0', 'X', 'Y']
    allowedGoals = ['minimum', 'min']
    goal = 'minimum'
    showGoal = False
    isObjFunValueASingleNumber = False
    
    def _Prepare(self):
        self.X = atleast_2d(self.X)
        self.Y = array(self.Y, float)
        if self.X.shape[0] != self.Y.shape[0]:
            if self.X.shape[1] != self.Y.shape[0]:
                self.err('incorrect shape of input data')
            else:
                self.X = self.X.T
        NonLinProblem._Prepare(self)
        #if self.Y.ndim

        if self.userProvided.df:
            assert len(self.user.df) == 1
            self.dfff = self.user.df[0]
            def dff(x):
                r = zeros(self.n)
                for i in range(self.Y.shape[0]):
                    #print asfarray(self.fff(x, self.X[i])-self.Y[i]), asfarray(self.dfff(x, self.X[i]))
                    r += dot(2.0 * asfarray(self.fff(x, self.X[i])-self.Y[i]), asfarray(self.dfff(x, self.X[i])))
                return r        
            self.df = self.user.df = dff

    def __finalize__(self):
        NonLinProblem.__finalize__(self)
        if self.userProvided.df:
            self.df = self.dfff
            self.f = self.fff

    def __init__(self, *args, **kwargs):
        NonLinProblem.__init__(self, *args, **kwargs)
        assignScript(self, kwargs)
        self.fff = self.f
        def ff(x):
            r = []
            for i in range(self.Y.shape[0]):
                r.append(asfarray(self.fff(x, self.X[i])-self.Y[i])**2)
            return r
        self.f = ff

    def objFuncMultiple2Single(self, fv):
        assert all(fv.flatten()>=0)
        return fv.sum() # they are squared in other place

    def dfp2nlp(self, solver, **solver_params):
        ff = lambda x: (asfarray(self.f(x))).sum()
        if self.userProvided.df:
            #dff = lambda x: dot(2*asfarray(ff(x)), asfarray(self.df(x)))
            p = NLP.NLP(ff, self.x0, df=self.df)
        else:
            p = NLP.NLP(ff, self.x0)
            
        #p = NLP.NLP(lambda x: self.f(x), self.x0)
        #if self.userProvided.df: p.df = dff
        self.inspire(p, sameConstraints=True)
        p.f = ff # to prefent overwriting

        def dfp_iterfcn(*args,  **kwargs):
            self.iterfcn(*args,  **kwargs)
            if self.istop != 0: p.istop, p.msg = self.istop, self.msg
            tmp_iterfcn(*args,  **kwargs)
            if p.istop != 0: self.istop, self.msg = p.istop, p.msg

        p.iterfcn, tmp_iterfcn = dfp_iterfcn, p.iterfcn

        #p.iprint = -1
        self.iprint = -1
        if self.plot: 
            self.plot, p.plot = 0, 1
            p.show = self.show
        #p.show = False
        
        p.checkdf()
        
        r = p.solve(solver, **solver_params)
        #r.ff = ff(r.xf)

        return r
