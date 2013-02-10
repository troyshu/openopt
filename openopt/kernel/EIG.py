from baseProblem import MatrixProblem
#from numpy.linalg import norm
from numpy import vstack, isscalar

class EIG(MatrixProblem):
    probType = 'EIG'
    goal = 'all'
    allowedGoals = None
    showGoal = True
    expectedArgs = ['C']
    M = None
    _optionalData = ['M']
    xtol = 0.0
    FuncDesignerSign = 'C'
    N = 0
    
    #ftol = None
    def __init__(self, *args, **kwargs):
        MatrixProblem.__init__(self, *args, **kwargs)

        if self.goal == 'all':
            Name, name = 'all eigenvectors and eigenvalues', 'all'
            if not isinstance(self.C[0], dict):
                self.N = self.C.shape[0]
        else:
            assert type(self.goal) in (dict, tuple, list) and len(self.goal) == 1, \
            'EIG goal argument should be "all" or Python dict {goal_name: number_of_required_eigenvalues}'
            if type(self.goal) == dict:
                goal_name, N = list(self.goal.items())[0]
            else:
                goal_name, N = self.goal
            self.N = N
            name = ''.join(goal_name.lower().split())
            if name  in ('lm', 'largestmagnitude'):
                Name, name = 'largest magnitude', 'le'
            elif name in ('sm', 'smallestmagnitude'):
                Name, name = 'smallest magnitude', 'sm'
            elif name in ('lr', 'largestrealpart'):
                Name, name = 'largest real part', 'lr'
            elif name in ('sr', 'smallestrealpart'):
                Name, name = 'smallest real part', 'sr'
            elif name in ('li', 'largestimaginarypart'):
                Name, name = 'largest imaginary part', 'li'
            elif name in ('si', 'smallestimaginarypart'):
                Name, name = 'smallest imaginary part', 'si'
            elif name in ('la', 'largestamplitude'):
                Name, name = 'largestamplitude', 'la'
            elif name in ('sa', 'smallestamplitude'):
                Name, name = 'smallest amplitude', 'sa'
            elif name in ('be', 'bothendsofthespectrum'):
                Name, name = 'both ends of the spectrum', 'be'
        
        self.goal = Name
        self._goal = name
        #if not isinstance(self.C[0], dict):
            
    
    def solve(self, *args, **kw):
        C = self.C
        if type(C) in (tuple,  list) and isinstance(C[0], dict):
            from FuncDesigner import ootranslator
            K = set()
            N = 0
            varSizes = {}
            for d in C:
                K.update(d.keys())
                for key in d.keys():
                    if key in varSizes:
                        if varSizes[key] != d[key].shape[1]:
                            s = 'incorrect shape 2nd coordinate %d for variable %s, defined in other array as %d' %(d[key].shape[1], key.name, varSizes[key])
                            self.err(s)
                    else:
                        varSizes[key] = d[key].shape[1] if not isscalar(d[key]) else 1
                tmp = list(d.values())
                N += tmp[0].shape[0] if not isscalar(tmp[0]) else 1
            P = dict([(key, [0]*val) for key, val in varSizes.items()])
            T = ootranslator(P)
            C2 = vstack([T.pointDerivative2array(d) for d in C])
            self.C = C2
            if C2.shape != (N, N):
                self.err('square matrix of shape (%d,%d) expected, shape %s obtained instead' % (N, N, C2.shape))

        r = MatrixProblem.solve(self, *args, **kw)
        if type(C) in (tuple,  list) and isinstance(C[0], dict):
            r.eigenvectors = [T.vector2point(v) for v in self.eigenvectors.T]
        return r
        
    
    def objFunc(self, x):
        return 0
        #raise 'unimplemented yet'
        
