from baseProblem import MatrixProblem

class MCP(MatrixProblem):
    _optionalData = []
    probType = 'MCP'
    expectedArgs = ['graph']
    allowedGoals = ['maximum clique']
    showGoal = False

    _init = False
    
    def __setattr__(self, attr, val): 
        if self._init: self.err('openopt MCP instances are immutable, arguments should pass to constructor or solve()')
        self.__dict__[attr] = val

    def __init__(self, *args, **kw):
        MatrixProblem.__init__(self, *args, **kw)
        self.__init_kwargs = kw
        self._init = True
        
    def solve(self, *args, **kw):
        import networkx as nx
        graph = nx.complement(self.graph)
        from openopt import STAB
        KW = self.__init_kwargs
        KW.update(kw)
        P = STAB(graph, **KW)
        r = P.solve(*args)
        return r
        
