from baseProblem import MatrixProblem

class STAB(MatrixProblem):
    _optionalData = []
    probType = 'STAB'
    expectedArgs = ['graph']
    allowedGoals = ['graph stability number']
    showGoal = False
    
    def solve(self, *args, **kw):
        if len(args) > 1:
            self.err('''
            incorrect number of arguments for solve(), 
            must be at least 1 (solver), other must be keyword arguments''')
        solver = args[0] if len(args) != 0 else kw.get('solver', self.solver)
        
        graph = self.graph # must be networkx instance
        nodes = graph.nodes()
        edges = graph.edges()
        
        n = len(nodes)
        
        node2index = dict([(node, i) for i, node in enumerate(nodes)])
        index2node = dict([(i, node) for i, node in enumerate(nodes)])
        
        import FuncDesigner as fd, openopt
        x = fd.oovars(n, domain=bool)
        objective = fd.sum(x)
        startPoint = {x:[0]*n}
        
        fixedVars = {}
       
        includedNodes = getattr(kw, 'includedNodes', None)
        if includedNodes is None:
            includedNodes = getattr(self, 'includedNodes', ())
        for node in includedNodes:
            fixedVars[x[node2index[node]]] = 1

        excludedNodes = getattr(kw, 'excludedNodes', None)
        if excludedNodes is None:
            excludedNodes = getattr(self, 'excludedNodes', ())
        for node in excludedNodes:
            fixedVars[x[node2index[node]]] = 0

        if openopt.oosolver(solver).__name__ == 'interalg':
            constraints =  [fd.NAND(x[node2index[i]], x[node2index[j]]) for i, j in edges]
            P = openopt.GLP
        else:
            constraints =  [x[node2index[i]]+x[node2index[j]] <=1 for i, j in edges]
            P = openopt.MILP
            
        p = P(objective, startPoint, constraints = constraints, fixedVars = fixedVars, goal = 'max')
        
        for key, val in kw.items():
            setattr(p, key, val)
        r = p.solve(solver, **kw)
        r.solution = [index2node[i] for i in range(n) if r.xf[x[i]] == 1]
        r.ff = len(r.solution)
        return r
    
    
#    def objFunc(self, x):
#        return dot(self.f, x) + self._c



