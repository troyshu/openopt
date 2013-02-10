from baseProblem import MatrixProblem
import numpy as np

class TSP(MatrixProblem):
    _optionalData = []
    probType = 'TSP'
    expectedArgs = ['graph', 'objective']
    allowedGoals = ['min', 'max', 'minimum', 'maximum']
    showGoal = True
    allowRevisit = False # True is unimplemented properly yet
    _init = False
    start = None 
    returnToStart = True
    
    def __setattr__(self, attr, val): 
        if self._init: self.err('openopt TSP instances are immutable, arguments should pass to constructor or solve()')
        self.__dict__[attr] = val
    
    def __init__(self, *args, **kw):
        self.goal = 'min'
        self.objective = 'weight'
        MatrixProblem.__init__(self, *args, **kw)
        self.__init_kwargs = kw
        self._init = True

    def solve(self, *args, **kw):
        if len(args) > 1:
            self.err('''
            incorrect number of arguments for solve(), 
            must be at least 1 (solver), other must be keyword arguments''')
        if self.start is None and not self.returnToStart:
            self.err('for returnToStart=False mode you should provide start, other cases are unimplemented yet')
        solver = args[0] if len(args) != 0 else kw.get('solver', self.solver)
        KW = self.__init_kwargs.copy()
        KW.update(kw)
        
        objective = KW.get('objective', self.objective)
        if isinstance(objective, (list, tuple, set)):
            nCriteria = len(self.objective)
            if 3 * nCriteria != np.asarray(self.objective).size:
                objective = [(objective[3*i], objective[3*i+1], objective[3*i+2]) for i in range(int(round(np.asarray(self.objective).size / 3)))]
            if len(objective) == 1:
                KW['fTol'], KW['goal'] = objective[0][1:]
        else:
            objective = [(self.objective, KW.get('fTol', getattr(self, 'fTol')), KW.get('goal', getattr(self, 'goal')))]

        nCriteria = len(objective)
        isMOP = nCriteria > 1
        mainCr = objective[0][0]
         
        import FuncDesigner as fd, openopt as oo 
        solverName = solver if type(solver) == str else solver.__name__
        is_interalg = solverName == 'interalg'
        is_glp = solverName == 'sa'
        if is_glp:
            assert nCriteria == 1, 'you cannot solve multiobjective tsp by the solver'
            
        is_interalg_raw_mode = is_interalg and KW.get('dataHandling', oo.oosolver(solver).dataHandling) in ('auto','raw')
        KW.pop('objective', None)
        P = oo.MOP if nCriteria > 1 else oo.GLP if is_interalg else oo.MILP if not is_glp else oo.GLP

        import networkx as nx
        graph = self.graph # must be networkx Graph instance
        
        init_graph_is_directed = graph.is_directed()
        init_graph_is_multigraph = graph.is_multigraph()
        if not init_graph_is_multigraph or not init_graph_is_directed:
            graph = nx.MultiDiGraph(graph) #if init_graph_is_directed else nx.MultiGraph(graph)
        
        nodes = graph.nodes()
        edges = graph.edges()
        n = len(nodes)
        m = len(edges)

        node2index = dict([(node, i) for i, node in enumerate(nodes)])
        
        # TODO: implement MOP with interalg_gdp mode (requires interpolation interval analysis for non-monotone funcs)
        interalg_gdp = 1
        if not is_interalg:# or isMOP:
            # !!!!!!!!!!!!! TODO: add handling of MOP with interalg_gdp?
            interalg_gdp = 0 
            
        if interalg_gdp:
            x = []
            edge_ind2x_ind_val = {}
        else:
            pass
            #x = fd.oovars(m, domain=bool)

        #cr_values = dict([(obj[0], []) for obj in objective])
        cr_values = {}
        constraints = []
        EdgesDescriptors, EdgesCoords = [], []
        # mb rework it by successors etc?
        
        Funcs = {}
        Cons = KW.pop('constraints', [])
        if type(Cons) not in (list, tuple):
            Cons = [Cons]
        usedValues = getUsedValues(Cons)
        usedValues.update(getUsedValues([obj[0] for obj in objective]))
        
        MainCr = mainCr if type(mainCr) in (str, np.str_) else list(usedValues)[0]
        
        isMainCrMin = objective[0][2] in ('min', 'minimum')
        node_out_edges_num = []
        for node in nodes:
            Edges = graph[node]
            node_out_edges_num.append(len(Edges))
            out_nodes = Edges.keys()
            if len(out_nodes) == 0:
                self.err('input graph has node %s that does not lead to any other node; solution is impossible' % node)            
            
            if init_graph_is_multigraph and not isMOP and type(mainCr) in [str, np.str_]:
                W = {}
                for out_node in out_nodes:
                    ww = list(Edges[out_node].values())
                    for w in ww:
                        tmp = W.get(out_node, None)
                        if tmp is None:
                            W[out_node] = w
                            continue
                        th = tmp[mainCr]
                        w_main_cr_val = w[mainCr]
                        if isMainCrMin  == (th > w_main_cr_val):
                            W[out_node] = w
                Out_nodes, W = np.array(list(W.keys())), np.array(list(W.values()))
            else:
                W = np.hstack([list(Edges[out_node].values()) for out_node in out_nodes])
                Out_nodes = np.hstack([[out_node] * len(Edges[out_node]) for out_node in out_nodes])
            
            if interalg_gdp:
                rr = np.array([w[MainCr] for w in W])
                if isMainCrMin:
                    rr = -rr
                elif objective[0][2] not in ('max', 'maximum'):
                    self.err('unimplemented for fixed value goal in TSP yet, only min/max is possible for now')

                ind = rr.argsort()
                W = W[ind]
                Out_nodes = Out_nodes[ind]

            lc = 0
            for i, w in enumerate(W):
                if interalg_gdp:
                    edge_ind2x_ind_val[len(EdgesCoords)] = (len(x), lc)
                lc += 1
                EdgesCoords.append((node, Out_nodes[i]))
                EdgesDescriptors.append(w)

                for key, val in w.items():
                    # for undirected:
                    #if node2index[key] < node2index[out_node]: continue
                    Val = val if self.returnToStart or node != self.start else 0
                    if key in cr_values:
                        cr_values[key].append(Val)
                    else:
                        cr_values[key] = [Val]
            if interalg_gdp:
                x.append(fd.oovar(domain = np.arange(lc)))            
        
        m = len(EdgesCoords) # new value
        
        if is_glp:
            if type(mainCr) not in (str, np.str_):
                self.err('for the solver "sa" only text name objectives are implemented (e.g. "time", "price")')
            if init_graph_is_multigraph:
                self.err('the solver "sa" cannot handle multigraphs yet')
            if len(Cons) != 0:
                self.err('the solver "sa" cannot handle constrained TSP yet')
            M = np.empty((n, n))
            M.fill(np.nan)
            Cr_values = np.array(cr_values[mainCr])
            isMax = objective[0][-1] in ('max', 'maximum')
            if isMax:
                Cr_values = -Cr_values
            for i, w in enumerate(EdgesDescriptors):
                node_in, node_out = EdgesCoords[i]
                M[node_in, node_out] = Cr_values[i]
            S = np.abs(Cr_values).sum() + 1.0
            
            # TODO: check it
            M[np.isnan(M)] = S
            
            prob = P(lambda x: 0, np.zeros(n), iprint = 1)
            prob.f = lambda x: np.nan if not hasattr(prob, 'ff') else (prob.ff if isMax else -prob.ff)
            prob.M = dict([((i, j), M[i, j]) for i in range(n) for j in range(n) if i != j])
            r = prob.solve(solver, **KW)
            xf = [nodes[j] for j in np.array(r.xf, int)]
            r.nodes = xf#.tolist() 
            if self.start is not None:
                j = r.nodes.index(self.start)
                r.nodes = r.nodes[j:] + r.nodes[:j]
            if self.returnToStart:
                r.nodes += [r.nodes[0]]
            r.edges = [(r.nodes[i], r.nodes[i+1]) for i in range(n-1)] 
            r.Edges = [(r.nodes[i], r.nodes[i+1], graph[r.nodes[i]][r.nodes[i+1]][0]) for i in range(n-1)] 
            if self.returnToStart:
                r.edges.append((r.nodes[-2], r.nodes[0]))
                print(r.nodes[-1], r.nodes[0], type(r.nodes[-1]), type(r.nodes[0]), graph[2])
                r.Edges.append((r.nodes[-2], r.nodes[0], graph[r.nodes[-2]][r.nodes[0]][0]))
            #r.xf = r.xk = r.nodes
            # TODO: Edges
            return r
            
        
        #TODO: fix ooarray lb/ub
        #u = np.array([1] + [fd.oovar(lb=2, ub=n) for i in range(n-1)])
        u = fd.hstack((1, fd.oovars(n-1, lb=2, ub=n)))
        for i in range(1, u.size):
            u[i]('u' + str(i))
            
        
        if is_interalg_raw_mode:
            for i in range(n-1):
                u[1+i].domain = np.arange(2, n+1)
        
        if interalg_gdp: 
            assert len(x) == n
            x = fd.ooarray(x)
            
            # TODO: remove it when proper enum implementation in FD engine will be done
            for i in range(n):
                x[i]._init_domain = x[i].domain
                constraints.append(x[i]-x[i]._init_domain[-1] <= 0)
                x[i].domain = np.arange(int(2 ** np.ceil(np.log2(node_out_edges_num[i]))))
#            for i in range(n-1):
#                u[1+i]._init_domain = u[1+i].domain
#                constraints.append(u[1+i]-u[1+i]._init_domain[-1] <= 0)
#                u[1+i].domain = np.arange(u[1+i]._init_domain[0], u[1+i]._init_domain[0]+int(2 ** np.ceil(np.log2(u[1+i]._init_domain[-1]-u[1+i]._init_domain[0]+1))))
#                u[1+i].ub = u[1+i].domain[-1]

        else:
            x = fd.oovars(m, domain=bool) # new m value
        for i in range(x.size):
            x[i]('x'+str(i))
            
            
#        if init_graph_is_directed:
        dictFrom = dict([(node, []) for node in nodes])
        dictTo = dict([(node, []) for node in nodes])
        for i, edge in enumerate(EdgesCoords):
            From, To = edge
            dictFrom[From].append(i)
            dictTo[To].append(i)
        
        
        engine = fd.XOR
        
        # number of outcoming edges = 1
        if not interalg_gdp:
            for node, edges_inds in dictFrom.items():
                # !!!!!!!!!! TODO for interalg_raw_mode: and if all edges have sign similar to goal
                if 1 and is_interalg_raw_mode:
                    c = engine([x[j] for j in edges_inds])
                else:
                    nEdges = fd.sum([x[j] for j in edges_inds]) 
                    c =  nEdges >= 1 if self.allowRevisit else nEdges == 1
                constraints.append(c)

        # number of incoming edges = 1
        for node, edges_inds in dictTo.items():
            if len(edges_inds) == 0:
                self.err('input graph has node %s that has no edge from any other node; solution is impossible' % node)
            
            if interalg_gdp:
                x_inds, x_vals = [], []
                for elem in edges_inds:
                    x_ind, x_val = edge_ind2x_ind_val[elem]
                    x_inds.append(x_ind)
                    x_vals.append(x_val)
                c = engine([(x[x_ind] == x_val)(tol = 0.5) for x_ind, x_val in zip(x_inds, x_vals)])
            else:
                if 1 and is_interalg_raw_mode and engine == fd.XOR:
                    c = engine([x[j] for j in edges_inds])
                else:            
                    nEdges = fd.sum([x[j] for j in edges_inds]) 
                    c =  nEdges >= 1 if self.allowRevisit else nEdges == 1
            constraints.append(c)
        
        # MTZ
        for i, (I, J) in enumerate(EdgesCoords):
            ii, jj = node2index[I], node2index[J]
            if ii != 0 and jj != 0:
                if interalg_gdp:
                    x_ind, x_val = edge_ind2x_ind_val[i]
                    c = fd.ifThen((x[x_ind] == x_val)(tol=0.5), u[ii] - u[jj]  <= - 1.0)
                elif is_interalg_raw_mode:
                    c = fd.ifThen(x[i], u[ii] - u[jj]  <= - 1.0)#u[jj] - u[ii]  >= 1)
                else:
                    c = u[ii] - u[jj] + 1 <= (n-1) * (1-x[i])
                constraints.append(c)
        
        # handling objective(s)
        FF = []
        
        for optCrName in usedValues:
            
            tmp = cr_values.get(optCrName, [])

            if len(tmp) == 0:
                self.err('seems like graph edgs have no attribute "%s" to perform optimization on it' % optCrName)
            elif len(tmp) != m:
                self.err('for optimization creterion "%s" at least one edge has no this attribute' % optCrName)
            if interalg_gdp:
                F = []
                lc = 0
                for X in x:
                    domain = X._init_domain
                    vals = [tmp[i] for i in range(lc, lc + domain.size)]
                    lc += domain.size
                    #F = sum(x)
                    #F.append(fd.interpolator(domain, vals, k=1, s=0.00000001)(X))
                    F.append(fd.interpolator(domain, vals, k=1)(X))
                    #print(domain, vals)
                F = fd.sum(F)
            else:
                F = fd.sum(x*tmp)
            Funcs[optCrName] = F
        
        for obj in objective:
            FF.append((Funcs[obj[0]] if type(obj[0]) in (str, np.str_) else obj[0](Funcs), obj[1], obj[2]))

        for c in Cons:
            tmp = c(Funcs)
            if type(tmp) in (list, tuple, set):
                constraints += list(tmp)
            else:
                constraints.append(tmp)
        
        startPoint = {x:[0]*(m if not interalg_gdp else n)}
        startPoint.update(dict([(U, i+2) for i, U in enumerate(u[1:])]))

        p = P(FF if isMOP else FF[0][0], startPoint, constraints = constraints)#, fixedVars = fixedVars)
        for param in ('start', 'returnToStart'):
            KW.pop(param, None)
        r = p.solve(solver, **KW)

        if P != oo.MOP:
            r.ff = p.ff
            if interalg_gdp:
                x_ind_val2edge_ind = dict([(elem[1], elem[0]) for elem in edge_ind2x_ind_val.items()])
                SolutionEdges = [(EdgesCoords[i][0], EdgesCoords[i][1], EdgesDescriptors[i]) for i in [x_ind_val2edge_ind[(ind, x[ind](r))] for ind in range(n)]]
            else:
                SolutionEdges = [(EdgesCoords[i][0], EdgesCoords[i][1], EdgesDescriptors[i]) for i in range(m) if r.xf[x[i]] == 1]
            if len(SolutionEdges) == 0: 
                r.nodes = r.edges = r.Edges = []
                return r
                
            S = dict([(elem[0], elem) for elem in SolutionEdges])
            
            SE = [SolutionEdges[0]]
            for i in range(len(SolutionEdges)-1):
                SE.append(S[SE[-1][1]])
            SolutionEdgesCoords = [(elem[0], elem[1]) for elem in SE]

            nodes = [edge[1] for edge in SolutionEdgesCoords]
            if self.start is not None:
                shift_ind = nodes.index(self.start)
                nodes = nodes[shift_ind:] + nodes[:shift_ind]
            
            if self.returnToStart:
                nodes.append(nodes[0])

            edges = SolutionEdgesCoords[1:] + [SolutionEdgesCoords[0]]
            Edges = SE[1:] + [SE[0]]
            if self.start is not None:
                edges, Edges = edges[shift_ind:] + edges[:shift_ind], Edges[shift_ind:] + Edges[:shift_ind]
            if not self.returnToStart:
                edges, Edges = edges[:-1], Edges[:-1]
            r.nodes, r.edges, r.Edges = nodes, edges, Edges
        else:
            r.solution = 'for MOP see r.solutions instead of r.solution'
            tmp_c, tmp_v = r.solutions.coords, r.solutions.values
#            if interalg_gdp:
#                x_ind_val2edge_ind = dict([(elem[1], elem[0]) for elem in edge_ind2x_ind_val.items()])
#                SolutionEdges = [(EdgesCoords[i][0], EdgesCoords[i][1], EdgesDescriptors[i]) for i in [x_ind_val2edge_ind[(ind, x[ind](r))] for ind in range(n)]]
#            else:
#                SolutionEdges = [(EdgesCoords[i][0], EdgesCoords[i][1], EdgesDescriptors[i]) for i in range(m) if r.xf[x[i]] == 1]
            if interalg_gdp: # default for MOP
                x_ind_val2edge_ind = dict([(elem[1], elem[0]) for elem in edge_ind2x_ind_val.items()])
                r.solutions = MOPsolutions([[(EdgesCoords[i][0], EdgesCoords[i][1], EdgesDescriptors[i]) for i in [x_ind_val2edge_ind[(ind, x[ind](Point))] for ind in range(n)]] for Point in r.solutions])
            else:# non-default
                r.solutions = MOPsolutions([[(EdgesCoords[i][0], EdgesCoords[i][1], EdgesDescriptors[i]) for i in range(m) if Point[x[i]] == 1] for Point in r.solutions])
                
            r.solutions.values = tmp_v
        return r

class MOPsolutions(list):
    pass

class D:
    def __init__(self):
        self.used_vals = set()
    def __getitem__(self, item):
        self.used_vals.add(item)
        return 1.0
    
def getUsedValues(Iterator):
    d = D()
    r = set()
    for elem in Iterator:
        if type(elem) in [str, np.str_]:
            r.add(elem)
        else:
            elem(d)
    r.update(d.used_vals)
    return r
     
