'''
Simple OpenOpt maximum clique example;
requires networkx (http://networkx.lanl.gov)
and FuncDesigner installed.
Limitations on time, cputime, "enough" value, basic GUI features are available.
'''
from openopt import MCP

import networkx as nx
G = nx.path_graph(15) # [(0,1), (1,2), (2,3), ..., (13,14)]
# let's add edges (0,8), (1,9), (2,10) etc
G.add_edges_from([(i, (i+8)%15) for i in range(15)])
# you can use string and other types, e.g. 
# G = nx.Graph(); G.add_node('asdf'); G.add_nodes_from(['a1', 'b2','c3']); G.add_edge('qwerty1','qwerty2')
p = MCP(G) # you can use parameters "excludedNodes" and "includedNodes", e.g.
#includedNodes = [1, 5, 8], excludedNodes = [0, 4, 10])
# or 
# p = STAB(G, includedNodes = ['a1', 'c3'], excludedNodes = ['qwerty1'])
# includedNodes and excludedNodes are optional arguments - nodes that must be present or absent in the solution

# solve by BSD- licensed global nonlinear solver interalg (http://openopt.org/interalg):
#r = p.solve('interalg', iprint = 0) # set it to -1 to completely suppress interalg output; may not work for glpk, lpSolve
# or solve by MILP solver:
r = p.solve('glpk') # see http://openopt.org/MILP for other MILP solvers available in OpenOpt

# if your solver is cplex or interalg, you can provide some stop criterion, 
# e.g. maxTime, maxCPUTime, fEnough etc, for example 
# r = p.solve('cplex', maxTime = 100, fEnough = 10) 
# i.e. stop if solution with 10 nodes has been obtained or 100 sec were elapsed
# (gplk and lpSolve has no iterfcn connected and thus cannot handle those criterions)

# also you can use p.manage() to enable basic GUI (http://openopt.org/OOFrameworkDoc#Solving) 
# it requires tkinter installed, that is included into PythonXY, EPD;
# for linux use [sudo] easy_install tk or [sodo] apt-get install python-tk
#r = p.manage('cplex')

print(r.ff, r.solution)
'''
Solver:   Time Elapsed = 0.2 	CPU Time Elapsed = 0.05
objFunValue: 3 (feasible, MaxResidual = 0)
(3, [2, 3, 10])
for the example with text nodes the list will be made of text names of them
'''
