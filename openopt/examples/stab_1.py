'''
Simple OpenOpt graph stability number example;
requires networkx (http://networkx.lanl.gov)
and FuncDesigner installed.
For maximum clique problem you could use STAB on complementary graph, for networkx it's nx.complement(G) 
Unlike networkx maximum_independent_set() we search for *exact* solution.
Limitations on time, cputime, "enough" value, basic GUI features are available.
'''
from openopt import STAB

import networkx as nx
G = nx.path_graph(15) # [(0,1), (1,2), (2,3), ..., (13,14)]
# you can use string and other types, e.g. 
# G = nx.Graph(); G.add_node('asdf'); G.add_nodes_from(['a1', 'b2','c3']); G.add_edge('qwerty1','qwerty2')
p = STAB(G, includedNodes = [1, 5, 8], excludedNodes = [0, 4, 10])
# or 
# p = STAB(G, includedNodes = ['a1', 'c3'], excludedNodes = ['qwerty1'])

# includedNodes and excludedNodes are optional arguments - nodes that must be present or absent in the solution

# solve by BSD- licensed global nonlinear solver interalg (http://openopt.org/interalg):
r = p.solve('interalg', iprint = 0) # set it to -1 to completely suppress interalg output; may not work for glpk, lpSolve
# or solve by MILP solver:
#r = p.solve('lpSolve') # see http://openopt.org/MILP for other MILP solvers available in OpenOpt

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
Solver:   Time Elapsed = 0.34   CPU Time Elapsed = 0.28
(6, [1, 3, 5, 8, 12, 14])
for the example with text nodes it will be
['qwerty1', 'a1', 'b2', 'c3', 'asdf']
'''
