#!/usr/bin/python
'''
A simple OpenOpt TSP example for directed graph;
requires networkx (http://networkx.lanl.gov)
and FuncDesigner installed.
For some solvers limitations on time, cputime, "enough" value, basic GUI features are available.
See http://openopt.org/TSP for more details
'''
from openopt import *
from numpy import sin, cos
import networkx as nx

N = 5
G = nx.DiGraph()
G.add_edges_from(\
                 [('node %d' % i, 'node %d' % j,{'time': 1.5*(cos(i)+sin(j)+1)**2, 'cost':(i-j)**2 + 2*sin(i) + 2*cos(j)+1}) for i in range(N) for j in range(N) if i != j ])


# for solver sa handling of constraints is unimplemented yet
# when your solver is interalg you can use nonlinear constraints as well
# for nonlinear objective and constraints functions like arctan, abs etc should be imported from FuncDesigner
constraints = lambda values: (2 * values['time'] + 3 * values['cost'] > 100, 8 * values['time'] + 15*values['cost'] > 150)
objective = lambda values: values['time'] + 10*values['cost'] 
# objective and constraint funcs are applied on sums of all values for all involved edges, not for each standalone edge

p = TSP(G, objective = objective, constraints = constraints, start = 'node 3', returnToStart=False)

r = p.solve('lpSolve') # also you can use some other solvers - sa, interalg, OpenOpt MILP solvers

# if your solver is cplex or interalg, you can provide some stop criterion, 
# e.g. maxTime, maxCPUTime, fEnough etc, for example 
# r = p.solve('cplex', maxTime = 100, fEnough = 10) 
# i.e. stop if solution with 10 nodes has been obtained or 100 sec were elapsed
# (gplk and lpSolve has no iterfcn connected and thus cannot handle those criterions)

# also you can use p.manage() to enable basic GUI (http://openopt.org/OOFrameworkDoc#Solving) 
# it requires tkinter installed, that is included into PythonXY, EPD;
# for linux use [sudo] easy_install tk or [sodo] apt-get install python-tk
#r = p.manage('cplex')

print(r.nodes)
print(r.edges) # (node_from, node_to)
print(r.Edges) # full info on edges; unavailable for solver sa yet
'''
Solver:   Time Elapsed = 0.04 	CPU Time Elapsed = 0.04
# for N = 13 (full-13 graph) lpSolve Time Elapsed = 1.76 	CPU Time Elapsed = 1.74
objFunValue: 231.72217 (feasible, MaxResidual = 0)
['node 3', 'node 4', 'node 1', 'node 0', 'node 2']
[('node 3', 'node 4'), ('node 4', 'node 1'), ('node 1', 'node 0'), ('node 0', 'node 2'), ('node 2', 'node 3')]
[('node 3', 'node 4', {'cost': 0.9749527743925106, 'time': 0.83655413990914185}), 
('node 4', 'node 1', {'cost': 9.5669996211204236, 'time': 2.1164007698022411}), 
('node 1', 'node 0', {'cost': 5.6829419696157935, 'time': 3.5587967901940627}), 
('node 0', 'node 2', {'cost': 4.1677063269057157, 'time': 12.6960172766018}), 
('node 2', 'node 3', {'cost': 1.8386098604504726, 'time': 0.78837914911982798})]
'''
