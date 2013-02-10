#!/usr/bin/python
'''
A simple OpenOpt TSP example for directed multigraph;
requires networkx (http://networkx.lanl.gov)
and FuncDesigner installed.
For some solvers limitations on time, cputime, "enough" value, basic GUI features are available.
See http://openopt.org/TSP for more details
'''
from openopt import *
from numpy import sin, cos#, abs
import networkx as nx

N = 5
G = nx.MultiDiGraph()

G.add_edges_from(\
                 [(i,j,{'time': 1.5*(cos(i)+sin(j)+1)**2, 'cost':(i-j)**2 + 2*sin(i) + 2*cos(j)+1, 'way': 'aircraft'}) for i in range(N) for j in range(N) if i != j ])
G.add_edges_from(\
                 [(i,j,{'time': 4.5*(cos(i)-sin(j)+1)**2, 'cost':(i-j)**2 + sin(i) + cos(j)+1, 'way': 'railroad'}) for i in range(int(2*N/3)) for j in range(int(N)) if i != j ])

objective = 'time'

# for solver sa handling of constraints is unimplemented yet
# when your solver is interalg you can use nonlinear constraints as well
# for nonlinear objective and constraints functions like arctan, abs etc should be imported from FuncDesigner

constraints = lambda values: (2 * values['time'] + 3 * values['cost'] > 100, 8 * values['time'] + 15*values['cost'] > 150)

p = TSP(G, objective = objective, constraints = constraints, startNode = 2, returnToStart=True)

solver = 'glpk' # also you can use some other solvers - sa, interalg, OpenOpt MILP solvers
r = p.solve(solver) 

# if your solver is cplex or interalg, you can provide some stop criterion, 
# e.g. maxTime, maxCPUTime, fEnough etc, for example 
# r = p.solve('cplex', maxTime = 100, fEnough = 10) 
# i.e. stop if solution with 10 nodes has been obtained or 100 sec were elapsed
# (gplk and lpSolve has no iterfcn connected and thus cannot handle those criterions)
# also you can use p.manage() to enable basic GUI (http://openopt.org/OOFrameworkDoc#Solving) 
# it requires tkinter installed, that is included into PythonXY, EPD;
# for linux use [sudo] easy_install tk or [sodo] apt-get install python-tk
# r = p.manage('cplex')

print(r.Edges)
'''
Solver:   Time Elapsed = 0.04 	CPU Time Elapsed = 0.04
objFunValue: 6.0653623 (feasible, MaxResidual = 4.44089e-16)
[(4, 3, {'cost': -1.4935899838167472, 'way': 'aircraft', 'time': 0.35644984211087011}), 
(3, 1, {'cost': 6.3628446278560142, 'way': 'aircraft', 'time': 1.0875234238200695}), 
(1, 2, {'cost': 2.4253241482607542, 'way': 'railroad', 'time': 1.7917522081892421}), 
(2, 0, {'cost': 8.8185948536513639, 'way': 'aircraft', 'time': 0.51132677471086396}), 
(0, 4, {'cost': 15.692712758272776, 'way': 'aircraft', 'time': 2.318310053508891})]
'''
