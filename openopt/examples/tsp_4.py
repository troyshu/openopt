#!/usr/bin/python
'''
A simple OpenOpt multiobjective TSP example for directed multigraph using interalg solver;
requires networkx (http://networkx.lanl.gov)
and FuncDesigner installed.
For some solvers limitations on time, cputime, "enough" value, basic GUI features are available.
See http://openopt.org/TSP for more details
'''
from openopt import *
from numpy import sin, cos#, abs

import networkx as nx
N = 6
G = nx.MultiDiGraph()

G.add_edges_from(\
                 [(i,j,{'time': 1.5*(cos(i)+sin(j)+1)**2, 'cost':(i-j)**2 + 2*sin(i) + 2*cos(j)+1, 'way': 'aircraft'}) for i in range(N) for j in range(N) if i != j ])

G.add_edges_from(\
                 [(i,j,{'time': 4.5*(cos(i)-sin(j)+1)**2, 'cost':(i-j)**2 + sin(i) + cos(j)+1, 'way': 'railroad'}) for i in range(int(2*N/3)) for j in range(int(N)) if i != j ])

G.add_edges_from(\
                 [(i,j,{'time': 4.5*(cos(4*i)-sin(3*j)+1)**2, 'cost':(i-2*j)**2 + sin(10+i) + cos(2*j)+1, 'way': 'car'}) for i in range(int(2*N/3)) for j in range(int(N)) if i != j ])

G.add_edges_from(\
                 [(i,j,{'time': +(4.5*(cos(i)+cos(j)+1)**2 + abs(i - j)), 'cost': (0.2*i + 0.1*j)**2, 'way': 'bike'}) for i in range(int(N)) for j in range(int(N)) if i != j ])

objective = [
              # name, tol, goal
              'time', 0.005, 'min', 
              'cost', 0.005, 'min'
              ]

# for solver sa handling of constraints is unimplemented yet
# when your solver is interalg you can use nonlinear constraints as well
# for nonlinear objective and constraints functions like arctan, abs etc should be imported from FuncDesigner
from FuncDesigner import arctan, sqrt
constraints = lambda value: (
                             2 * value['time']**2 + 3 * sqrt(value['cost']) < 10000, 
                             8 * arctan(value['time']) + 15*value['cost'] > 15
                             )

p = TSP(G, objective = objective, constraints = constraints)
r = p.solve('interalg', nProc=2) # see http://openopt.org/interalg for more info on the solver

# you can provide some stop criterion, 
# e.g. maxTime, maxCPUTime, fEnough etc, for example 
# r = p.solve('interalg', maxTime = 100, maxCPUTime=100) 

# also you can use p.manage() to enable basic GUI (http://openopt.org/OOFrameworkDoc#Solving) 
# it requires tkinter installed, that is included into PythonXY, EPD;
# for linux use [sudo] easy_install tk or [sodo] apt-get install python-tk
#r = p.manage('interalg')

print(r.solutions.values)
print(r.solutions)
'''
istop: 1001 (all solutions have been obtained)
Solver:   Time Elapsed = 319.53 	CPU Time Elapsed = 297.35
4 solutions have been obtained
[[ 41.86372151   8.53983205]
 [ 56.35337592   4.51      ]
 [ 36.67772554  22.73951982]
 [ 34.649273    31.5852164 ]]
...
'''
