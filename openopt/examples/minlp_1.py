"""
Example of MINLP
It is recommended to read help(NLP) before
and /examples/nlp_1.py 
"""
from openopt import MINLP
from numpy import *
N = 150
K = 50

#objective function:
f = lambda x: ((x-5.45)**2).sum()

#optional: 1st derivatives
df = lambda x: 2*(x-5.45)

# start point
x0 = 8*cos(arange(N))

# assign prob:
# 1st arg - objective function
# 2nd arg - start point
# for more details see 
# http://openopt.org/Assignment 
p = MINLP(f, x0, df=df, maxIter = 1e3)

# optional: set some box constraints lb <= x <= ub
p.lb = [-6.5]*N
p.ub = [6.5]*N
# see help(NLP) for handling of other constraints: 
# Ax<=b, Aeq x = beq, c(x) <= 0, h(x) = 0
# see also /examples/nlp_1.py

# required tolerance for smooth constraints, default 1e-6
p.contol = 1.1e-6

p.name = 'minlp_1'

# required field: nlpSolver - should be capable of handling box-bounds at least
#nlpSolver = 'ralg' 
nlpSolver = 'ipopt'

# coords of discrete variables and sets of allowed values
p.discreteVars = {7:range(3, 10), 8:range(3, 10), 9:[2, 3.1, 9]}

# required tolerance for discrete variables, default 10^-5
p.discrtol = 1.1e-5

#optional: check derivatives, you could use p.checkdc(), p.checkdh() for constraints
#p.checkdf()

# optional: maxTime, maxCPUTime
# p.maxTime = 15
# p.maxCPUTime = 15

r = p.solve('branb', nlpSolver=nlpSolver, plot = False)
# optim point and value are r.xf and r.ff,
# see http://openopt.org/OOFrameworkDoc#Result_structure for more details
