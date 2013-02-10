"""
Example:
Solving nonsmooth problem
#K|x1| + |x2| -> min
#x0 = [10^4, 10]
x_opt = all-zeros
f_opt = 0
"""

from numpy import *
from openopt import NSP

K = 10**3

f = lambda x: abs(x[0]) + abs(x[1])*K + abs(x[2]) * K**2

x0 = [1000, 0.011, 0.01]

#OPTIONAL: user-supplied gradient/subgradient
df = lambda x: [sign(x[0]), sign(x[1])*K, sign(x[2]) * K**2]


#p.df = lambda x: 2*x
    
#p.plot = 0
#p.xlim = (inf,  5)
#p.ylim = (0, 5000000)
#p.checkdf()
solvers = ['r2', 'ipopt', 'algencan','ralg']
solvers = ['r2', 'algencan','ralg']
#solvers = ['ralg', 'r2']
solvers = ['r2', 'lincher']
solvers = ['ralg']
solvers = ['r2']
#solvers = ['scipy_slsqp']
#solvers = ['algencan']
#solvers = ['ipopt']
colors = ['r', 'b', 'k', 'g']
maxIter = 1000
for i, solver in enumerate(solvers):
    p = NSP(f, x0, df=df, xtol = 1e-11, ftol=1e-10,  maxIter = maxIter, maxTime=150)
    #p.checkdf()
    r = p.solve(solver, maxVectorNum=4, iprint=1, showLS=0, plot=0, color=colors[i], show=solver==solvers[-1]) # ralg is name of a solver
#for i, solver in enumerate(solvers):
#    p2 = NSP(f, r.xf, df=df, xtol = 1e-6, maxIter = 1200, maxTime=150, ftol=1e-6)
#    #p.checkdf()
#    r2 = p2.solve(solver, maxVectorNum=15, iprint=1, showLS=1, plot=0, color=colors[i], show=solver==solvers[-1]) # ralg is name of a solver
#print 'x_opt:\n', r.xf
print 'f_opt:', r.ff  # should print small positive number like 0.00056
