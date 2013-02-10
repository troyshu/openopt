"""
Example:
Solving nonsmooth problem
|x1| + 1.2|x2| + 1.44|x3| + ... + 1.2^N |xN| -> min
N=75
x0 = [cos(1), cos(2), ..., cos(N)]
x_opt = all-zeros
f_opt = 0
"""

from numpy import *
from openopt import NSP

#N = 12
#K = 11
N = 1000
K = 10

#N = 10
#K = 7

P = 1.0002
P2 = 1.001
a = 3.5

#N = 7
#K = 6

#objFun = lambda x: sum(1.2 ** arange(len(x)) * abs(x))
#objFun = lambda x: sum(1.2 ** arange(len(x)) * x**2)

f1 = lambda x: sum(abs(x) ** P)
f2 = lambda x: sum(a ** (3+arange(K)) * abs(x[:K])**P2)
f = lambda x: f1(x) + f2(x)

df1 = lambda x: P * sign(x) * abs(x) ** (P-1) 
df2 = lambda x: hstack((a ** (3+arange(K)) * P2 * abs(x[:K]) ** (P2-1)*sign(x[:K]), zeros(N-K)))
df = lambda x: df1(x) + df2(x)

#f, df = f1, df1

#objFun = lambda x: sum(x**2)

x0 = cos(1+asfarray(range(N)))


#OPTIONAL: user-supplied gradient/subgradient
#p.df = lambda x: 1.2 ** arange(len(x)) * 2*x#sign(x)


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
#solvers = ['r2']
#solvers = ['scipy_slsqp']
#solvers = ['algencan']
#solvers = ['ipopt']
colors = ['r', 'b', 'k', 'g']
maxIter = 1000
for i, solver in enumerate(solvers):
    p = NSP(f, x0, df=df, xtol = 1e-7, maxIter = maxIter, maxTime=150, ftol=1e-7)
    #p.checkdf()
    r = p.solve(solver, maxVectorNum=35, iprint=1, showLS=0, plot=0, color=colors[i], show=solver==solvers[-1]) # ralg is name of a solver
#p = NSP(f, x0=r.xf, df=df, xtol = 1e-7, maxIter = maxIter, maxTime=150, ftol=1e-7)
#r = p.solve(solver, maxVectorNum=45, iprint=1, showLS=0, plot=0, color=colors[i], show=solver==solvers[-1])
#for i, solver in enumerate(solvers):
#    p2 = NSP(f, r.xf, df=df, xtol = 1e-6, maxIter = 1200, maxTime=150, ftol=1e-6)
#    #p.checkdf()
#    r2 = p2.solve(solver, maxVectorNum=15, iprint=1, showLS=1, plot=0, color=colors[i], show=solver==solvers[-1]) # ralg is name of a solver
#print 'x_opt:\n', r.xf
print 'f_opt:', r.ff  # should print small positive number like 0.00056
