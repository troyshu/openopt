from numpy import log
from openopt import NLP
x0 = [4, 5, 6]
#h = lambda x: log(1+abs(4+x[1]))
#f = lambda x: log(1+abs(x[0]))
f = lambda x: x[0]**4 + x[1]**4 + x[2]**4
df = lambda x: [4*x[0]**3,  4*x[1]**3, 4*x[2]**3]
h = lambda x: [(x[0]-1)**2,  (x[1]-1)**4]
dh = lambda x: [[2*(x[0]-1), 0, 0],  [0, 4*(x[1]-1)**3, 0]]
colors = ['r', 'b', 'g', 'k', 'y']
solvers = ['ralg','scipy_cobyla', 'algencan', 'ipopt', 'scipy_slsqp']
solvers = ['ralg','algencan']
contol = 1e-8
gtol = 1e-8

for i, solver in enumerate(solvers):
    p = NLP(f, x0, df=df, h=h, dh=dh, gtol = gtol, diffInt = 1e-1, contol = contol,  iprint = 1000, maxIter = 1e5, maxTime = 50, maxFunEvals = 1e8, color=colors[i], plot=0, show = i == len(solvers))
    p.checkdh()
    r = p.solve(solver)

#
#x0 = 4
##h = lambda x: log(1+abs(4+x[1]))
##f = lambda x: log(1+abs(x[0]))
#f = lambda x: x**4
#h = lambda x: (x-1)**2
#colors = ['r', 'b', 'g', 'k', 'y']
#solvers = ['ralg','scipy_cobyla', 'algencan', 'ipopt', 'scipy_slsqp']
##solvers = ['algencan']
#contol = 1e-8
#gtol = 1e-8
#for i, solver in enumerate(solvers):
#    p = NLP(f, x0, h=h, gtol = gtol, contol = contol,  iprint = 1000, maxIter = 1e5, maxTime = 50, maxFunEvals = 1e8, color=colors[i], plot=0, show = i == len(solvers))
#    r = p.solve(solver)
