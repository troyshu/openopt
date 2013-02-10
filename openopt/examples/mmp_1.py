"""
Example of solving Mini-Max Problem
via converter to NLP

latter works via solving NLP
t -> min
subjected to
t >= f0(x)
t >= f1(x)
...
t >= fk(x)

Splitting f into separate funcs could benefit some solvers
(ralg, algencan; see NLP docpage for more details)
but is not implemented yet
"""
from numpy import *
from openopt import *

n = 15

f1 = lambda x: (x[0]-15)**2 + (x[1]-80)**2 + (x[2:]**2).sum()
f2 = lambda x: (x[1]-15)**2 + (x[2]-8)**2 + (abs(x[3:]-100)**1.5).sum()
f3 = lambda x: (x[2]-8)**2 + (x[0]-80)**2 + (abs(x[4:]+150)**1.2).sum()
f = [f1, f2, f3]

# you can define matrices as numpy array, matrix, Python lists or tuples

#box-bound constraints lb <= x <= ub
lb = [0]*n
ub = [15,  inf,  80] + (n-3) * [inf]

# linear ineq constraints A*x <= b
A = array([[4,  5,  6] + [0]*(n-3), [80,  8,  15] + [0]*(n-3)])
b = [100,  350]

# non-linear eq constraints Aeq*x = beq
Aeq = [15,  8,  80] + [0]*(n-3)
beq = 90

# non-lin ineq constraints c(x) <= 0
c1 = lambda x: x[0] + (x[1]/8) ** 2 - 15
c2 = lambda x: x[0] + (x[2]/80) ** 2 - 15
c = [c1, c2]
#or: c = lambda x: (x[0] + (x[1]/8) ** 2 - 15, x[0] + (x[2]/80) ** 2 - 15)

# non-lin eq constraints h(x) = 0
h = lambda x: x[0]+x[2]**2 - x[1]

x0 = [0, 1, 2] + [1.5]*(n-3)
p = MMP(f,  x0,  lb = lb,  ub = ub,   A=A,  b=b,   Aeq = Aeq,  beq = beq,  c=c,  h=h, xtol = 1e-6, ftol=1e-6)
# optional, matplotlib is required:
p.plot=1
r = p.solve('nlp:ipopt', iprint=50, maxIter=1e3)
print 'MMP result:',  r.ff

### let's check result via comparison with NSP solution
F= lambda x: max([f1(x),  f2(x),  f3(x)])
p = NSP(F,  x0, iprint=50, lb = lb, ub = ub,  c=c,  h=h,  A=A,  b=b,  Aeq = Aeq,  beq = beq, xtol = 1e-6, ftol=1e-6)
r_nsp = p.solve('ipopt', maxIter = 1e3)
print 'NSP result:',  r_nsp.ff,  'difference:', r_nsp.ff - r.ff
"""
-----------------------------------------------------
solver: ipopt   problem: unnamed   goal: minimax
 iter    objFunVal    log10(maxResidual)
    0  1.196e+04               1.89
   50  1.054e+04              -8.00
  100  1.054e+04              -8.00
  150  1.054e+04              -8.00
  161  1.054e+04              -6.10
istop:  1000
Solver:   Time Elapsed = 0.93   CPU Time Elapsed = 0.88
objFunValue: 10536.481 (feasible, max constraint =  7.99998e-07)
MMP result: 10536.4808622
-----------------------------------------------------
solver: ipopt   problem: unnamed   goal: minimum
 iter    objFunVal    log10(maxResidual)
    0  1.196e+04               1.89
   50  1.054e+04              -4.82
  100  1.054e+04             -10.25
  150  1.054e+04             -15.35
StdOut: Problem solved
[PyIPOPT] Ipopt will use Hessian approximation.
[PyIPOPT] nele_hess is 0
  192  1.054e+04             -13.85
istop:  1000
Solver:   Time Elapsed = 2.42   CPU Time Elapsed = 2.42
objFunValue: 10536.666 (feasible, max constraint =  1.42109e-14)
NSP result: 10536.6656339 difference: 0.184771728482
"""
