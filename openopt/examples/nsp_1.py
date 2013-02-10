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

N = 75
objFun = lambda x: sum(1.2 ** arange(len(x)) * abs(x))

x0 = cos(1+asfarray(range(N)))

p = NSP(objFun, x0, maxFunEvals = 1e7, xtol = 1e-8)
#These assignments are also correct:
#p = NLP(objFun, x0=x0)
#p = NLP(f=objFun, x0=x0)
#p = NLP(ftol = 1e-5, contol = 1e-5, f=objFun, x0=x0)


p.maxIter = 5000

#optional (requires matplotlib installed)
#p.plot = 1
#p.graphics.xlabel = 'cputime'#default: time, case-unsensetive; also here maybe 'cputime', 'niter'

#OPTIONAL: user-supplied gradient/subgradient
p.df = lambda x: 1.2 ** arange(len(x)) * sign(x)

r = p.solve('ralg') # ralg is name of a solver
print('x_opt: %s' % r.xf)
print('f_opt: %f' % r.ff)  # should print small positive number like 0.00056
