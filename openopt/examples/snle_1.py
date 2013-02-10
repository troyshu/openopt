"""
Solving system of equations:
x[0]**3+x[1]**3-9 = 0
x[0]-0.5*x[1] = 0
cos(x[2])+x[0]-1.5 = 0
"""

from openopt import SNLE
from numpy import asfarray, zeros, cos, sin

#f = lambda x: (x[0]**3+x[1]**3-9, x[0]-0.5*x[1], cos(x[2])+x[0]-1.5)
# or:
f = (lambda x: x[0]**3+x[1]**3-9, lambda x: x[0]-0.5*x[1], lambda x: cos(x[2])+x[0]-1.5)
# Python list, numpy.array are allowed as well:
#f = lambda x: [x[0]**3+x[1]**3-9, x[0]-0.5*x[1], cos(x[2])+x[0]-1.5]
#or f = lambda x: asfarray((x[0]**3+x[1]**3-9, x[0]-0.5*x[1], cos(x[2])+x[0]-1.5))

# start point
x0 = [8,15, 80]

#optional: gradient
df = (lambda x: [3*x[0]**2, 3*x[1]**2,  0], lambda x: [1, -0.5,  0], lambda x:[1, 0, -sin(x[2])])

#w/o gradient:
#p = NLSP(f, x0)
p = SNLE(f, x0, df = df)

#optional: user-supplied gradient check:
#p.checkdf()

#optional: graphical output, requires matplotlib installed
p.plot = 1

p.maxFunEvals = 1e5
p.iprint = 10

#r = p.solve('scipy_fsolve')
#r = p.solve('nssolve')
#or using converter nlsp2nlp, try to minimize sum(f_i(x)^2):
r = p.solve('nlp:ralg', plot=1)

print('solution: %s' % r.xf)
print('max residual: %e' % r.ff)
###############################
#should print:
#solution: [  1.           2.          55.50147021] (3rd coord may differ due to cos is periodic)
#max residual: 2.72366951215e-09
