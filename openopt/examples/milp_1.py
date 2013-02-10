__docformat__ = "restructuredtext en"

from numpy import *
from openopt import MILP

f = [1, 2, 3, 4, 5, 4, 2, 1]

# indexing starts from ZERO!
# while in native lpsolve-python wrapper from 1
# so if you used [5,8] for native lp_solve python binding
# you should use [4,7] instead
intVars = [4, 7]

lb = -1.5 * ones(8)
ub = 15 * ones(8)
A = zeros((5, 8))
b = zeros(5)
for i in xrange(5):
    for j in xrange(8):
        A[i,j] = -8+sin(8*i) + cos(15*j)
    b[i] = -150 + 80*sin(80*i)

p = MILP(f=f, lb=lb, ub=ub, A=A, b=b, intVars=intVars, goal='min')
r = p.solve('lpSolve')
#r = p.solve('glpk', iprint =-1)
#r = p.solve('cplex')

print('f_opt: %f' % r.ff) # 25.801450769161505
print('x_opt: %s' % r.xf) # [ 15. 10.15072538 -1.5 -1.5 -1.  -1.5 -1.5 15.]
