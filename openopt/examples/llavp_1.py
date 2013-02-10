from numpy import empty, sin, cos, arange, ones
from openopt import LLAVP

M, N = 150, 15
C = empty((M,N))
d =  empty(M)

for j in range(M):
    d[j] = 1.5*N+80*sin(j)
    C[j] = 8*sin(4.0+arange(N)) + 15*cos(j)

lb = sin(arange(N))
ub = lb + 1
p = LLAVP(C, d, lb=lb, ub=ub, dump = 10,  X = ones(N),  maxIter = 1e4, maxFunEvals = 1e100)

#optional: plot
p.plot=1

r = p.solve('nsp:ralg', iprint = 100, maxIter = 1000)
#r = p.solve('nsp:ipopt', iprint = 100, maxIter = 1000)


print('f_opt: %f' % r.ff)
#print 'x_opt:', r.xf

