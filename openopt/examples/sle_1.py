__docformat__ = "restructuredtext en"

from numpy import *
from openopt import SLE

N = 1000
C = empty((N,N))
d =  1.5+80*sin(arange(N))

for j in xrange(N):
    C[j] = 8*sin(4.0+arange(j, N+j)**2) + 15*cos(j)

p = SLE(C, d)
#r = p.solve('defaultSLEsolver'), or just
r = p.solve()

print('max residual: %e' % r.ff)
#print('solution: %s' % r.xf)

