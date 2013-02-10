"""
this is an example of using d2f - Hesse matrix (2nd derivatives)
d2c, d2h, d2l are intended to be implemented soon 
and to be connected to ALGENCAN and/or CVXOPT 
and/or other NLP solvers

//Dmitrey
"""
from openopt import NLP
from numpy import cos, arange, ones, asarray, abs, zeros, diag
N = 300
M = 5
ff = lambda x: ((x-M)**4).sum()
p = NLP(ff, cos(arange(N)))
p.df =  lambda x: 4*(x-M)**3
p.d2f = lambda x: diag(12*(x-M)**2)
# other valid assignment: 
# p = NLP(lambda x: ((x-M)**4).sum(), cos(arange(N)), df =  lambda x: 4*(x-M)**3, d2f = lambda x: diag(12*(x-M)**2))
# or 
# p = NLP(x0 = cos(arange(N)), f = lambda x: ((x-M)**4).sum(), df =  lambda x: 4*(x-M)**3, d2f = lambda x: diag(12*(x-M)**2))
r = p.solve('scipy_ncg')
print('objfunc val: %e' % r.ff) # it should be a small positive like 5.23656378549e-08

