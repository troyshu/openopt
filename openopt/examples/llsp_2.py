__docformat__ = "restructuredtext en"

from numpy import diag,  ones, sin, cos, arange, sqrt, vstack, zeros, dot
from openopt import LLSP, NLP

N = 150
C1 = diag(sqrt(arange(N)))
C2 = (1.5+arange(N)).reshape(1, -1) * (0.8+arange(N)).reshape(-1, 1)
C = vstack((C1, C2))
d = arange(2*N)
lb = -2.0+sin(arange(N))
ub = 5+cos(arange(N))

############################LLSP################################
LLSPsolver = 'bvls'
p = LLSP(C, d, lb=lb, ub=ub)
r = p.solve(LLSPsolver)
#############################NLP################################
NLPsolver = 'scipy_lbfgsb'# you could try scipy_tnc or ralg as well
#NLPsolver = 'scipy_tnc'
p2 = LLSP(C, d, lb=lb, ub=ub)
r2=p2.solve('nlp:'+NLPsolver)
##################################################################
print '###########Results:###########'
print 'LLSP solver '+ LLSPsolver + ':', r.ff
print 'NLP solver '+ NLPsolver + ':', r2.ff
