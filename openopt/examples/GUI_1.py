"""
OpenOpt GUI:
     function manage() usage example
"""

from openopt import NLP, manage
from numpy import cos, arange, ones, asarray, abs, zeros
N = 50
M = 5
p = NLP(lambda x: ((x-M)**2).sum(), cos(arange(N)))
p.lb, p.ub = -6*ones(N), 6*ones(N)
p.lb[3] = 5.5
p.ub[4] = 4.5
p.c = lambda x: [2* x[0] **4-32, x[1]**2+x[2]**2 - 8]
p.h = (lambda x: 1e1*(x[-1]-1)**4, lambda x: (x[-2]-1.5)**4)

"""
minTime is used here
for to provide enough time for user
to play with GUI
"""

minTime = 1.5 # sec
p.name = 'GUI_example'
p.minTime = minTime

"""
hence maxIter, maxFunEvals etc
will not trigger till minTime

only same iter point x_k-1=x_k
or some coords = nan
can stop calculations

other antistop criteria: minFunEvals, minIter, minCPUTime
however, some solvers cannot handle them
 """

# start=True means don't wait for user to press "Run"
r = manage(p,'ralg', plot=1, start=True)
"""
   or calling manage() as filed of p:
r = p.manage('algencan', plot=1)
"""
if r is not None:
    # r is None if user has pressed "Exit" button
    print 'objfunc val:', r.ff

