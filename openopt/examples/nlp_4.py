from openopt import NLP

from numpy import cos, arange, ones, asarray, zeros, mat, array, sin, cos, sign, abs, inf
N = 1500
K = 50
# 1st arg - objective function
# 2nd arg - x0

p = NLP(lambda x: (abs(x-5)).sum(), 8*cos(arange(N)), iprint = 50, maxIter = 1e3)

# f(x) gradient (optional):
p.df = lambda x: sign(x-5)

p.lb = 5*ones(N) + sin(arange(N)) - 0.1
p.ub = 5*ones(N) + sin(arange(N)) + 0.1
p.lb[:N/4] = -inf
p.ub[3*N/4:] = inf

#p.ub[4] = 4
#p.lb[5], p.ub[5] = 8, 15

#A = zeros((K, N))
#b = zeros(K)
#for i in xrange(K):
#    A[i] = 1+cos(i+arange(N))
#    b[i] = sin(i)
#p.A = A
#p.b = b

#p.Aeq = zeros(p.n)
#p.Aeq[100:102] = 1
#p.beq = 11

p.contol = 1e-6
p.plot = 1
p.maxFunEvals = 1e7
p.name = 'nlp_4'
p.debug=1
solver = 'ralg'
solver = 'gsubg'
#solver = 'scipy_cobyla'
solver = 'algencan'
#solver = 'ipopt'

r = p.solve(solver, xlabel = 'time', fTol = 10, debug=0, maxIter = 5500, plot=0, maxTime=1000, ftol = 1e-8, xtol = 1e-6, iprint=1, showLS=0, showFeas=0, show_hs=0)

