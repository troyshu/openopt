from openopt import NLP
from numpy import cos, arange, ones, asarray, zeros, mat, array

N = 50
# objfunc:
# (x0-1)^4 + (x2-1)^4 + ... +(x49-1)^4 -> min (N=nVars=50)
f = lambda x : ((x-1)**4).sum()
x0 = cos(arange(N))
p = NLP(f, x0, maxIter = 1e3, maxFunEvals = 1e5)

# f(x) gradient (optional):
p.df = lambda x: 4*(x-1)**3


# lb<= x <= ub:
# x4 <= -2.5
# 3.5 <= x5 <= 4.5
# all other: lb = -5, ub = +15
p.lb = -5*ones(N)
p.ub = 15*ones(N)
p.ub[4] = -2.5
p.lb[5], p.ub[5] = 3.5, 4.5



# Ax <= b
# x0+...+xN>= 1.1*N
# x9 + x19 <= 1.5
# x10+x11 >= 1.6
p.A = zeros((3, N))
p.A[0, 9] = 1
p.A[0, 19] = 1
p.A[1, 10:12] = -1
p.A[2] = -ones(N)
p.b = [1.5, -1.6, -1.1*N]
# you can use any types of A, Aeq, b, beq:
# Python list, numpy.array, numpy.matrix, even Python touple
# so p.b = array([1.5, -1.6, -825]) or p.b = (1.5, -1.6, -825) are valid as well


# Aeq x = beq
# x20+x21 = 2.5
p.Aeq = zeros(N)
p.Aeq[20:22] = 1
p.beq = 2.5


# non-linear inequality constraints c(x) <= 0
# 2*x0^4 <= 1/32
# x1^2+x2^2 <= 1/8
# x25^2 +x25*x35 + x35^2<= 2.5

p.c = lambda x: [2* x[0] **4-1./32, x[1]**2+x[2]**2 - 1./8, x[25]**2 + x[35]**2 + x[25]*x[35] -2.5]
# other valid c:
# p.c = [lambda x: c1(x), lambda x : c2(x), lambda x : c3(x)]
# p.c = (lambda x: c1(x), lambda x : c2(x), lambda x : c3(x))
# p.c = lambda x: numpy.array(c1(x), c2(x), c3(x))
# def c(x):
#      return c1(x), c2(x), c3(x)
# p.c = c


# dc(x)/dx: non-lin ineq constraints gradients (optional):
def DC(x):
    r = zeros((3, N))
    r[0,0] = 2 * 4 * x[0]**3
    r[1,1] = 2 * x[1]
    r[1,2] = 2 * x[2]
    r[2,25] = 2*x[25] + x[35]
    r[2,35] = 2*x[35] + x[25]
    return r
p.dc = DC

# non-linear equality constraints h(x) = 0
# 1e6*(x[last]-1)**4 = 0
# (x[last-1]-1.5)**4 = 0

p.h = lambda x: (1e4*(x[-1]-1)**4, (x[-2]-1.5)**4)
# dh(x)/dx: non-lin eq constraints gradients (optional):
def DH(x):
    r = zeros((2, p.n))
    r[0, -1] = 1e4*4 * (x[-1]-1)**3
    r[1, -2] = 4 * (x[-2]-1.5)**3
    return r
p.dh = DH

p.contol = 1e-3 # required constraints tolerance, default for NLP is 1e-6

# for ALGENCAN solver gtol is the only one stop criterium connected to openopt
# (except maxfun, maxiter)
# Note that in ALGENCAN gtol means norm of projected gradient of  the Augmented Lagrangian
# so it should be something like 1e-3...1e-5
p.gtol = 1e-5 # gradient stop criterium (default for NLP is 1e-6)


# see also: help(NLP) -> maxTime, maxCPUTime, ftol and xtol
# that are connected to / used in lincher and some other solvers

# optional: check of user-supplied derivatives
p.checkdf()
p.checkdc()
p.checkdh()

# last but not least:
# please don't forget,
# Python indexing starts from ZERO!!

p.plot = 0
p.iprint = 0
p.df_iter = 4
p.maxTime = 4000
p.debug=1
#r = p.solve('algencan')

r = p.solve('ralg')
#r = p.solve('lincher')

"""
typical output:
OpenOpt checks user-supplied gradient df (size: (50,))
according to:
prob.diffInt = 1e-07
prob.check.maxViolation = 1e-05
max(abs(df_user - df_numerical)) = 2.50111104094e-06
(is registered in df number 41)
sum(abs(df_user - df_numerical)) = 4.45203815948e-05
========================
OpenOpt checks user-supplied gradient dc (size: (50, 3))
according to:
prob.diffInt = 1e-07
prob.check.maxViolation = 1e-05
max(abs(dc_user - dc_numerical)) = 1.20371180401e-06
(is registered in dc number 0)
sum(abs(dc_user - dc_numerical)) = 1.60141862837e-06
========================
OpenOpt checks user-supplied gradient dh (size: (50, 2))
according to:
prob.diffInt = 1e-07
prob.check.maxViolation = 1e-05
dh num   i,j:dh[i]/dx[j]   user-supplied     numerical         difference
     98            49 / 0         -1.369e+04     -1.369e+04     -2.941e-03
max(abs(dh_user - dh_numerical)) = 0.00294061290697
(is registered in dh number 98)
sum(abs(dh_user - dh_numerical)) = 0.00294343472179
========================
starting solver ALGENCAN (GPL  license)  with problem  unnamed
solver ALGENCAN has finished solving the problem unnamed
istop:  1000
Solver:   Time elapsed = 0.34   CPU Time Elapsed = 0.34
objFunValue: 190.041570332 (feasible, max constraint =  0.000677961)
"""
