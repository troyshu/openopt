"""
Example:
(x0-5)^2 + (x2-5)^2 + ... +(x149-5)^2 -> min

subjected to

# lb<= x <= ub:
x4 <= 4
8 <= x5 <= 15

# Ax <= b
x0+...+x149 >= 825
x9 + x19 <= 3
x10+x11 <= 9

# Aeq x = beq
x100+x101 = 11

# c(x) <= 0
2*x0^4-32 <= 0
x1^2+x2^2-8 <= 0

# h(x) = 0
(x[149]-1)**6 = 0
(x[148]-1.5)**6 = 0
"""

from openopt import NLP
from numpy import cos, arange, ones, asarray, zeros, mat, array
N = 150

# objective function:
f = lambda x: ((x-5)**2).sum()

# objective function gradient (optional):
df = lambda x: 2*(x-5)

# start point (initial estimation)
x0 = 8*cos(arange(N))

# c(x) <= 0 constraints
c = [lambda x: 2* x[0] **4-32, lambda x: x[1]**2+x[2]**2 - 8]

# dc(x)/dx: non-lin ineq constraints gradients (optional):
dc0 = lambda x: [8 * x[0]**3] + [0]*(N-1)
dc1 = lambda x: [0, 2 * x[1],  2 * x[2]] + [0]*(N-3)
dc = [dc0, dc1]

# h(x) = 0 constraints
def h(x):
    return (x[N-1]-1)**6, (x[N-2]-1.5)**6
    # other possible return types: numpy array, matrix, Python list, tuple
# or just h = lambda x: [(x[149]-1)**6, (x[148]-1.5)**6]


# dh(x)/dx: non-lin eq constraints gradients (optional):
def dh(x):
    r = zeros((2, N))
    r[0, -1] = 6*(x[N-1]-1)**5
    r[1, -2] = 6*(x[N-2]-1.5)**5
    return r
    
# lower and upper bounds on variables
lb = -6*ones(N)
ub = 6*ones(N)
ub[4] = 4
lb[5], ub[5] = 8, 15

# general linear inequality constraints
A = zeros((3, N))
A[0, 9] = 1
A[0, 19] = 1
A[1, 10:12] = 1
A[2] = -ones(N)
b = [7, 9, -825]

# general linear equality constraints
Aeq = zeros(N)
Aeq[100:102] = 1
beq = 11

# required constraints tolerance, default for NLP is 1e-6
contol = 1e-7

# If you use solver algencan, NB! - it ignores xtol and ftol; using maxTime, maxCPUTime, maxIter, maxFunEvals, fEnough is recommended.
# Note that in algencan gtol means norm of projected gradient of  the Augmented Lagrangian
# so it should be something like 1e-3...1e-5
gtol = 1e-7 # (default gtol = 1e-6)

# Assign problem:
# 1st arg - objective function
# 2nd arg - start point
p = NLP(f, x0, df=df,  c=c,  dc=dc, h=h,  dh=dh,  A=A,  b=b,  Aeq=Aeq,  beq=beq,  
        lb=lb, ub=ub, gtol=gtol, contol=contol, iprint = 50, maxIter = 10000, maxFunEvals = 1e7, name = 'NLP_1')

#optional: graphic output, requires pylab (matplotlib)
p.plot = True

#optional: user-supplied 1st derivatives check
p.checkdf()
p.checkdc()
p.checkdh()

solver = 'ralg'
#solver = 'algencan'
#solver = 'ipopt'
#solver = 'scipy_slsqp'

# solve the problem

r = p.solve(solver, plot=0) # string argument is solver name


# r.xf and r.ff are optim point and optim objFun value
# r.ff should be something like 132.05
