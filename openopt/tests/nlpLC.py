from openopt import NLP
from numpy import cos, arange, ones, asarray, zeros, mat, array
N = 100

# objective function:
f = lambda x: ((x-5)**2).sum()

# objective function gradient (optional):
df = lambda x: 2*(x-5)

# start point (initial estimation)
x0 = 8*cos(arange(N))

# c(x) <= 0 constraints
def c(x):
    r = zeros(N-1)
    for i in xrange(N-1):
        r[i] = x[i]**2 + x[i+1]**2 - 1
    return r

def dc(x):
    r = zeros((N-1, N))
    for i in xrange(N-1):
        r[i, i] = 2*x[i]
        r[i, i+1] = 2*x[i+1]
    return r

#    counter = 0
#    r = zeros(N*(N-1)/2)
#    for i in xrange(N):
#        for j in xrange(i):
#            r[counter] = x[i]**2 + x[j]**2 - 1
#            counter += 1
#    return r

lb = -6*ones(N)
ub = 6*ones(N)
ub[4] = 4
lb[5], ub[5] = 8, 15

# general linear inequality constraints
#A = zeros((3, N))
#A[0, 9] = 1
#A[0, 19] = 1
#A[1, 10:12] = 1
#A[2] = -ones(N)
#b = [7, 9, -825]
A, b=None, None


# general linear equality constraints
#Aeq = zeros(N)
#Aeq[10:12] = 1
#beq = 11
Aeq, beq = None, None

# required constraints tolerance, default for NLP is 1e-6
contol = 1e-7

# If you use solver algencan, NB! - it ignores xtol and ftol; using maxTime, maxCPUTime, maxIter, maxFunEvals, fEnough is recommended.
# Note that in algencan gtol means norm of projected gradient of  the Augmented Lagrangian
# so it should be something like 1e-3...1e-5
gtol = 1e-7 # (default gtol = 1e-6)

# Assign problem:
# 1st arg - objective function
# 2nd arg - start point

p = NLP(f, x0, df=df,  c=c, dc=dc,   
        gtol=gtol, contol=contol, iprint = 50, maxIter = 10000, maxFunEvals = 1e7, name = 'NLP_1')

#p = NLP(f, x0, df=df,  Aeq=Aeq, beq=beq, 
#        gtol=gtol, contol=contol, iprint = 50, maxIter = 10000, maxFunEvals = 1e7, name = 'NLP_1')

#p = NLP(f, x0, df=df,  lb=lb, ub=ub, gtol=gtol, contol=contol, iprint = 50, maxIter = 10000, maxFunEvals = 1e7, name = 'NLP_1')

#optional: graphic output, requires pylab (matplotlib)
p.plot = 0

#optional: user-supplied 1st derivatives check
p.checkdf()
p.checkdc()
p.checkdh()


def MyIterFcn(p):
    return 0
#    print 'Iteration',p.iter
#    if p.iter == 50:
#       p.user.mylist.append(p.xk.copy()) 
#    return 0

p.user.mylist = []
# solve the problem
#p.debug=1
solver = 'algencan'
solver = 'ralg'
#solver = 'ipopt'
#solver = 'scipy_slsqp'

p.debug=1
r = p.solve(solver, showLS=0, iprint=10, maxTime = 50,  callback = MyIterFcn) # string argument is solver name
#r = p.solve('r2', iprint = 1, plot=0, showLS=1, maxIter=480)

# r.xf and r.ff are optim point and optim objFun value
# r.ff should be something like 132.0522 
