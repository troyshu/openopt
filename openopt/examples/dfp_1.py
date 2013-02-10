"""
In the DFP example we will search for z=(a, b, c, d) 
that minimizes Sum_i || F(z, X_i) - Y_i ||^2
for the function
F(x0, x1) = a^3 + b * x0 + c * x1 + d * (x0^2+x1^2) 

Suppose we have the following measurements
X_0 = [0, 1]; Y_0 = 15
X_1 = [1, 0]; Y_1 = 8
X_2 = [1, 1]; Y_2 = 80
X_3 = [3, 4]; Y_3 = 100
X_4 = [1, 15]; Y_4 = 150

subjected to a>=4, c<=30 
(we could handle other constraints as well: Ax <= b, Aeq x = beq, c(x) <= 0, h(x) = 0)
"""
from openopt import DFP
from numpy import inf

f = lambda z, X: z[0]**3 + z[1]*X[0] + z[2]*X[1] + z[3]*(X[0]+X[1])**2
initEstimation = [0] * 4 # start point for solver: [0, 0, 0, 0]
X = ([0, 1], [1, 0], [1, 1], [3, 4], [1, 15]) # list, tuple, numpy array or array-like are OK as well
Y = [15, 8, 80, 100, 150]
lb = [4, -inf, -inf, -inf]
ub = [inf, inf, 30, inf]
p = DFP(f, initEstimation, X, Y, lb=lb, ub=ub)

# optional: derivative
p.df = lambda z, X: [3*z[0]**2, X[0], X[1], (X[0]+X[1])**2]

r = p.solve('nlp:ralg', plot=0, iprint = 10)
print('solution: '+str(r.xf)+'\n||residuals||^2 = '+str(r.ff)+'\nresiduals: '+str([f(p.xf, X[i])-Y[i] for i in xrange(len(Y))]))
#solution: [  3.99999936   5.99708861 -12.25696614   1.04221073]
#||residuals||^2 = 5992.63887806
#residuals: [37.785213926923028, 63.039268675751572, -18.091065285780857, -15.968303801844399, 2.9485118103557397]

