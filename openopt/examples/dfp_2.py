"""
In the DFP example we will search for z=(a, b, c, d) 
that minimizes Sum_i || F(z, X_i) - Y_i ||^2
for the function F: R^2 -> R^2
F(x0, x1) = [
                a^3 + b * x0 + c * x1 + d * (x0^2+x1^2), 
                2*a + 3*b * x0 + 4*c * x1 + 5*d * (x0^2+x1^2)
                ]

Suppose we have the following measurements
X_0 = [0, 1]; Y_0 = [15, 1]
X_1 = [1, 0]; Y_1 = [8, 16]
X_2 = [1, 1]; Y_2 = [80, 800]
X_3 = [3, 4]; Y_3 = [100, 120]
X_4 = [1, 15]; Y_4 = [150, 1500]

subjected to a>=4, c<=30 
(we could handle other constraints as well: Ax <= b, Aeq x = beq, c(x) <= 0, h(x) = 0)
"""
from openopt import DFP
from numpy import *

f = lambda z, X: (z[0]**3 + z[1]*X[0] + z[2]*X[1] + z[3]*(X[0]+X[1])**2, 2*z[0] + 3*z[1]*X[0] + 4*z[2]*X[1] + 5*z[3]*(X[0]+X[1])**2)
initEstimation = [0] * 4 # start point for solver: [0, 0, 0, 0]
X = ([0, 1], [1, 0], [1, 1], [3, 4], [1, 15]) # list, tuple, numpy array or array-like are OK as well
Y = [[15, 1], [8, 16], [80, 800], [100, 120], [150, 1500]]
lb = [4, -inf, -inf, -inf]
ub = [inf, inf, 30, inf]
p = DFP(f, initEstimation, X, Y, lb=lb, ub=ub)

# optional: derivatives
#p.df = lambda z, X: ([3*z[0]**2, X[0], X[1], (X[0]+X[1])**2], [2, 3*X[0], 4*X[1], 5*(X[0]+X[1])**2])

r = p.solve('nlp:ralg', plot=1, iprint = 10)
print('solution: '+str(r.xf)+'\n||residuals||^2 = '+str(r.ff)+'\nresiduals: ')
rr = [array(f(p.xf, X[i]))-array(Y[i]) for i in xrange(len(Y))]
print rr
#solution: [  3.99999936  -1.9497013   18.25467922   0.24926213]
#||residuals||^2 = 653639.695247
#residuals: 
#[array([ 67.50391074,  81.26502627]), array([ 54.29953022, -12.60279452]), array([   1.30199584, -719.84514565]), array([  43.38342685,  223.59677693]), array([ 249.68156224,  -83.51282276])]

