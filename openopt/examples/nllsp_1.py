"""
Let us solve the overdetermined nonlinear equations:
a^2 + b^2 = 15
a^4 + b^4 = 100
a = 3.5

Let us concider the problem as
x[0]**2 + x[1]**2 - 15 = 0
x[0]**4 + x[1]**4 - 100 = 0
x[0] - 3.5 = 0

Now we will solve the one using solver scipy_leastsq
"""
from openopt import NLLSP
from numpy import *

f = lambda x: ((x**2).sum() - 15, (x**4).sum() - 100, x[0]-3.5)
# other possible f assignments:
# f = lambda x: [(x**2).sum() - 15, (x**4).sum() - 100, x[0]-3.5]
#f = [lambda x: (x**2).sum() - 15, lambda x: (x**4).sum() - 100, lambda x: x[0]-3.5]
# f = (lambda x: (x**2).sum() - 15, lambda x: (x**4).sum() - 100, lambda x: x[0]-3.5)
# f = lambda x: asfarray(((x**2).sum() - 15, (x**4).sum() - 100, x[0]-3.5))
#optional: gradient
def df(x):
    r = zeros((3,2))
    r[0,0] = 2*x[0]
    r[0,1] = 2*x[1]
    r[1,0] = 4*x[0]**3
    r[1,1] = 4*x[1]**3
    r[2,0] = 1
    return r

# init esimation of solution - sometimes rather pricise one is very important
x0 = [1.5, 8]

#p = NLLSP(f, x0, diffInt = 1.5e-8, xtol = 1.5e-8, ftol = 1.5e-8)
# or
# p = NLLSP(f, x0)
# or
p = NLLSP(f, x0, df = df, xtol = 1.5e-8, ftol = 1.5e-8)

#optional: user-supplied gradient check:
p.checkdf()
#r = p.solve('scipy_leastsq', plot=1, iprint = -1)
#or using converter lsp2nlp:
r = p.solve('nlp:ralg', iprint = 1, plot=1)
#r = p.solve('nlp:ipopt',plot=1), r = p.solve('nlp:algencan'), r = p.solve('nlp:ralg'), etc
#(some NLP solvers require additional installation)

print 'x_opt:', r.xf # 2.74930862,  +/-2.5597651
print 'funcs Values:', p.f(r.xf) # [-0.888904734668, 0.0678251418575, -0.750691380965]
print 'f_opt:', r.ff, '; sum of squares (should be same value):', (p.f(r.xf) ** 2).sum() # 1.35828942657
