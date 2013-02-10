"""
Example of solving Mini-Max Problem
max { (x0-15)^2+(x1-80)^2, (x1-15)^2 + (x2-8)^2, (x2-8)^2 + (x0-80)^2 } -> min
Currently nsmm is single OO solver available for MMP
It defines function F(x) = max_i {f[i](x)}
and solves NSP F(x) -> min using solver ralg.
It's very far from specialized solvers (like MATLAB fminimax),
but it's better than having nothing at all,
and allows using of nonsmooth and noisy funcs.
This solver is intended to be enhanced in future.
"""
from numpy import *
from openopt import *

f1 = lambda x: (x[0]-15)**2 + (x[1]-80)**2
f2 = lambda x: (x[1]-15)**2 + (x[2]-8)**2
f3 = lambda x: (x[2]-8)**2 + (x[0]-80)**2
f = [f1, f2, f3]

# you can define matrices as numpy array, matrix, Python lists or tuples

#box-bound constraints lb <= x <= ub
lb = [0]*3# i.e. [0,0,0]
ub = [15,  inf,  80]

# linear ineq constraints A*x <= b
A = mat('4 5 6; 80 8 15')
b = [100,  350]

# non-linear eq constraints Aeq*x = beq
Aeq = mat('15 8 80')
beq = 90

# non-lin ineq constraints c(x) <= 0
c1 = lambda x: x[0] + (x[1]/8) ** 2 - 15
c2 = lambda x: x[0] + (x[2]/80) ** 2 - 15
c = [c1, c2]
#or: c = lambda x: (x[0] + (x[1]/8) ** 2 - 15, x[0] + (x[2]/80) ** 2 - 15)

# non-lin eq constraints h(x) = 0
h = lambda x: x[0]+x[2]**2 - x[1]

x0 = [0, 1, 2]
p = MMP(f,  x0,  lb = lb,  ub = ub,   A=A,  b=b,   Aeq = Aeq,  beq = beq,  c=c,  h=h, xtol = 1e-6, ftol=1e-6)
#p = MMP(f,  x0, ftol=1e-8)
# optional, matplotlib is required:
#p.plot=1
r = p.solve('nsmm', iprint=1, NLPsolver = 'ralg', maxIter=1e3, minIter=1e2)
print 'MMP result:',  r.ff

#
### let's check result via comparison with NSP solution
F= lambda x: max([f1(x),  f2(x),  f3(x)])
p = NSP(F,  x0, lb = lb, ub = ub,  c=c,  h=h,  A=A,  b=b,  Aeq = Aeq,  beq = beq, xtol = 1e-6, ftol=1e-6)
#p = NSP(F,  x0)
r_nsp = p.solve('ralg')
#print 'NSP result:',  r_nsp.ff,  'difference:', r_nsp.ff - r.ff
#"""
#starting solver nsmm (license: BSD)  with problem  unnamed
#  iter       ObjFun        log10(maxResidual)
#     0       6.4660e+03   +1.89
#   10       6.4860e+03   -0.68
#   20       6.4158e+03   -1.23
#   30       6.4119e+03   -3.08
#   40       6.3783e+03   -2.95
#   50       6.3950e+03   -4.05
#   60       6.3951e+03   -6.02
#   70       6.3938e+03   -6.02
#   78       6.3936e+03   -6.00
#nsmm has finished solving the problem unnamed
#istop:  3 (|| X[k] - X[k-1] || < xtol)
#Solver:   Time Elapsed = 0.41   CPU Time Elapsed = 0.38
#objFunValue: 6393.6196095379446 (feasible, max constraint =  9.95421e-07)
#MMP result: 6393.6196095379446
#starting solver ralg (license: BSD)  with problem  unnamed
#itn 0 : Fk= 6466.0 MaxResidual= 78.0
#itn 10  Fk: 6485.9728487666425 MaxResidual: 2.07e-01 ls: 2
#itn 20  Fk: 6415.8358391383163 MaxResidual: 5.92e-02 ls: 1
#itn 30  Fk: 6411.9310394431113 MaxResidual: 8.22e-04 ls: 3
#itn 40  Fk: 6378.3471060481961 MaxResidual: 1.12e-03 ls: 2
#itn 50  Fk: 6394.9848936519056 MaxResidual: 8.94e-05 ls: 0
#itn 60  Fk: 6395.054402295913 MaxResidual: 9.57e-07 ls: 1
#itn 70  Fk: 6393.8314202292149 MaxResidual: 9.63e-07 ls: 1
#itn 78  Fk: 6393.6196095379446 MaxResidual: 9.95e-07 ls: 1
#ralg has finished solving the problem unnamed
#istop:  3 (|| X[k] - X[k-1] || < xtol)
#Solver:   Time Elapsed = 0.44   CPU Time Elapsed = 0.32
#objFunValue: 6393.6196095379446 (feasible, max constraint =  9.95421e-07)
#NSP result: 6393.6196095379446 difference: 0.0
#"""
