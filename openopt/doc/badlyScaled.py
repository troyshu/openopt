from numpy import *
from openopt import *

coeff = 1e-7

f = lambda x: (x[0]-20)**2+(coeff * x[1] - 80)**2 # objFun
c = lambda x: (x[0]-14)**2-1 # non-lin ineq constraint(s) c(x) <= 0
# for the problem involved: f_opt =25, x_opt = [15.0, 8.0e9]

x0 = [-4,4]
# even modification of stop criteria can't help to achieve the desired solution:
someModifiedStopCriteria = {'gtol': 1e-15,  'ftol': 1e-13,  'xtol': 1e-13, 'maxIter': 1e3}

# using default diffInt = 1e-7 is inappropriate:
p = NLP(f, x0, c=c, iprint = 100, **someModifiedStopCriteria)
r = p.solve('ralg')
print r.ff,  r.xf #  will print something like "6424.9999886000014 [ 15.0000005   4.       ]"
"""
 for to improve the solution we will use
 changing either p.diffInt from default 1e-7 to [1e-7,  1]
 or p.scale from default None to [1,  1e-7]

 latter (using p.scale) is more recommended
 because it affects xtol for those solvers
 who use OO stop criteria
 (ralg, lincher, nsmm, nssolve and mb some others)
  xtol will be compared to scaled x shift:
 is || (x[k] - x[k-1]) * scale || < xtol

 You can define scale and diffInt as
 numpy arrays, matrices, Python lists, tuples
 """
p = NLP(f, x0, c=c, scale = [1,  coeff], iprint = 100, **someModifiedStopCriteria)
r = p.solve('ralg')
print r.ff,  r.xf # "24.999996490694787 [  1.50000004e+01   8.00004473e+09]" - much better
"""
Full Output:
-----------------------------------------------------
solver: ralg   problem: unnamed   goal: minimum
 iter    objFunVal    log10(maxResidual)   
    0  6.976e+03               2.51 
   51  6.425e+03              -6.10 
istop:  4 (|| F[k] - F[k-1] || < ftol)
Solver:   Time Elapsed = 0.16 	CPU Time Elapsed = 0.16
objFunValue: 6424.9999 (feasible, max constraint =  8e-07)
6424.999932 [ 15.0000004   4.       ]
-----------------------------------------------------
solver: ralg   problem: unnamed   goal: minimum
 iter    objFunVal    log10(maxResidual)   
    0  6.976e+03               2.51 
  100  4.419e+01              -5.99 
  200  2.504e+01              -6.10 
  300  2.503e+01              -6.10 
  400  2.503e+01              -6.10 
  500  2.503e+01              -6.10 
  506  2.500e+01              -6.91 
istop:  3 (|| X[k] - X[k-1] || < xtol)
Solver:   Time Elapsed = 1.59 	CPU Time Elapsed = 1.59
objFunValue: 25.000189 (feasible, max constraint =  1.23911e-07)
25.0001894297 [  1.50000001e+01   8.00137858e+08]
"""
