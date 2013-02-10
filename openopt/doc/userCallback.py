"""
usage:
p = someOOclass(..., callback=MyIterFcn, ...)
or
p = ...
p.callback = MyIterFcn
or p.callback = (MyIterFcn1, MyIterFcn2, MyIterFcn3, ..., MyIterFcnN)
or p.callback = [MyIterFcn1, MyIterFcn2, MyIterFcn3, ..., MyIterFcnN]

each user-defined function MyIterFunc should return one of the following:

1. a flag value - 0, 1, True, False
flag = True or 1 means user want to stop calculations
 (r.istop=80, r.msg = 'user-defined' )

2. someRealValue like 15 or 80.15 or 1.5e4 (r.istop=someRealValue, r.msg = 'user-defined')

3. Python list (or tuple) - [istop, msg] (r.istop=istop, r.msg=msg)

works for ralg and lincher, but may doesn't work for some other solvers
(like scipy_cobyla, that has neither native callback nor call gradient)
"""

def MyIterFcn(p):
    # observing non-feasible ralg iter points

    if p.rk > p.contol: # p.rk is current iter max residual
        print '--= non-feasible ralg iter =--'
        print 'itn:',  p.iter
        #however, I inted to change p.iter to p.iter in OpenOpt code soon

        print 'curr f:',  p.fk
        # print 'curr x[:8]:',  p.xk[:8]
        print 'max constraint value',  p.rk

    """
    BTW you can store data in any unique field of p
    for example
    if some_cond:  p.JohnSmith = 15
    else: p.JohnSmith = 0

    However, special field "user" is intended for the purpose:
    p.user.mydata1 = (something)
    # or, for another example:
    if p.iter == 0: p.user.mylist = []
    p.user.mylist.append(something)
    """

    if p.fk < 1.5 and p.rk < p.contol:
        #NB! you could use p.fEnough = 15, p.contol=1e-5 in prob assignment instead
        return (15, 'value obtained is enough' )
        # or
        # return 15 (hence r.istop=15, r.msg='user-defined')
        # or return True (hence r.istop=80, r.msg='user-defined')
        # or return 1 (hence r.istop = 80, r.msg='user-defined')
    else:
        return False
        # or
        # return 0

from openopt import NSP
from numpy import cos,  asfarray,  arange,  sign
N = 75
f = lambda x: sum(1.2 ** arange(len(x)) * abs(x))
df = lambda x: 1.2 ** arange(len(x)) * sign(x)
x0 = cos(1+asfarray(range(N)))

#non-linear constraint c(x) <= 0:
c = lambda x: abs(x[4]-0.8) + abs(x[5]-1.5) - 0.015 

p = NSP(f,  x0,  df=df,  c=c, callback=MyIterFcn,  contol = 1e-5,  maxIter = 1e4,  iprint = 100, xtol = 1e-8, ftol = 1e-8)

#optional:
#p.plot = 1
r = p.solve('ralg')
print r.xf[:8]

"""
-----------------------------------------------------
solver: ralg   problem: unnamed   goal: minimum
 iter    objFunVal    log10(maxResidual)
    0  2.825e+06               0.02
--= non-feasible ralg iter =--
itn: 0
curr f: [ 2824966.83813157]
max constraint value 1.04116752789
--= non-feasible ralg iter =--
itn: 1
curr f: [ 2824973.2896607]
max constraint value 1.75725959686
--= non-feasible ralg iter =--
itn: 2
curr f: [ 2824966.83813157]
max constraint value 1.04116752789
--= non-feasible ralg iter =--
itn: 3
curr f: [ 2824970.22518437]
max constraint value 0.413756712605
--= non-feasible ralg iter =--
itn: 4
curr f: [ 2824969.02632034]
max constraint value 0.0818395397163
--= non-feasible ralg iter =--
itn: 5
curr f: [ 2824969.37414607]
max constraint value 0.0406513995891
--= non-feasible ralg iter =--
itn: 6
curr f: [ 2824969.20023321]
max constraint value 0.00849187556755
--= non-feasible ralg iter =--
itn: 7
curr f: [ 2824969.20119103]
max constraint value 0.00560799704173
--= non-feasible ralg iter =--
itn: 8
curr f: [ 2824969.2065267]
max constraint value 0.00416641026253
--= non-feasible ralg iter =--
itn: 9
curr f: [ 2824969.22185181]
max constraint value 0.0421905566026
--= non-feasible ralg iter =--
itn: 10
curr f: [ 2824969.2065267]
max constraint value 0.00416641026253
--= non-feasible ralg iter =--
itn: 11
curr f: [ 2824969.20952515]
max constraint value 0.00327175155207
  100  2.665e+04            -100.00
  200  4.845e+03            -100.00
  300  1.947e+02            -100.00
  400  9.298e+01            -100.00
  500  5.160e+01            -100.00
  600  2.600e+01            -100.00
  700  1.070e+01            -100.00
  800  6.994e+00            -100.00
  900  5.375e+00            -100.00
 1000  5.375e+00            -100.00
 1094  5.375e+00            -100.00
istop:  4 (|| F[k] - F[k-1] || < ftol)
Solver:   Time Elapsed = 4.62   CPU Time Elapsed = 4.48
objFunValue: 5.3748608 (feasible, max constraint =  0)
[ -1.06086135e-07   5.65437885e-08  -1.29682567e-07   6.12571176e-09
   7.95256506e-01   1.49731951e+00  -1.42518171e-09   4.15961658e-08]
"""
