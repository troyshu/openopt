from openopt import NLP
from numpy import cos, arange, ones, asarray, abs, zeros
N = 30
M = 5
ff = lambda x: ((x-M)**2).sum()
p = NLP(ff, cos(arange(N)))

def df(x):
    r = 2*(x-M)
    r[0] += 15 #incorrect derivative
    r[8] += 80 #incorrect derivative
    return r
p.df =  df

p.c = lambda x: [2* x[0] **4-32, x[1]**2+x[2]**2 - 8]

def dc(x):
    r = zeros((2, p.n))
    r[0,0] = 2 * 4 * x[0]**3
    r[1,1] = 2 * x[1]
    r[1,2] = 2 * x[2] + 15 #incorrect derivative
    return r
p.dc = dc

p.h = lambda x: (1e1*(x[-1]-1)**4, (x[-2]-1.5)**4)

def dh(x):
    r = zeros((2, p.n))
    r[0,-1] = 1e1*4*(x[-1]-1)**3
    r[1,-2] = 4*(x[-2]-1.5)**3 + 15 #incorrect derivative
    return r
p.dh = dh

p.checkdf()
p.checkdc()
p.checkdh()
"""
you can use p.checkdF(x) for other point than x0 (F is f, c or h)
p.checkdc(myX)
or
p.checkdc(x=myX)
values with difference greater than
maxViolation (default 1e-5)
will be shown
p.checkdh(maxViolation=1e-4)
p.checkdh(myX, maxViolation=1e-4)
p.checkdh(x=myX, maxViolation=1e-4)

#################################################################################
Typical output (unfortunately, in terminal or other IDEs the blank space used in strings separation can have other lengths):
Note that RD (relative difference) is defined as
int(ceil(log10(abs(Diff) / maxViolation + 1e-150)))
where
Diff = 1 - (info_user+1e-8)/(info_numerical + 1e-8)

OpenOpt checks user-supplied gradient df (shape: (30,) )
according to:
    prob.diffInt = [  1.00000000e-07]
    |1 - info_user/info_numerical| <= prob.maxViolation = 0.01
df num         user-supplied     numerical               RD
    0             +7.000e+00     -8.000e+00              3
    8             -2.291e+00     -1.029e+01              2
max(abs(df_user - df_numerical)) = 14.9999995251
(is registered in df number 0)
========================
OpenOpt checks user-supplied gradient dc (shape: (2, 30) )
according to:
    prob.diffInt = [  1.00000000e-07]
    |1 - info_user/info_numerical| <= prob.maxViolation = 0.01
dc num   i,j:dc[i]/dx[j]   user-supplied     numerical               RD
    32             1 / 2         +1.417e+01     -8.323e-01              4
max(abs(dc_user - dc_numerical)) = 14.9999999032
(is registered in dc number 32)
========================
OpenOpt checks user-supplied gradient dh (shape: (2, 30) )
according to:
    prob.diffInt = [  1.00000000e-07]
    |1 - info_user/info_numerical| <= prob.maxViolation = 0.01
dh num   i,j:dh[i]/dx[j]   user-supplied     numerical               RD
    58            1 / 28         -4.474e+01     -5.974e+01              2
max(abs(dh_user - dh_numerical)) = 14.9999962441
(is registered in dh number 58)
========================

"""
