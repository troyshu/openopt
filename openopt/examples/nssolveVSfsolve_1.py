"""
Solving system of equations:

x[0]**3 + x[1]**3 - 9 = 0
x[0] - 0.5*x[1] - 0.15*x[2]= 0
sinh(x[2]) + x[0] - 15 = 0
!! with numerical noise 1e-8 !!

Note that both fsolve and nssolve
get same gradient -
if no user-supplied one is available,
then same OO finite-difference one
is used (according to p.diffInt value)

If you have matplotlib installed,
you'll get a figure.
Typical fsolve fails number
(for scipy 0.6.0)
is ~ 10-15%

This test runs ~ a minute on my AMD 3800+
"""
noise = 1e-8

from openopt import SNLE
from numpy import asfarray, zeros, cos, sin, arange, cosh, sinh, log10, ceil, floor, arange, inf, nanmin, nanmax
from time import time
from scipy import rand

x0 = [8, 15, 0.80]
global count1, count2, count3
count1 = count2 = count3 = 0
def Count1():
    global count1; count1 +=1; return 0
def Count2():
    global count2; count2 +=1; return 0
def Count3():
    global count3; count3 +=1; return 0

# ATTENTION!
# when no user-supplied gradient is available
# nssolve can take benefites from splitting funcs
# so using f=[fun1, fun2, fun3, ...](or f=(fun1, fun2, fun3, ...))
# (where each fun is separate equation)
# is more appriciated than
# f = lambda x: (...)
# or def f(x): (...)

f_without_noise = \
[lambda x: x[0]**2+x[1]**2-9, lambda x: x[0]-0.5*x[1] - 0.15*x[2], lambda x: sinh(x[2])+x[0]-15]
def fvn(x):
    r = -inf
    for f in f_without_noise: r = max(r, abs(f(x)))
    return r
f = [lambda x: x[0]**2+x[1]**2-9 + noise*rand(1)+Count1(), lambda x: x[0]-0.5*x[1] - 0.15*x[2] + noise*rand(1)+Count2(), \
lambda x: sinh(x[2])+x[0]-15 + noise*rand(1)+Count3()]# + (2007 * x[3:]**2).tolist()

#optional: gradient
##def DF(x):
##    df = zeros((3,3))
##    df[0,0] = 3*x[0]**2
##    df[0,1] = 3*x[1]**2
##    df[1,0] = 1
##    df[1,1] = -0.5
##    df[1,2] = -0.15
##    df[2,0] = 1
##    df[2,2] = cosh(x[2])
##    return df

N = 100
desired_ftol = 1e-6
assert desired_ftol - noise*len(x0) > 1e-7
#w/o gradient:
scipy_fsolve_failed, fs = 0, []
print '----------------------------------'
print 'desired ftol:', desired_ftol, 'objFunc noise:', noise
############################################################################
print '---------- fsolve fails ----------'
t = time()
print 'N log10(MaxResidual) MaxResidual'
for i in xrange(N):
    p = SNLE(f, x0, ftol = desired_ftol - noise*len(x0), iprint = -1, maxFunEvals = int(1e7))
    r = p.solve('scipy_fsolve')
    v = fvn(r.xf)
    fs.append(log10(v))
    if v > desired_ftol:
        scipy_fsolve_failed += 1
        print i+1, '       %0.2f       ' % log10(v), v
    else:
        print i+1, 'OK'
print 'fsolve time elapsed', time()-t
#print 'fsolve_failed number:', scipy_fsolve_failed , '(from', N, '),', 100.0*scipy_fsolve_failed / N, '%'
print 'counters:', count1, count2, count3
############################################################################
count1 = count2 = count3 = 0
t = time()
print '---------- nssolve fails ---------'
nssolve_failed, ns = 0, []
print 'N log10(MaxResidual) MaxResidual'
for i in xrange(N):
    p = SNLE(f, x0, ftol = desired_ftol - noise*len(x0), iprint = -1, maxFunEvals = int(1e7))
    r = p.solve('nssolve')
    #r = p.solve('nlp:amsg2p')
    #r = p.solve('nlp:ralg')
    v = fvn(r.xf)
    ns.append(log10(v))
    if  v > desired_ftol:
        nssolve_failed += 1
        print i+1, '       %0.2f       ' % log10(v), v
    else:
        print i+1, 'OK'
print 'nssolve time elapsed', time()-t
print 'nssolve_failed number:', nssolve_failed , '(from', N, '),', 100.0 * nssolve_failed / N, '%'
print 'counters:', count1, count2, count3
############################################################################
print '------------ SUMMARY -------------'
print 'fsolve_failed number:', scipy_fsolve_failed , '(from', N, '),', 100.0*scipy_fsolve_failed / N, '%'
print 'nssolve_failed number:', nssolve_failed , '(from', N, '),', 100.0 * nssolve_failed / N, '%'

#try:
from pylab import *
subplot(2,1,1)
grid(1)
title('scipy.optimize fsolve fails to achive desired ftol: %0.1f%%' %(100.0*scipy_fsolve_failed / N))
xmin1, xmax1 = floor(nanmin(fs)), ceil(nanmax(fs))+1
hist(fs, arange(xmin1, xmax1))
#xlabel('log10(maxResidual)')
axvline(log10(desired_ftol), color='green', linewidth=3, ls='--')
[ymin1, ymax1] = ylim()

################
subplot(2,1,2)
grid(1)
title('openopt nssolve fails to achive desired ftol: %0.1f%%' % (100.0*nssolve_failed / N))
xmin2, xmax2 = floor(nanmin(ns)), ceil(nanmax(ns))+1
#hist(ns, 5)
hist(ns, arange(xmin2, xmax2))
xlabel('log10(maxResidual)')
axvline(log10(desired_ftol), color='green', linewidth=3, ls='--')
[ymin2, ymax2] = ylim()

################
xmin, xmax = min(xmin1, xmin2) - 0.1, max(xmax1, xmax2) + 0.1
ymin, ymax = 0, max(ymax1, ymax2) * 1.05
subplot(2,1,1)
xlim(xmin, xmax)
ylim(0, ymax)
subplot(2,1,2)
xlim(xmin, xmax)
show()
#except:
#    pass

