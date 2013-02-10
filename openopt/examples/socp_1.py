"""
OpenOpt SOCP example
for the problem http://openopt.org/images/2/28/SOCP.png
"""

from numpy import *
from openopt import SOCP

f = array([-2, 1, 5])

C0 = mat('-13 3 5; -12 12 -6')
d0 = [-3, -2]
q0 = array([-12, -6, 5])
s0 = -12

C1 = mat('-3 6 2; 1 9 2; -1 -19 3')
d1 = [0, 3, -42]
q1 = array([-3, 6, -10])
s1 = 27

p = SOCP(f,  C=[C0, C1],  d=[d0, d1], q=[q0, q1], s=[s0, s1]) 
# you could add lb <= x <= ub, Ax <= b, Aeq x = beq constraints 
# via p = SOCP(f,  ..., A=A, b=b, Aeq=Aeq, beq=beq,lb=lb, ub=ub)
r = p.solve('cvxopt_socp')
x_opt, f_opt = r.xf,  r.ff
print(' f_opt: %f    x_opt: %s' % (f_opt, x_opt))
# f_opt: -38.346368    x_opt: [-5.01428121 -5.76680444 -8.52162517]
