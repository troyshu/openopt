"""
This is OpenOpt SDP example,
for the problem
http://openopt.org/images/1/12/SDP.png
"""

from numpy import mat
from openopt import SDP
S, d = {}, {}
S[0, 0] = mat('-7 -11; -11 3') # numpy array, array-like, CVXOPT matrix are allowed as well
S[0, 1] = mat('7 -18; -18 8')
S[0, 2] = mat('-2 -8; -8 1')

d[0] = mat('33, -9; -9, 26')

S[1, 0] = mat('-21 -11 0; -11 10 8; 0 8 5')
S[1, 1] = mat('0 10 16; 10 -10 -10; 16 -10 3')
S[1, 2] = mat('-5 2 -17; 2 -6 8; -17 -7 6')

d[1] = mat('14, 9, 40; 9, 91, 10; 40, 10, 15')

p = SDP([1, -1, 1], S = S, d = d)
# Also you can use A, b, Aeq, beq for linear matrix (in)equality constraints
# and lb, ub for box-bound constraints lb <= x <= ub
# see /examples/lp_1.py 

#r = p.solve('cvxopt_sdp', iprint = 0)
r = p.solve('dsdp', iprint = -1)

f_opt, x_opt = r.ff, r.xf
print('x_opt: %s' % x_opt)
print('f_opt: %s' % f_opt)
#x_opt: [-0.36766609  1.89832827 -0.88755043]
#f_opt: -3.15354478797
