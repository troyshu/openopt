"""
Concider the MIQCQP problem
0.5 * (x1^2 + 2x2^2 + 3x3^2) + 15x1 + 8x2 + 80x3 -> min        (1)
subjected to
x1 + 2x2 + 3x3 <= 150            (2)
8x1 +  15x2 +  80x3 <= 800    (3)
x2 - x3 = 25.5                           (4)
x1 <= 15                                  (5)
x1^2 + 2.5 x2^2 + 3 x3^2 + 0.1 x1 + 0.2 x2 + 0.3 x3 - 1000 <= 0 (6)
2 x1^2 + x2^2 + 3 x3^2 + 0.1 x1 + 0.5 x2 + 0.3 x3  <= 1000 (7)
x1, x3 are integers
"""

from numpy import diag, matrix, inf
from openopt import QP

H = diag([1.0, 2.0,3.0])
f = [15,8,80]
A = matrix('1 2 3; 8 15 80')
b = [150, 800]

# QC should be list or tuple of triples (P, q, s): 0.5 x^T P x + q x + s <= 0
QC = ((diag([1.0, 2.5, 3.0]), [0.1, 0.2, 0.3], -1000), (diag([2.0, 1.0, 3.0]), [0.1, 0.5, 0.3], -1000))

p = QP(H, f, A = A, b = b, Aeq = [0, 1, -1], beq = 25.5, ub = [15,inf,inf], QC = QC, name='OpenOpt QCQP example 1')
# or p = QP(H=diag([1,2,3]), f=[15,8,80], ...)

r = p.solve('cplex', iprint = 0, plot=1)
f_opt, x_opt = r.ff, r.xf
# x_opt = array([ -2.99999999,   9.5       , -16.        ])
# f_opt = -770.24999989134858
