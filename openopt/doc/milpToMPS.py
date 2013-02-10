# Example of export OpenOpt MILP to MPS file
# you should have lpsolve and its Python binding properly installed
# (you may take a look at the instructions from openopt.org/LP)

# You can solve problems defined in MPS files 
# with a variety of solvers at NEOS server for free
# http://neos.mcs.anl.gov/
# BTW they have Python API along with web API and other

from numpy import *
from openopt import MILP

f = [1, 2, 3, 4, 5, 4, 2, 1]

# indexing starts from ZERO!
# while in native lpsolve-python wrapper from 1
# so if you used [5,8] for native lp_solve python binding
# you should use [4,7] instead
intVars = [4, 7]

lb = -1.5 * ones(8)
ub = 15 * ones(8)
A = zeros((5, 8))
b = zeros(5)
for i in xrange(5):
    for j in xrange(8):
        A[i,j] = -8+sin(8*i) + cos(15*j)
    b[i] = -150 + 80*sin(80*i)

p = MILP(f=f, lb=lb, ub=ub, A=A, b=b, intVars=intVars)

# if file name not ends with '.MPS' or '.mps'
# then '.mps' will be appended
success = p.exportToMPS('/home/dmitrey/PyTest/milp_1')
# or write into current dir: 
# success = p.exportToMPS('milp')
# success is False if a error occurred (read-only file system, no write access, etc)
# elseware success is True

# f_opt is 25.801450769161505
# x_opt is [ 15. 10.15072538 -1.5 -1.5 -1.  -1.5 -1.5 15.]
