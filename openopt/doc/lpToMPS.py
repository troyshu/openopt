# Example of export OpenOpt LP to MPS file
# you should have lpsolve and its Python binding properly installed
# (you may take a look at the instructions from openopt.org/LP)

# You can solve problems defined in MPS files 
# with a variety of solvers at NEOS server for free
# http://neos.mcs.anl.gov/
# BTW they have Python API along with web API and other

from numpy import *
from openopt import LP
f = array([15,8,80])
A = mat('1 2 3; 8 15 80; 8 80 15; -100 -10 -1') # numpy.ndarray is also allowed
b = [15, 80, 150, -800] # numpy.ndarray, matrix etc are also allowed
Aeq = mat('80 8 15; 1 10 100') # numpy.ndarray is also allowed
beq = (750, 80)

lb = [4, -80, -inf]
ub = [inf, -8, inf]
p = LP(f, A=A, Aeq=Aeq, b=b, beq=beq, lb=lb, ub=ub, name = 'lp_1')
# or p = LP(f=f, A=A, Aeq=Aeq, b=b, beq=beq, lb=lb, ub=ub)

# if file name not ends with '.MPS' or '.mps'
# then '.mps' will be appended
success = p.exportToMPS('asdf') 
# success is False if a error occurred (read-only file system, no write access, etc)
# elseware success is True

# objFunValue should be 204.48841578
# x_opt should be [ 9.89355041 -8.          1.5010645 ]
