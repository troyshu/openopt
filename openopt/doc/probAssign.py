# Problem assignment in OpenOpt is performed in the following way:
from openopt import NLP 
# or other constructor names: LP, MILP, QP etc, 
# for full list see http://openopt.org/Problems
# p = NLP(*args, **kwargs)

"""
you should read help(NLP) for more details, 
also reading /examples/nlp_1.py and other files from the directory is highly recommended

Each class has some expected arguments
e.g. for NLP it's f and x0 - objective function and start point
thus using NLP(myFunc, myStartPoint) will assign myFunc to f and myStartPoint to x0 prob fields

alternatively, you could use it as kwargs, possibly along with some other kwargs:
"""

p = NLP(x0=15, f = lambda x: x**2-0.4, df = lambda x: 2*x, iprint = 0, plot = 1)

# after the problem is assigned, you could turn the parameters, 
# along with some other that have been set as defaults:

p.x0 = 0.15
p.plot = 0

def f(x):
    return x if x>0 else x**2
p.f = f

# At last, you can modify any prob parameters in minimize/maximize/solve/manage functions:

r = p.minimize('ralg', x0 = -1.5,  iprint = -1, plot = 1, color = 'r') 
# or
#r = p.manage('ralg', start = False, iprint = 0, x0 = -1.5)

"""
Note that *any* kwarg passed to constructor will be assigned
e.g. 
p = NLP(f, x0, myName='JohnSmith')
is equivalent to 
p.myName='JohnSmith'
It can be very convenient for user-supplied callback functions 
(see /examples/userCallback.py)
(instead of using "global" as you have to do in MATLAB)

See also http://openopt.org/OOFrameworkDoc#Result_structure for result structure (r) fields 
"""
