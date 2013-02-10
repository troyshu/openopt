"""
The example illustrates oosolver usage
You should pay special attention for "isInstalled" field

oosolver work is untested for converters
"""
from openopt import oosolver, NLP

ipopt = oosolver('ipopt', color='r') # oosolver can hanlde prob parameters
ralg = oosolver('ralg', color='k', alp = 4.0) # as well as solver parameters
asdf = oosolver('asdf')

solvers = [ralg, asdf, ipopt]
# or just
# solvers = [oosolver('ipopt', color='r'), oosolver('asdf'), oosolver('ralg', color='k', alp = 4.0)]

for solver in solvers:
    if not solver.isInstalled:
        print 'solver ' + solver.__name__ + ' is not installed'
        continue
    p = NLP(x0 = 15, f = lambda x: x**4, df = lambda x: 4 * x**3, iprint = 0)
    r = p.solve(solver, plot=1, show = solver == solvers[-1])
