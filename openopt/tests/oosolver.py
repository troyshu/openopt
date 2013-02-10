from openopt import oosolver, NLP
ipopt = oosolver('ipopt', color='r') # oosolver can hanlde prob parameters
ralg = oosolver('ralg', color='k', alp = 4.0) # as well as solver parameters
solvers = [ralg, ipopt]
for solver in solvers:
    assert solver.isInstalled, 'solver ' + solver.__name__ + ' is not installed'
    p = NLP(x0 = 15, f = lambda x: x**4, df = lambda x: 4 * x**3, iprint = 0)
    r = p.solve(solver, plot=1, show = solver == solvers[-1])
