from runProbSolver import runProbSolver
def ooRun(prob, solvers, *args, **kwargs):
    r = []
    for solver in solvers:
        r.append(runProbSolver(prob.copy(), solver, *args, **kwargs))
    return r
