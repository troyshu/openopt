from openopt import NLP
from numpy import cos, arange, ones, asarray, abs, zeros, sqrt, asscalar
from pylab import legend, show, plot, subplot, xlabel, subplots_adjust
from string import rjust, ljust, expandtabs
N = 15
M = 5
f = lambda x: -(abs(x-M) ** 1.5).sum()

x0 = cos(arange(N))

#c = lambda x: [2* x[0] **4-32, x[1]**2+x[2]**2 - 8]
global cc1, cc2, cc3

def c1(x):
    global cc1
    cc1 += 1
    return 2* x[0] **4-32



def c2(x):
    global cc2
    cc2 += 1
    return x[1]**2+x[2]**2 - 8


def c3(x):
    global cc3
    cc3 += 1
    return x[1]**2+x[2]**2 + x[3]**2  - 35


c = [c1, c2, c3]

h1 = lambda x: 1e1*(x[-1]-1)**4
h2 = lambda x: (x[-2]-1.5)**4
h = (h1, h2)

lb = -6*ones(N)
ub = 6*ones(N)
lb[3] = 5.5
ub[4] = 4.5


colors = ['b', 'k', 'y', 'r', 'g']
#############
#solvers = ['ralg','scipy_cobyla', 'lincher']
solvers = ['ralg', 'scipy_cobyla', 'lincher','ipopt','algencan' ]
solvers = ['ralg', 'ralg3', 'ralg5']
solvers = ['ralg',  'scipy_cobyla']
#solvers = ['ipopt']
solvers = ['ralg', 'ipopt']
solvers = ['ralg']
solvers = ['scipy_slsqp']
solvers = ['ralg']
#############
colors = colors[:len(solvers)]

lines, results = [], {}
for j in range(len(solvers)):
    cc1,  cc2,  cc3 = 0, 0, 0
    solver = solvers[j]
    color = colors[j]
    p = NLP(f, x0, c=c, h=h, lb = lb, ub = ub, ftol = 1e-6, maxFunEvals = 1e7, maxIter = 1220, plot = 1, color = color, iprint = 0, legend = [solvers[j]], show= False, xlabel='time', goal='maximum', name='nlp3')
    if solver == 'algencan':
        p.gtol = 1e-1
    elif solver == 'ralg':
        p.debug = 1

    r = p.solve(solver, debug=1)
    print 'c1 evals:', cc1, 'c2 evals:', cc2, 'c3 evals:', cc3
    results[solver] = (r.ff, p.getMaxResidual(r.xf), r.elapsed['solver_time'], r.elapsed['solver_cputime'], r.evals['f'], r.evals['c'], r.evals['h'])
    subplot(2,1,1)
    F0 = asscalar(p.f(p.x0))
    lines.append(plot([0, 1e-15], [F0, F0], color= colors[j]))

for i in range(2):
    subplot(2,1,i+1)
    legend(lines, solvers)

subplots_adjust(bottom=0.2, hspace=0.3)

xl = ['Solver                              f_opt     MaxConstr   Time   CPUTime  fEvals  cEvals  hEvals']
for i in range(len(results)):
    xl.append((expandtabs(ljust(solvers[i], 16)+' \t', 15)+'%0.2f'% (results[solvers[i]][0]) + '        %0.1e' % (results[solvers[i]][1]) + ('      %0.2f'% (results[solvers[i]][2])) + '    %0.2f      '% (results[solvers[i]][3]) + str(results[solvers[i]][4]) + '   ' + rjust(str(results[solvers[i]][5]), 5) + expandtabs('\t' +str(results[solvers[i]][6]),8)))

xl = '\n'.join(xl)
subplot(2,1,1)
xlabel('Time elapsed (without graphic output), sec')

from pylab import *
subplot(2,1,2)
xlabel(xl)

show()

