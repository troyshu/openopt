from openopt import NLP
from numpy import cos, arange, ones, asarray, abs, zeros, sqrt, sign, asscalar
from pylab import legend, show, plot, subplot, xlabel, subplots_adjust
from string import rjust, ljust, expandtabs, center, lower
from scipy import rand
N = 10
M = 5
s = 1.3
f = lambda x: (abs(x-M) ** s).sum()
df = lambda x: s * sign(x-M) * abs(x-M) ** (s-1)

x0 = cos(arange(N)) #+ rand(N)

c = lambda x: [2* x[0] **4-32, x[1]**2+x[2]**2 - 8]
def dc(x):
    r = zeros((len(c(x0)), p.n))
    r[0,0] = 2 * 4 * x[0]**3
    r[1,1] = 2 * x[1]
    r[1,2] = 2 * x[2]
    return r

K = 1e2
h1 = lambda x: K*(x[-1]-1)**4
h2 = lambda x: (x[-2]-1.5)**4
h3 = lambda x: (x[-5]+x[-6]-2*M+1.5)**6
h = lambda x: (h1(x), h2(x), h3(x))
def dh(x):
    r = zeros((3, N))
    r[0, -1] = 4 * K * (x[-1]-1)**3
    r[1, -2] = 4 * (x[-2]-1.5)**3
    r[2, -5] = 6 * (x[-5]+x[-6]-2*M+1.5)**5
    r[2, -6] = 6 * (x[-5]+x[-6]-2*M+1.5)**5
    return r


lb = -6*ones(N)
ub = 6*ones(N)
lb[3] = 5.5
ub[4] = 4.5
gtol=1e-1
ftol = 1e-6
xtol = 1e-6
diffInt = 1e-8
contol = 1e-6
maxTime = 10
maxIter = 13700
colors = ['b', 'k', 'y', 'g', 'r']

########################################
solvers = ['ralg', 'scipy_cobyla', 'lincher', 'scipy_slsqp']
#solvers = ['scipy_cobyla', 'ralg']
#solvers = ['lincher', 'scipy_cobyla']
solvers = [ 'ralg','scipy_cobyla']
#solvers = [ 'algencan','algencan2']
#solvers = [ 'ralg', 'ralg3']
########################################
colors = colors[:len(solvers)]
lines, results = [], {}
for j in range(len(solvers)):
    solver = solvers[j]
    color = colors[j]
    p = NLP(f, x0, name = 'bench2', df = df, c=c, dc = dc, h=h, dh = dh, lb = lb, ub = ub, gtol=gtol, ftol = ftol, maxFunEvals = 1e7, maxIter = maxIter, maxTime = maxTime,  plot = 1, color = color, iprint = 10, legend = [solvers[j]], show=False,  contol = contol)
#    p = NLP(f, x0, name = 'bench2', df = df, c=c, dc = dc, lb = lb, ub = ub, gtol=gtol, ftol = ftol, maxFunEvals = 1e7, maxIter = 1e4, maxTime = maxTime,  plot = 1, color = color, iprint = 0, legend = [solvers[j]], show=False,  contol = contol)
    if solver[:4] == ['ralg']:
        pass
#        p.gtol = 1e-8
#        p.ftol = 1e-7
#        p.xtol = 1e-7
    elif solver == 'lincher':
        #p.iprint = 1
        p.maxTime = 1e15
        p.maxIter = 100

##    p.check.df = 1
##    p.check.dc = 1
##    p.check.dh = 1
    r = p.solve(solver)
    for fn in ('h','c'):
        if not r.evals.has_key(fn): r.evals[fn]=0 # if no c or h are used in problem
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
    s=(ljust(lower(solvers[i]), 40-len(solvers[i]))+'%0.3f'% (results[solvers[i]][0]) + '       %0.1e' % (results[solvers[i]][1]) + ('      %0.2f'% (results[solvers[i]][2])) + '     %0.2f      '% (results[solvers[i]][3]) + str(results[solvers[i]][4]) + '   ' + rjust(str(results[solvers[i]][5]), 5) + ' '*5 +str(results[solvers[i]][6]))
    xl.append(s)

xl = '\n'.join(xl)
subplot(2,1,1)
xlabel('Time elapsed (without graphic output), sec')

from pylab import *
subplot(2,1,2)
xlabel(xl)
show()

