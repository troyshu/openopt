__docformat__ = "restructuredtext en"

from numpy import diag, ones, inf, any, copy, sqrt, vstack, concatenate, asarray, nan, where, array, zeros, exp, isfinite
from openopt.kernel.baseSolver import *
from openopt import LP, QP, NLP, LLSP, NSP

from openopt.kernel.ooMisc import WholeRepr2LinConst
#from scipy.optimize import line_search as scipy_optimize_linesearch
#from scipy.optimize.linesearch import line_search as scipy_optimize_linesearch_f
from numpy import arange, sign, hstack
from UkrOptMisc import getDirectionOptimPoint, getConstrDirection
import os

class lincher(baseSolver):
    __name__ = 'lincher'
    __license__ = "BSD"
    __authors__ = "Dmitrey"
    __alg__ = "a linearization-based solver written in Cherkassy town, Ukraine"
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']

    __isIterPointAlwaysFeasible__ = lambda self, p: p.__isNoMoreThanBoxBounded__()
    iterfcnConnected = True

    def __init__(self): pass

    def __solver__(self, p):
        n = p.n
        x0 = copy(p.x0)
        xPrev = x0.copy()
        xf = x0.copy()
        xk = x0.copy()
        p.xk = x0.copy()

        f0 = p.f(x0)

        fk = f0
        ff = f0
        p.fk = fk

        df0 = p.df(x0)

        #####################################################################


##        #handling box-bounded problems
##        if p.__isNoMoreThanBoxBounded__():
##            for k in range(int(p.maxIter)):
##
##        #end of handling box-bounded problems
        isBB = p.__isNoMoreThanBoxBounded__()
##        isBB = 0
        H = diag(ones(p.n))
        if not p.userProvided.c:
            p.c = lambda x : array([])
            p.dc = lambda x : array([]).reshape(0, p.n)
        if not p.userProvided.h:
            p.h = lambda x : array([])
            p.dh = lambda x : array([]).reshape(0, p.n)

        p.use_subproblem = 'QP'

        #p.use_subproblem = 'LLSP'

        for k in range(p.maxIter+4):
            if isBB:
                f0 = p.f(xk)
                df = p.df(xk)
                direction = -df
                f1 = p.f(xk+direction)
                ind_l = direction<=p.lb-xk
                direction[ind_l] = (p.lb-xk)[ind_l]
                ind_u = direction>=p.ub-xk
                direction[ind_u] = (p.ub-xk)[ind_u]
                ff = p.f(xk + direction)
##                print 'f0', f0, 'f1', f1, 'ff', ff
            else:

                mr = p.getMaxResidual(xk)
                if mr > p.contol: mr_grad = p.getMaxConstrGradient(xk)
                lb = p.lb - xk #- p.contol/2
                ub = p.ub - xk #+ p.contol/2
                c, dc, h, dh, df = p.c(xk), p.dc(xk), p.h(xk), p.dh(xk), p.df(xk)
                A, Aeq = vstack((dc, p.A)), vstack((dh, p.Aeq))
                b = concatenate((-c, p.b-p.matmult(p.A,xk))) #+ p.contol/2
                beq = concatenate((-h, p.beq-p.matmult(p.Aeq,xk)))

                if b.size != 0:
                    isFinite = isfinite(b)
                    ind = where(isFinite)[0]
                    A, b = A[ind], b[ind]
                if beq.size != 0:
                    isFinite = isfinite(beq)
                    ind = where(isFinite)[0]
                    Aeq, beq = Aeq[ind], beq[ind]


                if p.use_subproblem == 'LP': #linear
                    linprob = LP(df, A=A, Aeq=Aeq, b=b, beq=beq, lb=lb, ub=ub)
                    linprob.iprint = -1
                    r2 = linprob.solve('cvxopt_glpk') # TODO: replace lpSolve by autoselect
                    if r2.istop <= 0:
                        p.istop = -12
                        p.msg = "failed to solve LP subproblem"
                        return
                elif p.use_subproblem == 'QP': #quadratic
                    qp = QP(H=H,f=df, A=A, Aeq=Aeq, b=b, beq=beq, lb=lb, ub = ub)
                    qp.iprint = -1
                    r2 = qp.solve('cvxopt_qp') # TODO: replace solver by autoselect
                    #r2 = qp.solve('qld') # TODO: replace solver by autoselect
                    if r2.istop <= 0:
                        for i in range(4):
                            if p.debug: p.warn("iter " + str(k) + ": attempt Num " + str(i) + " to solve QP subproblem has failed")
                            #qp.f += 2*N*sum(qp.A,0)
                            A2 = vstack((A, Aeq, -Aeq))
                            b2 = concatenate((b, beq, -beq)) + pow(10,i)*p.contol
                            qp = QP(H=H,f=df, A=A2, b=b2, iprint = -5)
                            qp.lb = lb - pow(10,i)*p.contol
                            qp.ub = ub + pow(10,i)*p.contol
                            # I guess lb and ub don't matter here
                            try:
                                r2 = qp.solve('cvxopt_qp') # TODO: replace solver by autoselect
                            except:
                                r2.istop = -11
                            if r2.istop > 0: break
                        if r2.istop <= 0:
                            p.istop = -11
                            p.msg = "failed to solve QP subproblem"
                            return
                elif p.use_subproblem == 'LLSP':
                    direction_c = getConstrDirection(p,  xk, regularization = 1e-7)
                else: p.err('incorrect or unknown subproblem')


            if isBB:
                X0 = xk.copy()
                N = 0
                result, newX = chLineSearch(p, X0, direction, N, isBB)
            elif p.use_subproblem != 'LLSP':
                duals = r2.duals
                N = 1.05*abs(duals).sum()
                direction = r2.xf
                X0 = xk.copy()
                result, newX = chLineSearch(p, X0, direction, N, isBB)
            else: # case LLSP
                direction_f = -df
                p2 = NSP(LLSsubprobF, [0.8, 0.8], ftol=0, gtol=0, xtol = 1e-5, iprint = -1)
                p2.args.f =  (xk, direction_f, direction_c, p, 1e20)
                r_subprob = p2.solve('ralg')
                alpha = r_subprob.xf
                newX = xk + alpha[0]*direction_f + alpha[1]*direction_c

#                dw = (direction_f * direction_c).sum()
#                cos_phi = dw/p.norm(direction_f)/p.norm(direction_c)
#                res_0, res_1 = p.getMaxResidual(xk), p.getMaxResidual(xk+1e-1*direction_c)
#                print cos_phi, res_0-res_1

#                res_0 = p.getMaxResidual(xk)
#                optimConstrPoint = getDirectionOptimPoint(p, p.getMaxResidual, xk, direction_c)
#                res_1 = p.getMaxResidual(optimConstrPoint)
#
#                maxConstrLimit = p.contol



                #xk = getDirectionOptimPoint(p, p.f, optimConstrPoint, -optimConstrPoint+xk+direction_f, maxConstrLimit = maxConstrLimit)
                #print 'res_0', res_0, 'res_1', res_1, 'res_2', p.getMaxResidual(xk)
                #xk = getDirectionOptimPoint(p, p.f, xk, direction_f, maxConstrLimit)
                #newX = xk.copy()
                result = 0
#                x_0 = X0.copy()
#                N = j = 0
#                while p.getMaxResidual(x_0) > Residual0 + 0.1*p.contol:
#                    j += 1
#                    x_0 = xk + 0.75**j * (X0-xk)
#                X0 = x_0
#                result, newX = 0, X0
#                print 'newIterResidual = ', p.getMaxResidual(x_0)

            if result != 0:
                p.istop = result
                p.xf = newX
                return

            xk = newX.copy()
            fk = p.f(xk)

            p.xk, p.fk = copy(xk), copy(fk)
            #p._df = p.df(xk)
            ####################
            p.iterfcn()

            if p.istop:
                p.xf = xk
                p.ff = fk
                #p._df = g FIXME: implement me
                return


class lineSearchFunction(object):
    def __init__(self, p, x0, N):
        self.p = p
        self.x0 = x0
        self.N = N

    def __call__(self, x):
        return float(self.p.f(x)+self.N*max(self.p.getMaxResidual(x), 0.999*self.p.contol))

    def gradient_numerical(self, x):
        g = zeros(self.p.n)
        f0 = self.__call__(x)
        for i in range(self.p.n):
            x[i] += self.p.diffInt
            g[i] = self.__call__(x) - f0
            x[i] -= self.p.diffInt
        g /= self.p.diffInt
        return g

    def gradient(self, x):
        N = self.N
        g = self.p.df(x) + N * self.p.getMaxConstrGradient(x)
        return g


def LLSsubprobF(alpha, x, direction_f, direction_c, p, S=1e30):
    x2 = x + alpha[0] * direction_f + alpha[1] * direction_c
    constr = p.getMaxResidual(x2)
    fval = p.f(x2)
    return max(constr-p.contol, 0)*S + fval
#    if constr > p.contol: return S * constr
#    else: return p.f(x2)



def chLineSearch(p, x0, direction, N, isBB):
    lsF = lineSearchFunction(p, x0, N)
    c1, c2 = 1e-4, 0.9
    result = 0
    #ls_solver = 'scipy.optimize.line_search'
    #ls_solver = 'Matthieu.optimizers.StrongWolfePowellRule'
    #ls_solver = 'Matthieu.optimizers.BacktrackingSearch'
    ls_solver = 'Armijo_modified'
##    if p.use_subproblem == 'LLSP':
##        ls_solver = 'Armijo_modified3'
##    else:
##        ls_solver = 'Armijo_modified'


    #debug
##    M, K = 1000000, 4
##    x0 = array(1.)
##    class example():
##        def __init__(self): pass
##        #def __call__(self, x): return M * max(x,array(0.))**K
##        def __call__(self, x): return 1e-5*x
##        def gradient(self, x): return array(1e-5)
##        #def gradient(self, x): return M*K*max(x,array(0.))**(K-1)
##    ff = example()
##    state = {'direction' : array(-5.5), 'gradient': M*K*x0**(K-1)}
##    mylinesearch = line_search.StrongWolfePowellRule(sigma = 0.001)
##    destination = mylinesearch(function = ff, origin = x0, step = array(-5.5), state = state)

    #debug end



    if ls_solver == 'scipy.optimize.line_search':#TODO: old_fval, old_old_fval

        old_fval = p.dotmult(lsF.gradient(x0), direction).sum() # just to make
        old_old_fval = old_fval / 2.0 # alpha0 from scipy line_search 1

        results = scipy_optimize_linesearch(lsF, lsF.gradient, x0, direction, lsF.gradient(x0), old_fval, old_old_fval, c1=c1, c2=c2)
        alpha = results[0]
##        results_f = scipy_optimize_linesearch_f(lsF, lsF.gradient, x0, direction, lsF.gradient(x0), old_fval, old_old_fval, c1=c1, c2=c2)
##        alpha = results_f[0]
        destination = x0+alpha*direction
    elif ls_solver == 'Matthieu.optimizers.BacktrackingSearch':
        #state = {'direction' : direction}
        state = {'direction' : direction, 'gradient': lsF.gradient(x0)}
        mylinesearch = line_search.BacktrackingSearch()
        destination = mylinesearch(function = lsF, origin = x0, step = direction, state = state)

    elif ls_solver == 'Matthieu.optimizers.StrongWolfePowellRule':
        state = {'direction' : direction, 'gradient': lsF.gradient(x0)}
        mylinesearch = line_search.StrongWolfePowellRule()
        destination = mylinesearch(function = lsF, origin = x0, step = direction, state = state)

    elif ls_solver == 'Armijo_modified3':
        alpha,  alpha_min = 1.0, 0.45*p.xtol / p.norm(direction)
        lsF_x0 = lsF(x0)
        C1 = abs(c1 * (p.norm(direction)**2).sum())
        iterValues.r0 = p.getMaxResidual(x0)
        #counter = 1
        while 1:
            print 'stage 1'
            if lsF(x0 + direction*alpha) <= lsF_x0 - alpha * C1 and p.getMaxResidual(x0 + direction*alpha) <= max(p.contol, iterValues.r0):
                assert alpha>=0
                #print counter, C1
                break
            alpha /= 2.0
            #counter += 1
            if alpha < alpha_min:
                if p.debug: p.warn('alpha less alpha_min')
                break
        if alpha == 1.0:
            print 'stage 2'
            K = 1.5
            lsF_prev = lsF_x0
            for i in range(p.maxLineSearch):
                lsF_new = lsF(x0 + K * direction*alpha)
                newConstr = p.getMaxResidual(x0 + K * direction*alpha)
                if lsF_new > lsF_prev or  newConstr > max(p.contol, iterValues.r0):
                    break
                else:
                    alpha *= K
                    lsF_prev = lsF_new
        destination = x0 + direction*alpha

    elif ls_solver == 'Armijo_modified':
        alpha,  alpha_min = 1.0, 0.15*p.xtol / p.norm(direction)
        grad_x0 = lsF.gradient(x0)
        #C1 = abs(c1 * p.dotmult(direction, grad_x0).sum())
        #if p.debug: print p.dotmult(direction, grad_x0).sum(), p.norm(direction)**2
        C1 = abs(c1 * (p.norm(direction)**2).sum())
        lsF_x0 = lsF(x0)
        #counter = 1
        while 1:
##            print 'stage 11'
##            print 'alpha', alpha, 'lsF', lsF(x0 + direction*alpha), 'f', p.f(x0 + direction*alpha), 'maxC', p.getMaxResidual(x0 + direction*alpha)
            if lsF(x0 + direction*alpha) <= lsF_x0 - alpha * C1:
                assert alpha>=0

##                print '11 out: alpha = ', alpha
                break
            alpha /= 2.0
            if alpha < alpha_min:
                if p.debug: p.warn('alpha less alpha_min')
                break

        destination = x0 + direction*alpha
        #TODO: check lb-ub here?

        if alpha == 1.0 and not isBB:
            K = 1.5
            lsF_prev = lsF_x0
            for i in range(p.maxLineSearch):
                x_new = x0 + K * direction*alpha
##                ind_u, ind_l = x_new>p.ub, x_new<p.lb
##                x_new[ind_l] = p.lb[ind_l]
##                x_new[ind_u] = p.ub[ind_u]

                lsF_new = lsF(x_new)
##                print 'stage 22'
##                print  'alpha', K*alpha, 'lsF', lsF_new, 'f', p.f(x0 + K * direction*alpha), 'maxC', p.getMaxResidual(x0 + K * direction*alpha)
                if lsF_new >= lsF_prev:# - K * alpha * C1:
##                    print '22 out: alpha = ', alpha
                    break
                else:
                    destination = x_new
                    alpha *= K
                    lsF_prev = lsF_new


    elif ls_solver == 'Armijo_modified2':
        grad_objFun_x0 = p.df(x0)
        grad_iterValues.r_x0 = p.getMaxConstrGradient(x0)
        C1_objFun = c1 * p.dotmult(direction, grad_objFun_x0).sum()
        C1_constr = c1 * p.dotmult(direction, grad_iterValues.r_x0).sum()
        f0 = p.f(x0)
        f_prev = f0
        allowedConstr_start = max(0.999*p.contol, p.getMaxResidual(x0))
        #currConstr = allowedConstr_start + 1.0
        alpha,  alpha_min = 1.0, 1e-11
        isConstrAccepted = False
        isObjFunAccepted = False

        #debug
##        if p.iter == 100:
##            pass

        while alpha >= alpha_min:
            x_new = x0 + direction*alpha
            if not isConstrAccepted:
                currConstr = p.getMaxResidual(x_new)
                if currConstr > allowedConstr_start + alpha * C1_constr:
                    #print 'case bigger:', currConstr, allowedConstr
                    #allowedConstr = max(0.999*p.contol, min(allowedConstr, currConstr))
                    alpha /= 2.0; continue
                else:
                    AcceptedConstr = max(0.999*p.contol, currConstr)
                    isConstrAccepted = True

            if not isObjFunAccepted:
                currConstr = p.getMaxResidual(x_new)
                #if currConstr > allowedConstr_start + alpha * C1_constr:#min(AcceptedConstr, 0.1*allowedConstr_start):
                #AllowedConstr2 = 1.2 * AcceptedConstr
                if currConstr > p.contol and (currConstr > 1.3*AcceptedConstr or currConstr > allowedConstr_start + alpha * C1_constr):# or currConstr > AllowedConstr2):
                    isObjFunAccepted = True
                    alpha = min(1.0, 2.0*alpha)#i.e. return prev alpha value
                    break

                f_new = p.f(x_new)
                #if f_new > f_prev:
                if f_new > f0 + alpha * C1_objFun:
                    alpha /= 2.0
                    f_prev = f_new
                    continue
                else:
                    isObjFunAccepted = True # and continue
                    break

        #print '!!!!!!!!!', alpha
        #print currConstr, allowedConstr_start, allowedConstr_start + alpha * C1_constr
##            else:
##                #print 'allowedConstr:', allowedConstr, '  currConstr:', currConstr
##                allowedConstr = max(0.999*p.contol, min(allowedConstr, currConstr))
##                print '33'
##                alpha /= 2.0
##                continue
##            elif p.f(x_new) <= f0 - alpha * C1 and currConstr <= allowedConstr:
##               # accept the alpha value
##               assert alpha>=0
##               allowedConstr = max(0.99*p.contol, currConstr)
##               break


        if p.debug and alpha < alpha_min:
            p.warn('alpha less alpha_min')

        if alpha == 1.0:
            K = 1.5
            f_prev = f0
            allowedConstr = allowedConstr_start
            for i in range(p.maxLineSearch):
                x_new = x0 + K*direction*alpha
                f_new = p.f(x_new)
                if f_new > f_prev or p.getMaxResidual(x_new) > allowedConstr:# - K * alpha * C1:
                    break
                else:
                    allowedConstr = max(0.99*p.contol, min(allowedConstr, currConstr))
                    alpha *= K
                    f_new = f_prev
        destination = x0 + direction*alpha
        #print 'alpha:', alpha

    else:
        p.error('unknown line-search optimizer')

    return result, destination
