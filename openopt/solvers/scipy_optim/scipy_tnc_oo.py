from scipy.optimize.tnc import fmin_tnc
import scipy.optimize.tnc as tnc
import openopt
from openopt.kernel.setDefaultIterFuncs import *
from openopt.kernel.ooMisc import WholeRepr2LinConst
from openopt.kernel.baseSolver import baseSolver

class scipy_tnc(baseSolver):
    __name__ = 'scipy_tnc'
    __license__ = "BSD"
    __authors__ = "Stephen G. Nash"
    __alg__ = "undefined"
    __info__ = 'box-bounded NLP solver, can handle lb<=x<=ub constraints, some lb-ub coords can be +/- inf'
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
    __isIterPointAlwaysFeasible__ = lambda self, p: True

    def __init__(self): pass

    def __solver__(self, p):
        #WholeRepr2LinConst(p)#TODO: remove me
        bounds = []
        for i in range(p.n): bounds.append((p.lb[i], p.ub[i]))
        messages = 0#TODO: edit me

        maxfun=p.maxFunEvals
        if maxfun > 1e8:
            p.warn('tnc cannot handle maxFunEvals > 1e8, the value will be used')
            maxfun = int(1e8)

        xf, nfeval, rc = fmin_tnc(p.f, x0 = p.x0, fprime=p.df, args=(), approx_grad=0, bounds=bounds, messages=messages, maxfun=maxfun, ftol=p.ftol, xtol=p.xtol, pgtol=p.gtol)

        if rc in (tnc.INFEASIBLE, tnc.NOPROGRESS): istop = FAILED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON
        elif rc == tnc.FCONVERGED: istop = SMALL_DELTA_F
        elif rc == tnc.XCONVERGED: istop = SMALL_DELTA_X
        elif rc == tnc.MAXFUN: istop = IS_MAX_FUN_EVALS_REACHED
        elif rc == tnc.LSFAIL: istop = IS_LINE_SEARCH_FAILED
        elif rc == tnc.CONSTANT: istop = IS_ALL_VARS_FIXED
        elif rc == tnc.LOCALMINIMUM: istop = SOLVED_WITH_UNIMPLEMENTED_OR_UNKNOWN_REASON
        else:
            #TODO: IMPLEMENT USERABORT
            p.err('unknown stop reason')
        msg = tnc.RCSTRINGS[rc]
        p.istop, p.msg = istop, msg
        p.xf = xf

##        INFEASIBLE   = -1 # Infeasible (low > up)
##        LOCALMINIMUM =  0 # Local minima reach (|pg| ~= 0)
##        FCONVERGED   =  1 # Converged (|f_n-f_(n-1)| ~= 0)
##        XCONVERGED   =  2 # Converged (|x_n-x_(n-1)| ~= 0)
##        MAXFUN       =  3 # Max. number of function evaluations reach
##        LSFAIL       =  4 # Linear search failed
##        CONSTANT     =  5 # All lower bounds are equal to the upper bounds
##        NOPROGRESS   =  6 # Unable to progress
##        USERABORT    =  7 # User requested end of minimization

##RCSTRINGS = {
##        INFEASIBLE   : "Infeasible (low > up)",
##        LOCALMINIMUM : "Local minima reach (|pg| ~= 0)",
##        FCONVERGED   : "Converged (|f_n-f_(n-1)| ~= 0)",
##        XCONVERGED   : "Converged (|x_n-x_(n-1)| ~= 0)",
##        MAXFUN       : "Max. number of function evaluations reach",
##        LSFAIL       : "Linear search failed",
##        CONSTANT     : "All lower bounds are equal to the upper bounds",
##        NOPROGRESS   : "Unable to progress",
##        USERABORT    : "User requested end of minimization"
##}







