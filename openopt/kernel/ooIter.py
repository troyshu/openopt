__docformat__ = "restructuredtext en"

from time import time, clock
from numpy import isscalar,  array_equal

######################
# don't cjhange to mere ooMisc! 
from openopt.kernel.ooMisc import isSolved 
######################

from setDefaultIterFuncs import USER_DEMAND_STOP, IS_NAN_IN_X, SMALL_DELTA_X, IS_MAX_ITER_REACHED, \
IS_MAX_CPU_TIME_REACHED, IS_MAX_TIME_REACHED, IS_MAX_FUN_EVALS_REACHED

has_Tkinter = True
try:
    import Tkinter
except ImportError:
    has_Tkinter = False

NoneType = type(None)

def ooIter(p, *args,  **kwargs):
    """
    this func is called from iter to iter
    it is default iter function of OpenOpt kernel
    lots of solvers use this one
    it provides basic graphics output (provided plot option is turned on),
    maybe in future some text output will also be generated here.
    also, some stop criteria are handled via the func.
    """
    
    if p.finalIterFcnFinished: return
    
    if has_Tkinter:
        if p.state == 'paused':
            p.GUI_root.wait_variable(p.statusTextVariable)

    if not hasattr(p, 'timeStart'): return#called from check 1st derivatives
    
    p.currtime = time()
    if not p.iter:
        p.lastDrawTime = p.currtime
        p.lastDrawIter = 0

    if not p.isFinished or len(p.iterValues.f) == 0:
        p.solver.__decodeIterFcnArgs__(p,  *args,  **kwargs)
        condEqualLastPoints = hasattr(p, 'xk_prev') and array_equal(p.xk,  p.xk_prev) 
        p.xk_prev = p.xk.copy()
        if p.graphics.xlabel == 'nf': p.iterValues.nf.append(p.nEvals['f'])
        p.iterCPUTime.append(clock() - p.cpuTimeStart)
        p.iterTime.append(p.currtime - p.timeStart)

        # TODO: rework it
        if p.probType not in ('GLP', 'MILP') and p.solver.__name__ not in ('de', 'pswarm', 'interalg') \
        and ((p.iter == 1 and array_equal(p.xk,  p.x0)) or condEqualLastPoints):
            elems = [getattr(p.iterValues,  fn) for fn in dir(p.iterValues)] + [p.iterTime, p.iterCPUTime]#dir(p.iterValues)
            for elem in elems:
                if type(elem) == list:
                    elem.pop(-1)

            #TODO: handle case x0 = x1 = x2 = ...
            if not (p.isFinished and condEqualLastPoints): 
                return

        #TODO: turn off xtol and ftol for artifically iterfcn funcs

        if not p.userStop and (not condEqualLastPoints or p.probType == 'GLP' or p.solver.__name__ in ('de', 'pswarm', 'interalg')):
            for key, fun in p.kernelIterFuncs.items():
                r =  fun(p)
                if r is not False:
                    p.stopdict[key] = True
                    if p.istop == 0 or key not in [IS_MAX_ITER_REACHED, IS_MAX_CPU_TIME_REACHED, IS_MAX_TIME_REACHED, IS_MAX_FUN_EVALS_REACHED]:
                        p.istop = key
                        if type(r) == tuple:
                            p.msg = r[1]
                        else:
                            p.msg = 'unkown, if you see the message inform openopt developers'
            if IS_NAN_IN_X in p.stopdict.keys():pass
            elif SMALL_DELTA_X in p.stopdict.keys() and array_equal(p.iterValues.x[-1], p.iterValues.x[-2]): pass
            else:
                p.nonStopMsg = ''
                for fun in p.denyingStopFuncs.keys():
                    if not fun(p):
                        p.istop = 0
                        p.stopdict = {}
                        p.msg = ''
                        p.nonStopMsg = p.denyingStopFuncs[fun]
                        break
                for fun in p.callback:
                    r =  fun(p)
                    if r is None: p.err('user-defined callback function returned None, that is forbidden, see /doc/userCallback.py for allowed return values')
                    if r not in [0,  False]:
                        if r in [True,  1]:  p.istop = USER_DEMAND_STOP
                        elif isscalar(r):
                            p.istop = r
                            p.msg = 'user-defined'
                        else:
                            p.istop = r[0]
                            p.msg = r[1]
                        p.stopdict[p.istop] = True
                        p.userStop = True
    if not p.solver.properTextOutput: 
        p.iterPrint()
    T, cpuT = 0., 0.

    if p.plot and (p.iter == 0 or p.iter <2 or p.isFinished or \
    p.currtime - p.lastDrawTime > p.graphics.rate * (p.currtime - p.iterTime[p.lastDrawIter] - p.timeStart)):
        for df in p.graphics.drawFuncs: df(p)
        T = time() - p.timeStart - p.iterTime[-1]
        cpuT = clock() - p.cpuTimeStart - p.iterCPUTime[-1]
        p.lastDrawTime = time()
        p.lastDrawIter = p.iter
    if p.plot:
        p.timeElapsedForPlotting.append(T+p.timeElapsedForPlotting[-1])
        p.cpuTimeElapsedForPlotting.append(cpuT+p.cpuTimeElapsedForPlotting[-1])
    
    p.iter += 1

    if p.isFinished: p.finalIterFcnFinished = True
    if p.istop and p.istop != 1000 and not p.solver.iterfcnConnected and not p.isFinished and p.solver.useStopByException:
        p.debugmsg('exit solver via exception; istop=%d' % p.istop)
        raise isSolved


