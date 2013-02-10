from numpy import hstack,  asarray, abs, atleast_1d, where, \
logical_not, argsort, vstack, sum, array, nan, all

from FuncDesigner import oopoint

def interalg_ODE_routine(p, solver):
    isIP = p.probType == 'IP'
    isODE = p.probType == 'ODE'
    if isODE:
        f, y0, t, r30, ftol = p.equations, p.x0, p.timeVariable, p.times, p.ftol
        assert len(f) == 1, 'multiple ODE equations are unimplemented for FuncDesigner yet'
        f = list(f.values())[0]
    elif isIP:
        assert p.n == 1 and p.__isNoMoreThanBoxBounded__()
        f, y0, ftol = p.user.f[0], 0.0, p.ftol
        if p.fTol is not None: ftol = p.fTol
        t = list(f._getDep())[0]
        r30 = p.domain[t]
        p.iterfcn(p.point([nan]*p.n))
    else:
        p.err('incorrect prob type for interalg ODE routine') 
    
    eq_var = list(p._x0.keys())[0]

    dataType = solver.dataType
    if type(ftol) == int: 
        ftol = float(ftol) # e.g. someone set ftol = 1
    # Currently ftol is scalar, in future it can be array of same length as timeArray
    if len(r30) < 2:
        p.err('length ot time array must be at least 2')    
#    if any(r30[1:] < r30[:-1]):
#        p.err('currently interalg can handle only time arrays sorted is ascending order')  
#    if any(r30 < 0):
#        p.err('currently interalg can handle only time arrays with positive values')  
#    if p.times[0] != 0:
#        p.err('currently solver interalg requires times start from zero')  
    
    r37 = abs(r30[-1] - r30[0])
    r28 = asarray(atleast_1d(r30[0]), dataType)
    r29 = asarray(atleast_1d(r30[-1]), dataType)
    storedr28 = []
    r27 = []
    r31 = []
    r32 = []
    r33 = ftol
    F = 0.0
    p._Residual = 0
    
    # Main cycle
    for itn in range(p.maxIter+1):
        if r30[-1] > r30[0]:
            mp = oopoint({t: [r28, r29]}, skipArrayCast = True)
        else:
            mp = oopoint({t: [r29, r28]}, skipArrayCast = True)
        mp.isMultiPoint = True
        delta_y = f.interval(mp, dataType)
        if not all(delta_y.definiteRange):
            p.err('''
            solving ODE with interalg is implemented for definite (real) range only, 
            no NaN values in integrand are allowed''')
        # TODO: perform check on NaNs
        r34 = atleast_1d(delta_y.ub)
        r35 = atleast_1d(delta_y.lb)
        r36 = atleast_1d(r34 - r35   <= 0.95 * r33 / r37)
        ind = where(r36)[0]
        if isODE:
            storedr28.append(r28[ind])
            r27.append(r29[ind])
            r31.append(r34[ind])
            r32.append(r35[ind])
        else:
            assert isIP
            F += 0.5 * sum((r29[ind]-r28[ind])*(r34[ind]+r35[ind]))
            
            #p._Residual = p._residual + 0.5*sum((abs(r34) +abs(r35)) * (r29 - r28))
        
        if ind.size != 0: 
            tmp = abs(r29[ind] - r28[ind])
            Tmp = sum((r34[ind] - r35[ind]) * tmp)
            r33 -= Tmp
            if isIP: p._residual += Tmp
            r37 -= sum(tmp)
        ind = where(logical_not(r36))[0]
        if ind.size == 0:
            p.istop = 1000
            p.msg = 'problem has been solved according to required tolerance'
            break
            
        # OLD
#        for i in ind:
#            t0, t1 = r28[i], r29[i]
#            t_m = 0.5 * (t0+t1)
#            newr28.append(t0)
#            newr28.append(t_m)
#            newr29.append(t_m)
#            newr29.append(t1)
        # NEW
        r38, r39 = r28[ind], r29[ind]
        r40 = 0.5 * (r38 + r39)
        r28 = vstack((r38, r40)).flatten()
        r29 = vstack((r40, r39)).flatten()
        
        # !!! unestablished !!!
        if isODE:
            p.iterfcn(fk = r33/ftol)
        elif isIP:
            p.iterfcn(xk=array(nan), fk=F, rk = 0)
        else:
            p.err('bug in interalgODE.py')
            
        if p.istop != 0 : 
            break
        
        #print(itn, r28.size)

    if isODE:
        
        t0, t1, lb, ub = hstack(storedr28), hstack(r27), hstack(r32), hstack(r31)
        ind = argsort(t0)
        if r30[0] > r30[-1]:
            ind = ind[::-1] # reverse
        t0, t1, lb, ub = t0[ind], t1[ind], lb[ind], ub[ind]
        lb, ub = y0+(lb*(t1-t0)).cumsum(), y0+(ub*(t1-t0)).cumsum()
        #y_var = p._x0.keys()[0]
        #p.xf = p.xk = 0.5*(lb+ub)
        p.extras = {'startTimes': t0, 'endTimes': t1, eq_var:{'infinums': lb, 'supremums': ub}}
        return t0, t1, lb, ub
    elif isIP:
        P = p.point([nan]*p.n)
        P._f = F
        P._mr = r33
        P._mrName = 'None'
        P._mrInd = 0
#        p.xk = array([nan]*p.n)
#        p.rk = r33
#        p.fk = F
        #p._Residual = 
        p.iterfcn(asarray([nan]*p.n), fk=F, rk=0)
    else:
        p.err('incorrect prob type in interalg ODE routine')

#        for i in range(len(T)-1): # TODO: is or is not T0 = 0?
#            delta_y = f.interval({t: [T[i], T[i+1]]})
#            delta_y_supremum[i] = delta_y.ub
#            delta_y_infinum[i] = delta_y.lb
#        diffTime = diff(T)
#        y_supremums, y_infinums = cumsum(delta_y_supremum*diffTime), cumsum(delta_y_infinum*diffTime)
#        
#        #!!!! Currently for all time points ub-lb <= ftol is required, not only for those from timeArray
#        
#        if y_supremums[-1] - y_infinums[-1] < ftol:
#            # hence prob is solved
#            if p.debug: 
#                assert all(y_supremums - y_infinums < ftol)
#            break
#
#        d = (y_supremums - y_infinums) / ftol
#        minNewPoints = int(2*(y_supremums[-1] - y_infinums[-1])/ftol) + 1000
#        #print(itn, minNewPoints)
#        tmp = diff(d)
#        tmp /= max(tmp)
#        ff = lambda x: abs(floor(x*tmp).sum() - minNewPoints)
##        def ff(x):
##            #print x
##            return abs(floor(x*tmp).sum() - minNewPoints)
#        P = NLP(ff, 0, lb = 1, ub = minNewPoints+1, iprint = -1)
#        R = P.solve('goldenSection', ftol = max((2, 0.1*minNewPoints)), xtol = 0.0)
#        if p.debug: assert R.stopCase == 1
#        tmp = floor(R.xf * tmp)
#        D = asarray(hstack((0, tmp)) , int)
#        
#        newT = []
#        for i in range(len(T)-1):
#            m = 1 + D[i]
#            newT.append(linspace(T[i], T[i+1], m, False))
#        newT.append(T[-1])
#        #print('new T len - prev T len = ',  len(hstack(newT)) - len(T))
#        T = hstack(newT)
##        ind = where(d > ftol)
    
