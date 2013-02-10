from numpy import isnan, take, any, all, logical_or, logical_and, logical_not, atleast_1d, where, \
asarray, argmin, argsort, isfinite
import numpy as np
from bisect import bisect_right
from FuncDesigner.Interval import adjust_lx_WithDiscreteDomain, adjust_ux_WithDiscreteDomain
try:
    from bottleneck import nanmin
except ImportError:
    from numpy import nanmin


def adjustDiscreteVarBounds(y, e, p):
    #n = p.n
    # TODO: remove the cycle, use vectorization
    for i in p._discreteVarsNumList:
        v = p._freeVarsList[i]
        #y += 100
        adjust_lx_WithDiscreteDomain(y[:, i], v)
        adjust_ux_WithDiscreteDomain(e[:, i], v)

    ind = any(y>e, 1)

    # TODO:  is it triggered?
    if any(ind):
        #print('asdf')
        ind = where(logical_not(ind))[0]
        s = ind.size
        y = take(y, ind, axis=0, out=y[:s])
        e = take(e, ind, axis=0, out=e[:s])
    return y, e
    
    
def func7(y, e, o, a, _s, indT, nlhc, residual):
    r10 = logical_and(all(isnan(o), 1), all(isnan(a), 1))
    if any(r10):
        j = where(logical_not(r10))[0]
        lj = j.size
        y = take(y, j, axis=0, out=y[:lj])
        e = take(e, j, axis=0, out=e[:lj])
        o = take(o, j, axis=0, out=o[:lj])
        a = take(a, j, axis=0, out=a[:lj])
        _s = _s[j]
        if indT is not None:
            indT = indT[j]
        if nlhc is not None:
            nlhc = take(nlhc, j, axis=0, out=nlhc[:lj])
        if residual is not None:
            residual = take(residual, j, axis=0, out=residual[:lj])
    return y, e, o, a, _s, indT, nlhc, residual
    

def func9(an, fo, g, p):
    
    #ind = searchsorted(ar, fo, side='right')
    if p.probType in ('NLSP', 'SNLE') and p.maxSolutions != 1:
        mino = atleast_1d([node.key for node in an])
        ind = mino > 0
        if not any(ind):
            return an, g
        else:
            g = nanmin((g, nanmin(mino[ind])))
            ind2 = where(logical_not(ind))[0]
            #an = take(an, ind2, axis=0, out=an[:ind2.size])
            an = asarray(an[ind2])
            return an, g
            
        
    elif p.solver.dataHandling == 'sorted':
        #OLD
        mino = [node.key for node in an]
        ind = bisect_right(mino, fo)
        if ind == len(mino):
            return an, g
        else:
            g = nanmin((g, nanmin(atleast_1d(mino[ind]))))
            return an[:ind], g
    elif p.solver.dataHandling == 'raw':
        
        #NEW
        mino = [node.key for node in an]
        mino = atleast_1d(mino)
        r10 = mino > fo
        if not any(r10):
            return an, g
        else:
            ind = where(r10)[0]
            g = nanmin((g, nanmin(atleast_1d(mino)[ind])))
            an = asarray(an)
            ind2 = where(logical_not(r10))[0]
            #an = take(an, ind2, axis=0, out=an[:ind2.size])
            an = asarray(an[ind2])
            return an, g

        # NEW 2
#        curr_tnlh = [node.tnlh_curr for node in an]
#        import warnings
#        warnings.warn('! fix g')
        
        return an, g
        
    else:
        assert 0, 'incorrect nodes remove approach'

def func5(an, nn, g, p):
    m = len(an)
    if m <= nn: return an, g
    
    mino = [node.key for node in an]
    
    if nn == 1: # box-bound probs with exact interval analysis
        ind = argmin(mino)
        assert ind in (0, 1), 'error in interalg engine'
        g = nanmin((mino[1-ind], g))
        an = atleast_1d([an[ind]])
    elif m > nn:
        if p.solver.dataHandling == 'raw':
            ind = argsort(mino)
            th = mino[ind[nn]]
            ind2 = where(mino < th)[0]
            g = nanmin((th, g))
            #an = take(an, ind2, axis=0, out=an[:ind2.size])
            an = an[ind2]
        else:
            g = nanmin((mino[nn], g))
            an = an[:nn]
    return an, g

def func4(p, y, e, o, a, fo, tnlhf_curr = None):
    if fo is None and tnlhf_curr is None: return False# used in IP
    if y.size == 0: return False
    cs = (y + e)/2
    n = y.shape[1]
    
    
    if tnlhf_curr is not None:
        tnlh_modL = tnlhf_curr[:, 0:n]
        ind = logical_not(isfinite(tnlh_modL))
    else:
        s = o[:, 0:n]
        ind = logical_or(s > fo, isnan(s)) # TODO: assert isnan(s) is same to isnan(a_modL)
        
    indT = any(ind, 1)
    if any(ind):
        y[ind] = cs[ind]
        # Changes
#        ind = logical_and(ind, logical_not(isnan(a[:, n:2*n])))
##        ii = len(where(ind)[0])
##        if ii != 0: print ii
        if p.probType != 'MOP':
            a[:, 0:n][ind] = a[:, n:2*n][ind]
            o[:, 0:n][ind] = o[:, n:2*n][ind]
        if tnlhf_curr is not None:
            tnlhf_curr[:, 0:n][ind] = tnlhf_curr[:, n:2*n][ind]
#        for arr in arrays:
#            if arr is not None:
#                arr[:, 0:n][ind] = arr[:, n:2*n][ind]

    if tnlhf_curr is not None:
        tnlh_modU = tnlhf_curr[:, n:2*n]
        ind = logical_not(isfinite(tnlh_modU))
    else:
        q = o[:, n:2*n]
        ind = logical_or(q > fo, isnan(q)) # TODO: assert isnan(q) is same to isnan(a_modU)
        
    indT = logical_or(any(ind, 1), indT)
    if any(ind):
        # copy is used to prevent y and e being same array, that may be buggy with discret vars
        e[ind] = cs[ind].copy() 
        # Changes
#        ind = logical_and(ind, logical_not(isnan(a[:, n:])))
##        ii = len(where(ind)[0])
##        if ii != 0: print ii
        if p.probType != 'MOP':
            a[:, n:2*n][ind] = a[:, 0:n][ind]
            o[:, n:2*n][ind] = o[:, 0:n][ind]
        if tnlhf_curr is not None:
            tnlhf_curr[:, n:2*n][ind] = tnlhf_curr[:, 0:n][ind]
#        for arr in arrays:
#            if arr is not None:
#                arr[:, n:2*n][ind] = arr[:, 0:n][ind]
        
    return indT


def TruncateByCuttingPlane(f, f_val, y, e, lb, ub, point, gradient):
    gradient_squared_norm = np.sum(gradient**2)
    #gradient_norm = np.sqrt(np.sum(gradient**2))
    #normed_gradient = gradient / gradient_norm
    gradient_multiplier = gradient / gradient_squared_norm
    delta_l = gradient_multiplier * (f_val - lb)
    H = point + delta_l
    
def truncateByPlane(y, e, indT, A, b):
    #!!!!!!!!!!!!!!!!!!!
    # TODO: vectorize it by matrix A
    #!!!!!!!!!!!!!!!!!!!
    ind_trunc = True
    assert np.asarray(b).size <= 1, 'unimplemented yet'
    m, n = y.shape
    if m == 0:
        assert e.size == 0, 'bug in interalg engine'
        return y, e, indT, ind_trunc
    # TODO: remove the cycle
#    for i in range(m):
#        l, u = y[i], e[i]
    ind_positive = where(A > 0)[0]
    
    ind_negative = where(A < 0)[0]
    
    A1 = A[ind_positive] 
    S1 = A1 * y[:, ind_positive]
    A2 = A[ind_negative]
    S2 = A2 * e[:, ind_negative]
    s1, s2 = np.sum(S1, 1), np.sum(S2, 1)
    S = s1 + s2
    
    if ind_positive.size != 0:
        S1_ = b - S.reshape(-1, 1) + S1
        Alt_ub = S1_ / A1
#        ind = e[:, ind_positive] > Alt_ub
#        #e[:, ind_positive[ind]] = Alt_ub[ind]
#        e[:, np.tile(ind_positive,(ind.shape[0],1))[ind]] = Alt_ub[ind]
#        #e[:, ind_positive.reshape(ind.shape)[ind]] = Alt_ub[ind]
#        
#        indT[np.any(ind, 1)] = True
        
        for _i, i in enumerate(ind_positive):
            #s = S - S1[:, _i]
            #alt_ub = (b - s) / A[i]
            #alt_ub = S1_[:, _i] / A[i]
            alt_ub = Alt_ub[:, _i]
            ind = e[:, i] > alt_ub
            e[ind, i] = alt_ub[ind]
            indT[ind] = True
    
    if ind_negative.size != 0:
        S2_ = b - S.reshape(-1, 1) + S2
        Alt_lb = S2_ / A2
        for _i, i in enumerate(ind_negative):
            #s = S - S2[:, _i]
            #alt_lb = (b - s) / A[i]
            #alt_lb = S2_[:, _i] / A[i]
            alt_lb = Alt_lb[:, _i]
            ind = y[:, i] < alt_lb
            y[ind, i] = alt_lb[ind]
            indT[ind] = True

    ind = np.all(e>=y, 1)
    if not np.all(ind):
        ind_trunc = where(ind)[0]
        lj = ind_trunc.size
        y = take(y, ind_trunc, axis=0, out=y[:lj])
        e = take(e, ind_trunc, axis=0, out=e[:lj])
        indT = indT[ind_trunc]
            
    return y, e, indT, ind_trunc

    
def truncateByPlane2(cs, centerValues, y, e, indT, gradient, fo, p):
#    #debug
#    t = np.array([ 0.63056964, -1.        , -1.        , -1.        , -1.        ,       -1.        , -1.        ])
#    cond_present_1 = np.any([logical_and([u>=t for u in e], [l<=t for l in y])])
#    print(cond_present_1)
    
    ind_trunc = True
    assert np.asarray(fo).size <= 1, 'unimplemented yet'
    m, n = y.shape
    if m == 0:
        assert e.size == 0, 'bug in interalg engine'
        return y, e, indT, ind_trunc
    # TODO: remove the cycle
#    for i in range(m):
#        l, u = y[i], e[i]

    oovarsIndDict = p._oovarsIndDict
    ind = np.array([oovarsIndDict[oov][0] for oov in gradient.keys()])
    y2, e2 = y[:, ind], e[:, ind]
    
    #print(gradient)
    A = np.vstack([np.asarray(elem).reshape(1, -1) for elem in gradient.values()]).T
    cs = 0.5 * (y2 + e2)
    #print(gradient.values())
    b = np.sum(A * cs, 1) - centerValues.view(np.ndarray) + fo

#    ind_positive = where(A > 0)
#    ind_negative = where(A < 0)
    
    A_positive = where(A>0, A, 0)
    A_negative = where(A<0, A, 0)
    #S1 = A[ind_positive] * y2[ind_positive]
    #S2 = A[ind_negative] * e2[ind_negative]
    S1 = A_positive * y2
    S2 = A_negative * e2
    s1, s2 = np.sum(S1, 1), np.sum(S2, 1)
    S = s1 + s2
    
    alt_fo1 = where(A_positive != 0, (b.reshape(-1, 1) - S.reshape(-1, 1) + S1) / A_positive, np.inf)
    ind1 = logical_and(e2 > alt_fo1, A_positive != 0)
    e2[ind1] = alt_fo1[ind1]
    
    alt_fo2 = where(A_negative != 0, (b.reshape(-1, 1) - S.reshape(-1, 1) + S2) / A_negative, -np.inf)
    ind2 = logical_and(y2 < alt_fo2, A_negative != 0)
    y2[ind2] = alt_fo2[ind2]
    
    
    # TODO: check it
    y[:, ind], e[:, ind] = y2, e2
    
    # TODO: check indT
    indT[np.any(ind1, 1)] = True
    indT[np.any(ind2, 1)] = True
    
#    for _i, i in enumerate(ind_positive):
#        s = S - S1[:, _i]
#        alt_ub = (b - s) / A[i]
#        ind = e[:, i] > alt_ub
#        e[ind, i] = alt_ub[ind]
#        indT[ind] = True
#    
#    for _i, i in enumerate(ind_negative):
#        s = S - S2[:, _i]
#        alt_lb = (b - s) / A[i]
#        ind = y[:, i] < alt_lb
#        y[ind, i] = alt_lb[ind]
#        indT[ind] = True

    ind = np.all(e>=y, 1)
    if not np.all(ind):
        ind_trunc = where(ind)[0]
        lj = ind_trunc.size
        y = take(y, ind_trunc, axis=0, out=y[:lj])
        e = take(e, ind_trunc, axis=0, out=e[:lj])
        indT = indT[ind_trunc]
        
#    cond_present_2 = np.any([logical_and([u>=t for u in e], [l<=t for l in y])])
#    print('!', cond_present_2)
    
    return y, e, indT, ind_trunc
    
    
    
def  truncateByConvexFunc():
    if 0 and (p._linear_objective or p.convex in (1, True)) and fo_prev < 1e300:# and p.convex is True:
    # TODO: rework it
    #cs = dict([(key, val.view(multiarray)) for key, val in cs.items()])
    #cs = dict([(key, 0) for key, val in p._x0.items()])
    

        # TODO: handle indT corrently
        indT2 = np.empty(y.shape[0])
        indT2.fill(False)

        #y, e, indT2 = truncateByPlane(y, e, indT2, np.hstack([d[v][0] for v in vv]), fo_prev)
        #y, e, indT2, ind_t = truncateByPlane(y, e, indT2, np.hstack([d[oov] for oov in vv]), fo_prev)
        
    #        print('==')
    #        print(y.sum())
        if p._linear_objective:
            d = p._linear_objective_factor
            th = fo_prev + (p._linear_objective_scalar if p.goal not in ('min', 'minimum') else - p._linear_objective_scalar)
            y, e, indT2, ind_t = truncateByPlane(y, e, indT2, d if p.goal in ('min', 'minimum') else -d, th)
        elif p.convex in (1, True):

            assert p.goal in ('min', 'minimum') 
            wr4 = (y+e) / 2
            adjustr4WithDiscreteVariables(wr4, p)
            #cs = dict([(oovar, asarray((y[:, i]+e[:, i])/2, dataType)) for i, oovar in enumerate(vv)])
            cs = dict([(oovar, asarray(wr4[:, i], dataType).view(multiarray)) for i, oovar in enumerate(vv)])            
            #cs = dict([(key, val.view(multiarray)) for key, val in cs.items()])
            
            #TODO: add other args
            centerValues = asdf1(cs)
            gradient = asdf1.D(cs)
            y, e, indT2, ind_t = truncateByPlane2(cs, centerValues, y, e, indT2, gradient, fo_prev, p)
        else:
            assert 0, 'bug in FD kernel'
            
        if ind_t is not True:
            lj = ind_t.size
            o = take(o, ind_t, axis=0, out=o[:lj])
            a = take(a, ind_t, axis=0, out=a[:lj])
            if nlhc is not None:
                nlhc = take(nlhc, ind_t, axis=0, out=nlhc[:lj])
                indTC = np.logical_or(indTC[ind_t], indT2)
            _s = _s[ind_t]
            #residual = residual[ind_t]
        






