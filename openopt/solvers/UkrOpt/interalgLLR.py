from numpy import tile, isnan, array, atleast_1d, asarray, logical_and, all, logical_or, any, nan, isinf, \
arange, vstack, inf, where, logical_not, take, abs, hstack, empty, \
prod, int16, int32, int64, log2, searchsorted, cumprod
import numpy as np
from FuncDesigner import oopoint
from interalgT import *

try:
    from bottleneck import nanargmin, nanmin, nanargmax, nanmax
except ImportError:
    from numpy import nanmin, nanargmin, nanargmax, nanmax


#def func82(y, e, vv, f, dataType, pool=None, nProc=1):
#    if pool is None:
#        return func82_seq(y, e, vv, f, dataType)
#    from numpy import array_split
#    yl, el = array_split(y, nProc), array_split(e, nProc)
#    Args = [(yl[i], el[i], vv, lf, uf) for s in ss]
#    R = func82_seq(y, e, vv, f, dataType)
    
def func82(y, e, vv, f, dataType, p, Th = None):
    domain = oopoint([(v, (y[:, i], e[:, i])) for i, v in enumerate(vv)], skipArrayCast=True, isMultiPoint=True)
    domain.dictOfFixedFuncs = p.dictOfFixedFuncs
    #print(Th)
    r, r0 = f.iqg(domain, dataType, UB = Th)
    o_l, o_u, a_l, a_u = [], [], [], []
    definiteRange = r0.definiteRange
    for v in vv:
        # TODO: rework and optimize it
        tmp = r.get(v, None)
        if tmp is not None:
            o_l.append(tmp[0].lb)
            o_u.append(tmp[1].lb)
            a_l.append(tmp[0].ub)
            a_u.append(tmp[1].ub)
            definiteRange = logical_and(definiteRange, tmp[0].definiteRange)
            definiteRange = logical_and(definiteRange, tmp[1].definiteRange)
        else:
            o_l.append(r0.lb)
            o_u.append(r0.lb)
            a_l.append(r0.ub)
            a_u.append(r0.ub)
            #definiteRange = logical_and(definiteRange, r0.definiteRange)
    o, a = hstack(o_l+o_u), hstack(a_l+a_u)    
    return o, a, definiteRange

def func10(y, e, vv):
    m, n = y.shape
    LB = [[] for i in range(n)]
    UB = [[] for i in range(n)]

    r4 = (y + e) / 2
    
    # TODO: remove the cycle
    #T1, T2 = tile(y, (2*n,1)), tile(e, (2*n,1))
    
    for i in range(n):
        t1, t2 = tile(y[:, i], 2*n), tile(e[:, i], 2*n)
        #t1, t2 = T1[:, i], T2[:, i]
        #T1[(n+i)*m:(n+i+1)*m, i] = T2[i*m:(i+1)*m, i] = r4[:, i]
        t1[(n+i)*m:(n+i+1)*m] = t2[i*m:(i+1)*m] = r4[:, i]
        
#        if vv[i].domain is bool:
#            indINQ = y[:, i] != e[:, i]
#            tmp = t1[(n+i)*m:(n+i+1)*m]
#            tmp[indINQ] = 1
#            tmp = t2[i*m:(i+1)*m]
#            tmp[indINQ] = 0
            
#        if vv[i].domain is bool:
#            t1[(n+i)*m:(n+i+1)*m] = 1
#            t2[i*m:(i+1)*m] = 0
#        else:
#            t1[(n+i)*m:(n+i+1)*m] = t2[i*m:(i+1)*m] = r4[:, i]
        
        LB[i], UB[i] = t1, t2


####        LB[i], UB[i] = T1[:, i], T2[:, i]

#    sh1, sh2, inds = [], [], []
#    for i in range(n):
#        sh1+= arange((n+i)*m, (n+i+1)*m).tolist()
#        inds +=  [i]*m
#        sh2 += arange(i*m, (i+1)*m).tolist()

#    sh1, sh2, inds = asdf(m, n)
#    asdf2(T1, T2, r4, sh1, sh2, inds)
    
    #domain = dict([(v, (T1[:, i], T2[:, i])) for i, v in enumerate(vv)])
    domain = dict([(v, (LB[i], UB[i])) for i, v in enumerate(vv)])
    
    domain = oopoint(domain, skipArrayCast = True)
    domain.isMultiPoint = True
    return domain

def func8(domain, func, dataType):
    TMP = func.interval(domain, dataType)
    #assert TMP.lb.dtype == dataType
    return asarray(TMP.lb, dtype=dataType), asarray(TMP.ub, dtype=dataType), TMP.definiteRange

def getr4Values(vv, y, e, tnlh, func, C, contol, dataType, p, fo = inf):
    #print where(y), where(e!=1)
    n = y.shape[1]
    # TODO: rework it wrt nlh
    #cs = dict([(key, asarray((val[0]+val[1])/2, dataType)) for key, val in domain.items()])
    if tnlh is None:# or p.probType =='MOP':
        wr4 = (y+e) / 2
        adjustr4WithDiscreteVariables(wr4, p)
        #cs = dict([(oovar, asarray((y[:, i]+e[:, i])/2, dataType)) for i, oovar in enumerate(vv)])
        cs = dict([(oovar, asarray(wr4[:, i], dataType)) for i, oovar in enumerate(vv)])
    else:
        tnlh = tnlh.copy()
        tnlh[atleast_1d(tnlh<1e-300)] = 1e-300 # to prevent division by zero
        tnlh[atleast_1d(isnan(tnlh))] = inf #- check it!
        tnlh_l_inv, tnlh_u_inv = 1.0 / tnlh[:, :n], 1.0 / tnlh[:, n:]
        wr4 = (y * tnlh_l_inv + e * tnlh_u_inv) / (tnlh_l_inv + tnlh_u_inv)
        ind = tnlh_l_inv == tnlh_u_inv # especially important for tnlh_l_inv == tnlh_u_inv = 0
        wr4[ind] = (y[ind] + e[ind]) / 2
        adjustr4WithDiscreteVariables(wr4, p)
        cs = dict([(oovar, asarray(wr4[:, i], dataType)) for i, oovar in enumerate(vv)])
        
    cs = oopoint(cs, skipArrayCast = True)
    cs.isMultiPoint = True
    cs.update(p.dictOfFixedFuncs)
    
    m = y.shape[0]
    
    kw =  {'Vars': p.freeVars} if p.freeVars is None or (p.fixedVars is not None and len(p.freeVars) < len(p.fixedVars)) else {'fixedVars': p.fixedVars}
    kw['fixedVarsScheduleID'] = p._FDVarsID
    if len(C) != 0:
        r15 = empty(m, bool)
        r15.fill(True)
        for _c, f, r16, r17 in C:
            c = f(cs, **kw)
            ind = logical_and(c  >= r16, c <= r17) # here r16 and r17 are already shifted by required tolerance
            r15 = logical_and(r15, ind)
    else:
        r15 = True
        
    isMOP = p.probType =='MOP'
    if not any(r15):
        F = empty(m, dataType)
        F.fill(2**31-2 if dataType in (int16, int32, int64, int) else nan) 
        if isMOP:
            FF = [F for k in range(p.nf)]
    elif all(r15):
        if isMOP:
            FF = [fun(cs, **kw) for fun in func]
        else:
            F = func(cs, **kw) if func is not None else zeros(m) # func is None for SNLE
    else:
        #cs = dict([(oovar, (y[r15, i] + e[r15, i])/2) for i, oovar in enumerate(vv)])
        #cs = ooPoint(cs, skipArrayCast = True)
        #cs.isMultiPoint = True
        if isMOP:
            FF = []
            for fun in func:
                tmp = fun(cs, **kw) 
                F = empty(m, dataType)
                F.fill(2**31-15 if dataType in (int16, int32, int64, int) else nan)
                F[r15] = tmp[r15]    
                FF.append(F)
        else:
            tmp = asarray(func(cs, **kw), dataType) if func is not None else zeros(m) # func is None for SNLE
            F = empty(m, dataType)
            F.fill(2**31-15 if dataType in (int16, int32, int64, int) else nan)
            F[r15] = tmp[r15]
    if isMOP:
        return array(FF).T.reshape(m, len(func)).tolist(), wr4.tolist()
    else:
        return atleast_1d(F) , wr4


def adjustr4WithDiscreteVariables(wr4, p):
    for i in p._discreteVarsNumList:
        v = p._freeVarsList[i]
        
        if v.domain is bool or v.domain is 'bool':
            wr4[:, i] = where(wr4[:, i]<0.5, 0, 1)
        else:
            tmp = wr4[:, i]
            d = v.domain 
            ind = searchsorted(d, tmp, side='left')
            ind2 = searchsorted(d, tmp, side='right')
            ind3 = where(ind!=ind2)[0]
            Tmp = tmp[ind3].copy()
            
            ind[ind==d.size] -= 1 # may be due to roundoff errors
            ind[ind==1] = 0
            ind2[ind2==d.size] -=1
            ind2[ind2==0] = 1 # may be due to roundoff errors
            tmp1 = asarray(d[ind], p.solver.dataType)
            tmp2 = asarray(d[ind2], p.solver.dataType)
            if Tmp.size!=0:
                if str(tmp1.dtype).startswith('int'):
                    Tmp = asarray(Tmp, p.solver.dataType)
                tmp2[ind3] = Tmp.copy()
                tmp1[ind3] = Tmp.copy()
            tmp = where(abs(tmp-tmp1)<abs(tmp-tmp2), tmp1, tmp2)
            #print max(abs(tmp-tmp1)), max(abs(tmp-tmp2))
            wr4[:, i] = tmp
    #print where(wr4==0)[0].size, where(wr4==1)[0].size

def r2(PointVals, PointCoords, dataType):
    r23 = nanargmin(PointVals)
    if isnan(r23):
        r23 = 0
    # TODO: check it , maybe it can be improved
    #bestCenter = cs[r23]
    #r7 = array([(val[0][r23]+val[1][r23]) / 2 for val in domain.values()], dtype=dataType)
    #r8 = atleast_1d(r3)[r23] if not isnan(r23) else inf
    r7 = array(PointCoords[r23], dtype=dataType)
    r8 = atleast_1d(PointVals)[r23] 
    return r7, r8
    
def func3(an, maxActiveNodes, dataHandling):
    m = len(an)
    if m <= maxActiveNodes:
        return an, array([], object)
    
    an1, _in = an[:maxActiveNodes], an[maxActiveNodes:]    
        
    if getattr(an1[0], 'tnlh_curr_best', None) is not None:
        #t0 = an1[0].tnlh_curr_best
        tnlh_curr_best_values = asarray([node.tnlh_curr_best for node in an1])
        
        #changes
        tmp = 2 ** (-tnlh_curr_best_values)
        Tmp = -cumprod(1.0-tmp)
        ind2 = searchsorted(Tmp, -0.05)
        #changes end
        
#        ind = min((ind, ind2))
        ind = ind2
        
#        ind = maxActiveNodes
        #M = max((5, maxActiveNodes/2, ind))
        #M = ind
        n = an[0].y.size
        M = max((5, int(maxActiveNodes/n), ind))
        M = ind
#        M = max((5, maxActiveNodes/5, ind))
        
        # IMPORTANT!
        if M == 0: M = 1
        
        tmp1, tmp2 = an1[:M], an1[M:]
        an1 = tmp1
        _in = hstack((tmp2, _in))
        
    # TODO: implement it for MOP as well
    cond_min_uf = 0 and dataHandling == 'raw' and hasattr(an[0], 'key')            
    
    if cond_min_uf:
        num_nlh = min((max((1, int(0.8*maxActiveNodes))), an1.size))
        num_uf = min((maxActiveNodes - num_nlh, int(maxActiveNodes/2)))
        if num_uf < 15:
            num_uf = 15
        #an1, _in = an[:num_nlh], an[num_nlh:]    
        Ind = np.argsort([node.key for node in _in])
        min_uf_nodes = _in[Ind[:num_uf]]
        _in = _in[Ind[num_uf:]]
        an1 = np.hstack((an1, min_uf_nodes))
        
    # changes end
    #print maxActiveNodes, len(an1), len(_in)
    
    return an1, _in

def func1(tnlhf, tnlhf_curr, residual, y, e, o, a, _s_prev, p, indT):
    m, n = y.shape
    w = arange(m)
    
    if p.probType == 'IP':
        oc_modL, oc_modU = o[:, :n], o[:, n:]
        ac_modL, ac_modU = a[:, :n], a[:, n:]
#            # TODO: handle nans
        mino = where(oc_modL < oc_modU, oc_modL, oc_modU)
        maxa = where(ac_modL < ac_modU, ac_modU, ac_modL)
    
        # Prev
        tmp = a[:, 0:n]-o[:, 0:n]+a[:, n:]-o[:, n:]
        t = nanargmin(tmp,1)
        d = 0.5*tmp[w, t]
        
        
        #New
#        tmp = a - o
#        t_ = nanargmin(tmp,1)
#        t = t_% n
#        d = tmp[w, t_]

#        ind = 2**(-n) >= (_s_prev - d)/asarray(d, 'float64')
        ind = 2**(1.0/n) * d >= _s_prev
        #new
#        ind = 2**(1.0/n) * d >= nanmax(maxa-mino, 1)
        
        #ind = 2**(-n) >= (_s_prev - _s)/asarray(_s, 'float64')
    
        #s2 = nanmin(maxa - mino, 1)
        #print (abs(s2/_s))
        
        # Prev
        _s = nanmin(maxa - mino, 1)
        
        # New
        #_s = nanmax(maxa - mino, 1)
#        _s = nanmax(a - o, 1)
        
        #ind = _s_prev  <= _s + ((2**-n / log(2)) if n > 15 else log2(1+2**-n)) 
        indD = logical_not(ind)
        indD = ind
        indD = None
        #print len(where(indD)[0]), len(where(logical_not(indD))[0])
#    elif p.probType == 'MOP':
#
#        raise 'unimplemented'
    else:
        if p.solver.dataHandling == 'sorted':
            _s = func13(o, a)
            t = nanargmin(a, 1) % n
            d = nanmax([a[w, t] - o[w, t], 
                    a[w, n+t] - o[w, n+t]], 0)
            
            ## !!!! Don't replace it by (_s_prev /d- 1) to omit rounding errors ###
            #ind = 2**(-n) >= (_s_prev - d)/asarray(d, 'float64')
            
            #NEW
            ind = d  >=  _s_prev / 2 ** (1.0e-12/n)
            #ind = d  >=  _s_prev / 2 ** (1.0/n)
            indD = empty(m, bool)
            indD.fill(True)
            #ind.fill(False)
            ###################################################
        elif p.solver.dataHandling == 'raw':
            if p.probType == 'MOP':
                t = p._t[:m]
                p._t = p._t[m:]
                d = _s = p.__s[:m]
                p.__s = p.__s[m:]
            else:
#                tnlh_1, tnlh_2 = tnlhf[:, 0:n], tnlhf[:, n:]
#                TNHLF_min =  where(logical_or(tnlh_1 > tnlh_2, isnan(tnlh_1)), tnlh_2, tnlh_1)
#               # Set _s
#                _s = nanmin(TNHLF_min, 1)
                T = tnlhf_curr
                tnlh_curr_1, tnlh_curr_2 = T[:, 0:n], T[:, n:]
                TNHL_curr_min =  where(logical_or(tnlh_curr_1 < tnlh_curr_2, isnan(tnlh_curr_2)), tnlh_curr_1, tnlh_curr_2)
                t = nanargmin(TNHL_curr_min, 1)
                T = tnlhf
                d = nanmin(vstack(([T[w, t], T[w, n+t]])), 0)
                _s = d

            #OLD
            #!#!#!#! Don't replace it by _s_prev - d <= ... to omit inf-inf = nan !#!#!#
            #ind = _s_prev  <= d + ((2**-n / log(2)) if n > 15 else log2(1+2**-n)) 
            #ind = _s_prev - d <= ((2**-n / log(2)) if n > 15 else log2(1+2**-n)) 
            
            #NEW
            if any(_s_prev < d):
                pass
            ind = _s_prev  <= d + 1.0/n
#            T = TNHL_curr_min
            #ind2 = nanmin(TNHL_curr_min, 0)
            
            indQ = d >= _s_prev - 1.0/n 
            #indQ = logical_and(indQ, False)
            indD = logical_or(indQ, logical_not(indT))
#            print _s_prev[:2], d[:2]
            #print len(where(indD)[0]), len(where(indQ)[0]), len(where(indT)[0])
            #print _s_prev - d
            ###################################################
            #d = ((tnlh[w, t]* tnlh[w, n+t])**0.5)
        else:
            assert 0

    if any(ind):
        r10 = where(ind)[0]
        #print('r10:', r10)
#        print _s_prev
#        print ((_s_prev -d)*n)[r10]
#        print('ind length: %d' % len(where(ind)[0]))
#        print where(ind)[0].size
        #bs = e[ind] - y[ind]
        #t[ind] = nanargmax(bs, 1) # ordinary numpy.argmax can be used as well
        bs = e[r10] - y[r10]
        t[r10] = nanargmax(bs, 1) # ordinary numpy.argmax can be used as well

    return t, _s, indD
    
def func13(o, a): 
    m, n = o.shape
    n /= 2
#    if case == 1:
#        U1, U2 = a[:, :n].copy(), a[:, n:] 
#        #TODO: mb use nanmax(concatenate((U1,U2),3),3) instead?
#        U1 = where(logical_or(U1<U2, isnan(U1)),  U2, U1)
#        return nanmin(U1, 1)
        
    L1, L2, U1, U2 = o[:, :n], o[:, n:], a[:, :n], a[:, n:] 
#    if case == 2:
    U = where(logical_or(U1<U2, isnan(U1)),  U2, U1)
    L = where(logical_or(L2<L1, isnan(L1)), L2, L1)
    return nanmax(U-L, 1)

def func2(y, e, t, vv, tnlhf_curr):
    new_y, new_e = y.copy(), e.copy()
    m, n = y.shape
    w = arange(m)
    
    #!!!! TODO: omit or imporove it for all-float problems    
    th = (new_y[w, t] + new_e[w, t]) / 2
    
    ### !!!!!!!!!!!!!!!!!!!!!
    # TODO: rework it for integer dataType 
    
    BoolVars = [(v.domain is bool or v.domain is 'bool') for v in vv]
    if not str(th.dtype).startswith('float') and any(BoolVars):
        indBool = where(BoolVars)[0]
        if len(indBool) != n:
            #boolCoords = list(set(indBool) & set(t))
            boolCoords = where([t[j] in indBool for j in range(m)])[0]
            new_y[w, t] = th
            new_e[w, t] = th
            new_y[boolCoords, t[boolCoords]] = 1
            new_e[boolCoords, t[boolCoords]] = 0
        else:
            new_y[w, t] = 1
            new_e[w, t] = 0
    else:
        new_y[w, t] = th
        new_e[w, t] = th
    
    new_y = vstack((y, new_y))
    new_e = vstack((new_e, e))
    
    if tnlhf_curr is not None:
        tnlhf_curr_local = hstack((tnlhf_curr[w, t], tnlhf_curr[w, n+t]))
    else:
        tnlhf_curr_local = None
    return new_y, new_e, tnlhf_curr_local


def func12(an, maxActiveNodes, p, Solutions, vv, varTols, fo):
    solutions, r6 = Solutions.solutions, Solutions.coords
    if len(an) == 0:
        return array([]), array([]), array([]), array([])
    _in = an
    
    if r6.size != 0:
        r11, r12 = r6 - varTols, r6 + varTols
    y, e, S = [], [], []
    Tnlhf_curr_local = []
    n = p.n
    N = 0
    maxSolutions = p.maxSolutions
    
#    new = 1
#    # new
#    if new and p.probType in ('MOP', 'SNLE', 'GLP', 'NLP', 'MINLP') and p.maxSolutions == 1:
#        
#        
#        return y, e, _in, _s
        
    
    while True:
        an1Candidates, _in = func3(_in, maxActiveNodes, p.solver.dataHandling)

        #print nanmax(2**(-an1Candidates[0].tnlh_curr)) ,  nanmax(2**(-an1Candidates[-1].tnlh_curr))
        yc, ec, oc, ac, SIc = asarray([t.y for t in an1Candidates]), \
        asarray([t.e for t in an1Candidates]), \
        asarray([t.o for t in an1Candidates]), \
        asarray([t.a for t in an1Candidates]), \
        asarray([t._s for t in an1Candidates])

        
        if p.probType == 'MOP':
            tnlhf_curr = asarray([t.tnlh_all for t in an1Candidates])
            tnlhf = None        
        elif p.solver.dataHandling == 'raw':
            tnlhf = asarray([t.tnlhf for t in an1Candidates]) 
            tnlhf_curr = asarray([t.tnlh_curr for t in an1Candidates]) 
        else:
            tnlhf, tnlhf_curr = None, None
        
        
        if p.probType != 'IP': 
            #nlhc = asarray([t.nlhc for t in an1Candidates])
            indtc = asarray([t.indtc for t in an1Candidates])
            #residual = asarray([t.residual for t in an1Candidates]) 
            residual = None
            
            indT = func4(p, yc, ec, oc, ac, fo, tnlhf_curr)

            if indtc[0] is not None:
                indT = logical_or(indT, indtc)
        else:
            residual = None
            indT = None
        t, _s, indD = func1(tnlhf, tnlhf_curr, residual, yc, ec, oc, ac, SIc, p, indT)

        new = 0
        nn = 0
        if new and p.probType in ('MOP', 'SNLE', 'NLSP','GLP', 'NLP', 'MINLP') and p.maxSolutions == 1:
            arr = tnlhf_curr if p.solver.dataHandling == 'raw' else oc
            M = arr.shape[0]
            w = arange(M)
            Midles = 0.5*(yc[w, t] + ec[w, t])
            arr_1, arr2 = arr[w, t], arr[w, n+t]
            Arr = hstack((arr_1, arr2))
            ind = np.argsort(Arr)
            Ind = set(ind[:maxActiveNodes])
            tag_all, tag_1, tag_2 = [], [], []
            sn = []
            
            # TODO: get rid of the cycles
            for i in range(M):
                cond1, cond2 = i in Ind, (i+M) in Ind
                if cond1:
                    if cond2:
                        tag_all.append(i)
                    else:
                        tag_1.append(i)
                else:
                    if cond2:
                        tag_2.append(i)
                    else:
                        sn.append(an1Candidates[i])

            list_lx, list_ux = [], []
            
            _s_new = []
            updateTC = an1Candidates[0].indtc is not None
            isRaw = p.solver.dataHandling == 'raw'
            for i in tag_1:
                node = an1Candidates[i]
                I = t[i]
#                if node.o[n+I] >= node.o[I]:
#                    print '1'
#                else:
#                    print i, I, node.o[n+I] ,  node.o[I], node.key, node.a[n+I] ,  node.a[I], node.nlhc[n+I], node.nlhc[I]
                node.key = node.o[n+I]
                node._s = _s[i]
                
                if isRaw:
                    node.tnlh_curr[I] = node.tnlh_curr[n+I]
                    node.tnlh_curr_best = nanmin(node.tnlh_curr)
                
                #assert node.o[n+I] >= node.o[I]
                #lx, ux = node.y, node.e
                lx, ux = yc[i], ec[i]
                if nn:
                    #node.o[I], node.a[I] = node.o[n+I], node.a[n+I]
                    node.o[I], node.a[I] = node.o[n+I], node.a[n+I]
                    node.o[node.o<node.o[n+I]], node.a[node.a>node.a[n+I]] = node.o[n+I], node.a[n+I]
                else:
                    node.o[n+I], node.a[n+I] = node.o[I], node.a[I]
                    node.o[node.o<node.o[I]], node.a[node.a>node.a[I]] = node.o[I], node.a[I]

#                if p.solver.dataHandling == 'raw':
                for Attr in ('nlhf','nlhc', 'tnlhf', 'tnlh_curr', 'tnlh_all'):
                    r = getattr(node, Attr, None)
                    if r is not None:
                        if nn: r[I] = r[n+I]
                        else: 
                            r[n+I] = r[I]

                mx = ux.copy()
                mx[I] = Midles[i]#0.5*(lx[I] + ux[I])
                list_lx.append(lx)
                list_ux.append(mx)
                node.y = lx.copy()
                node.y[I] = Midles[i]#0.5*(lx[I] + ux[I])
                if updateTC: 
                    node.indtc = True
                
                _s_new.append(node._s)
                sn.append(node)
            
            for i in tag_2:
                node = an1Candidates[i]
                I = t[i]
                node.key = node.o[I]

                
                node._s = _s[i]
                
                # for raw only
                if isRaw:
                    node.tnlh_curr[n+I] = node.tnlh_curr[I]
                    node.tnlh_curr_best = nanmin(node.tnlh_curr)
                
                #assert node.o[I] >= node.o[n+I]
                #lx, ux = node.y, node.e
                lx, ux = yc[i], ec[i]

                if nn:
                    node.o[n+I], node.a[n+I] = node.o[I], node.a[I]
                    node.o[node.o<node.o[I]], node.a[node.a>node.a[I]] = node.o[I], node.a[I]
                else:
                    node.o[I], node.a[I] = node.o[n+I], node.a[n+I]
                    node.o[node.o<node.o[n+I]], node.a[node.a>node.a[n+I]] = node.o[n+I], node.a[n+I]
                for Attr in ('nlhf','nlhc', 'tnlhf', 'tnlh_curr', 'tnlh_all'):
                    r = getattr(node, Attr, None)
                    if r is not None:
                        if nn: r[n+I] = r[I]
                        else: 
                            r[I] = r[n+I]

                mx = lx.copy()
                mx[I] = Midles[i]#0.5*(lx[I] + ux[I])
                list_lx.append(mx)
                list_ux.append(ux)
                node.e = ux.copy()
                node.e[I] = Midles[i]#0.5*(lx[I] + ux[I])
                if updateTC: 
                    node.indtc = True

                _s_new.append(node._s)
                sn.append(node)
            
            for i in tag_all:
                node = an1Candidates[i]
                I = t[i]
                
                #lx, ux = node.y, node.e
                lx, ux = yc[i], ec[i]
                mx = ux.copy()
                mx[I] = Midles[i]#0.5 * (lx[I] + ux[I])
                
                list_lx.append(lx)
                list_ux.append(mx)
                
                mx = lx.copy()
                mx[I] = Midles[i]#0.5 * (lx[I] + ux[I])
                #mx[n+ t] = 0.5 * (lx[n + t] + ux[n + t])
                list_lx.append(mx)
                list_ux.append(ux)
                
                #_s_new += [_s[i]] * 2
                _s_new.append(_s[i])
                _s_new.append(_s[i])
                
#            print 'y_new:', vstack(list_lx)
#            print 'e_new:', vstack(list_ux)
#            print '_s_new:', hstack(_s)
            _in = sn + _in.tolist()
            if p.solver.dataHandling == 'sorted':
                _in.sort(key = lambda obj: obj.key)
            else:
                #pass
                _in.sort(key = lambda obj: obj.tnlh_curr_best)
#            print 'tag 1:', len(tag_1), 'tag 2:', len(tag_2), 'tag all:', len(tag_all)
#            print 'lx:', list_lx
#            print 'sn lx:', [node.y for node in sn]
#            print 'ux:', list_ux
#            print 'sn ux:', [node.e for node in sn]
#            print '-'*10
            #print '!', vstack(list_lx), vstack(list_ux), hstack(_s_new)
            NEW_lx, NEW_ux, NEW__in, NEW__s = \
            vstack(list_lx), vstack(list_ux), array(_in), hstack(_s_new)
            return NEW_lx, NEW_ux, NEW__in, NEW__s
        
        NewD = 1
        
        if NewD and indD is not None: 
            s4d = _s[indD]
            sf = _s[logical_not(indD)]

            _s = hstack((s4d, s4d, sf))
            yf, ef = yc[logical_not(indD)], ec[logical_not(indD)]
            yc, ec = yc[indD], ec[indD]
            t = t[indD]
        else:
            _s = tile(_s, 2)

        yc, ec, tnlhf_curr_local = func2(yc, ec, t, vv, tnlhf_curr)

        if NewD and indD is not None:
            yc = vstack((yc, yf))
            ec = vstack((ec, ef))
            
        if maxSolutions == 1 or len(solutions) == 0: 
            y, e, Tnlhf_curr_local = yc, ec, tnlhf_curr_local
            break
        
        # TODO: change cycle variable if len(solutions) >> maxActiveNodes
        for i in range(len(solutions)):
            ind = logical_and(all(yc >= r11[i], 1), all(ec <= r12[i], 1))
            if any(ind):
                j = where(logical_not(ind))[0]
                lj = j.size
                yc = take(yc, j, axis=0, out=yc[:lj])
                ec = take(ec, j, axis=0, out=ec[:lj])
                _s = _s[j]
#                if tnlhf_curr_local is not None:
#                    tnlhf_curr_local = tnlhf_curr_local[j]
        y.append(yc)
        e.append(ec)
        S.append(_s)
        #Tnlhf_curr_local.append(tnlhf_curr_local)
        N += yc.shape[0]
        if len(_in) == 0 or N >= maxActiveNodes: 
            y, e, _s = vstack(y), vstack(e), hstack(S)
            #Tnlhf_curr_local = hstack(Tnlhf_curr_local)
            break
            
#    if Tnlhf_curr_local is not None and len(Tnlhf_curr_local) != 0 and Tnlhf_curr_local[0] is not None:
#        #print len(where(isfinite(Tnlhf_curr_local))[0]), Tnlhf_curr_local.size
#        pass

#    print 'y_prev:', y
#    print 'e_prev:', e
#    print '_s_prev:', hstack(_s)
    #print 'prev!', y, e, _s
    
#    from numpy import array_equal
#    if not array_equal(NEW_lx.sort(), y.sort()):
#        pass
#    if not array_equal(NEW_ux.sort(), e.sort()):
#        pass
#    if not array_equal(NEW__s.sort(), _s.sort()):
#        pass
        
        #, NEW_ux, NEW__in, NEW__s
    return y, e, _in, _s

Fields = ['key', 'y', 'e', 'nlhf','nlhc', 'indtc','residual','o', 'a', '_s']
MOP_Fields = ['y', 'e', 'nlhf','nlhc', 'indtc','residual','o', 'a', '_s']

#FuncValFields = ['key', 'y', 'e', 'nlhf','nlhc', 'o', 'a', '_s','r18', 'r19']
IP_fields = ['key', 'minres', 'minres_ind', 'complementary_minres', 'y', 'e', 'o', 'a', '_s','F', 'volume', 'volumeResidual']

def func11(y, e, nlhc, indTC, residual, o, a, _s, p): 
    m, n = y.shape
    if p.probType == "IP":
        w = arange(m)
        # TODO: omit recalculation from func1
        ind = nanargmin(a[:, 0:n] - o[:, 0:n] + a[:, n:] - o[:, n:], 1)
        sup_inf_diff = 0.5*(a[w, ind] - o[w, ind] + a[w, n+ind] - o[w, n+ind])
        diffao = a - o
        minres_ind = nanargmin(diffao, 1) 
        minres = diffao[w, minres_ind]
        complementary_minres = diffao[w, where(minres_ind<n, minres_ind+n, minres_ind-n)]
        volume = prod(e-y, 1)
        volumeResidual = volume * sup_inf_diff
        F = 0.25 * (a[w, ind] + o[w, ind] + a[w, n+ind] + o[w, n+ind])
        return [si(IP_fields, sup_inf_diff[i], minres[i], minres_ind[i], complementary_minres[i], y[i], e[i], o[i], a[i], _s[i], F[i], volume[i], volumeResidual[i]) for i in range(m)]
        
    else:
        
        residual = None
        tmp = asarray(a)-asarray(o)
        tmp[tmp<1e-300] = 1e-300
        nlhf = log2(tmp)#-log2(p.fTol)
#        nlhf[a==inf] = 1e300# to make it not inf and nan
#        nlhf[o==-inf] = 1e300# to make it not inf and nan
        if nlhf.ndim == 3: # in MOP
            nlhf = nlhf.sum(axis=1)
        
        if p.probType == "MOP":
            # make correct o,a wrt each target
            return [si(MOP_Fields, y[i], e[i], nlhf[i], 
                          nlhc[i] if nlhc is not None else None, 
                          indTC[i] if indTC is not None else None, 
                          residual[i] if residual is not None else None, 
                          [o[i][k] for k in range(p.nf)], [a[i][k] for k in range(p.nf)], 
                          _s[i]) for i in range(m)]
        else:
            s, q = o[:, 0:n], o[:, n:2*n]
            Tmp = nanmax(where(q<s, q, s), 1)
            
            nlhf[logical_and(isinf(a), isinf(nlhf))] = 1e300
            assert p.probType in ('GLP', 'NLP', 'NSP', 'SNLE', 'NLSP', 'MINLP')
        
#            residual = None

            return [si(Fields, Tmp[i], y[i], e[i], nlhf[i], 
                          nlhc[i] if nlhc is not None else None, 
                          indTC[i] if indTC is not None else None, 
                          residual[i] if residual is not None else None, 
                          o[i], a[i], _s[i]) for i in range(m)]
    
#    else:
#        r18, r19 = r3[:, :n], r3[:, n:]
#        return [si(FuncValFields, Tmp[i], y[i], e[i], nlhf[i], nlhc[i] if nlhc is not None else None, o[i], a[i], _s[i], r18[i], r19[i]) for i in range(m)]

class si:
    def __init__(self, fields, *args, **kwargs):
        for i in range(len(fields)):
            setattr(self, fields[i], args[i])
    
