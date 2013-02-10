from numpy import empty, where, logical_and, logical_not, take, zeros, isfinite, any, asarray
from interalgT import adjustDiscreteVarBounds, truncateByPlane
    
def processConstraints(C0, y, e, _s, p, dataType):
    n = p.n
    m = y.shape[0]
    indT = empty(m, bool)
    indT.fill(False)
#    isSNLE = p.probType in ('NLSP', 'SNLE')
    
    for i in range(p.nb):
        y, e, indT, ind_trunc = truncateByPlane(y, e, indT, p.A[i], p.b[i]+p.contol)
        if ind_trunc is not True:
            _s = _s[ind_trunc]
    for i in range(p.nbeq):
        # TODO: handle it via one func
        y, e, indT, ind_trunc = truncateByPlane(y, e, indT, p.Aeq[i], p.beq[i]+p.contol)
        if ind_trunc is not True:
            _s = _s[ind_trunc]
        y, e, indT, ind_trunc = truncateByPlane(y, e, indT, -p.Aeq[i], -p.beq[i]+p.contol)
        if ind_trunc is not True:
            _s = _s[ind_trunc]
   
    
    DefiniteRange = True
    if len(p._discreteVarsNumList):
        adjustDiscreteVarBounds(y, e, p)

    m = y.shape[0]
    nlh = zeros((m, 2*n))
    nlh_0 = zeros(m)
    
    
    for c, f, lb, ub, tol in C0:
        
        m = y.shape[0] # is changed in the cycle
        if m == 0: 
            return y.reshape(0, n), e.reshape(0, n), nlh.reshape(0, 2*n), None, True, False, _s
            #return y.reshape(0, n), e.reshape(0, n), nlh.reshape(0, 2*n), residual.reshape(0, 2*n), True, False, _s
        assert nlh.shape[0] == y.shape[0]
        T0, res, DefiniteRange2 = c.nlh(y, e, p, dataType)
        DefiniteRange = logical_and(DefiniteRange, DefiniteRange2)
        
        assert T0.ndim <= 1, 'unimplemented yet'
        nlh_0 += T0
        assert nlh.shape[0] == m
        # TODO: rework it for case len(p._freeVarsList) >> 1
        if len(res):
            for j, v in enumerate(p._freeVarsList):
                tmp = res.get(v, None)
                if tmp is None:
                    continue
                else:
                    nlh[:, n+j] += tmp[:, tmp.shape[1]/2:].flatten() - T0
                    nlh[:, j] += tmp[:, :tmp.shape[1]/2].flatten() - T0
        assert nlh.shape[0] == m
        ind = where(logical_and(any(isfinite(nlh), 1), isfinite(nlh_0)))[0]
        lj = ind.size
        if lj != m:
            y = take(y, ind, axis=0, out=y[:lj])
            e = take(e, ind, axis=0, out=e[:lj])
            nlh = take(nlh, ind, axis=0, out=nlh[:lj])
            nlh_0 = nlh_0[ind]
    #            residual = take(residual, ind, axis=0, out=residual[:lj])
            indT = indT[ind]
            _s = _s[ind]
            if asarray(DefiniteRange).size != 1: 
                DefiniteRange = take(DefiniteRange, ind, axis=0, out=DefiniteRange[:lj])
        assert nlh.shape[0] == y.shape[0]


        ind = logical_not(isfinite((nlh)))
        if any(ind):
            indT[any(ind, 1)] = True
            
            ind_l,  ind_u = ind[:, :ind.shape[1]/2], ind[:, ind.shape[1]/2:]
            tmp_l, tmp_u = 0.5 * (y[ind_l] + e[ind_l]), 0.5 * (y[ind_u] + e[ind_u])
            y[ind_l], e[ind_u] = tmp_l, tmp_u
            # TODO: mb implement it
            if len(p._discreteVarsNumList):
                if tmp_l.ndim > 1:
                    adjustDiscreteVarBounds(tmp_l, tmp_u, p)
                else:
                    adjustDiscreteVarBounds(y, e, p)

            nlh_l, nlh_u = nlh[:, nlh.shape[1]/2:], nlh[:, :nlh.shape[1]/2]
            
            # copy() is used because += and -= operators are involved on nlh in this cycle and probably some other computations
            nlh_l[ind_u], nlh_u[ind_l] = nlh_u[ind_u].copy(), nlh_l[ind_l].copy()        

    # !! matrix - vector
    nlh += nlh_0.reshape(-1, 1)
        
    residual = None

    return y, e, nlh, residual, DefiniteRange, indT, _s




