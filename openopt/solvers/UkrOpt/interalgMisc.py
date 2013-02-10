from numpy import isnan, array, atleast_1d, asarray, all, searchsorted, logical_or, any, nan, \
vstack, inf, where, logical_not, min, abs, hstack, insert, logical_xor, argsort
try:
    from numpy import append
except ImportError:
    def append(*args, **kw):
        raise ImportError('function append() is absent in PyPy yet')
        
from interalgLLR import *

try:
    from bottleneck import nanmin, nanmax
except ImportError:
    from numpy import nanmin, nanmax


def r14(p, nlhc, residual, definiteRange, y, e, vv, asdf1, C, r40, g, nNodes,  \
         r41, fTol, Solutions, varTols, _in, dataType, \
         maxNodes, _s, indTC, xRecord):

    isSNLE = p.probType in ('NLSP', 'SNLE')

    maxSolutions, solutions, coords = Solutions.maxNum, Solutions.solutions, Solutions.coords
    if len(p._discreteVarsNumList):
        y, e = adjustDiscreteVarBounds(y, e, p)

    
    o, a, r41 = r45(y, e, vv, p, asdf1, dataType, r41, nlhc)
    fo_prev = float(0 if isSNLE else min((r41, r40 - (fTol if maxSolutions == 1 else 0))))
    if fo_prev > 1e300:
        fo_prev = 1e300
    y, e, o, a, _s, indTC, nlhc, residual = func7(y, e, o, a, _s, indTC, nlhc, residual)    

    if y.size == 0:
        return _in, g, fo_prev, _s, Solutions, xRecord, r41, r40
    
    nodes = func11(y, e, nlhc, indTC, residual, o, a, _s, p)
    #nodes, g = func9(nodes, fo_prev, g, p)
    #y, e = func4(y, e, o, a, fo)
    

    if p.solver.dataHandling == 'raw':
        
        tmp = o.copy()
        tmp[tmp > fo_prev] = -inf
        M = atleast_1d(nanmax(tmp, 1))
        for i, node in enumerate(nodes):
            node.th_key = M[i]
            
        if not isSNLE:
            for node in nodes:
                node.fo = fo_prev       
        if nlhc is not None:
            for i, node in enumerate(nodes): node.tnlhf = node.nlhf + node.nlhc
        else:
            for i, node in enumerate(nodes): node.tnlhf = node.nlhf # TODO: improve it
            
        an = hstack((nodes, _in))
        
        #tnlh_fixed = vstack([node.tnlhf for node in an])
        tnlh_fixed_local = vstack([node.tnlhf for node in nodes])#tnlh_fixed[:len(nodes)]

        tmp = a.copy()

        
        tmp[tmp>fo_prev] = fo_prev
        tmp2 = tmp - o
        tmp2[tmp2<1e-300] = 1e-300
        tmp2[o > fo_prev] = nan
        tnlh_curr = tnlh_fixed_local - log2(tmp2)
        tnlh_curr_best = nanmin(tnlh_curr, 1)
        for i, node in enumerate(nodes):
            node.tnlh_curr = tnlh_curr[i]
            node.tnlh_curr_best = tnlh_curr_best[i]
        
        # TODO: use it instead of code above
        #tnlh_curr = tnlh_fixed_local - log2(where() - o)
    else:
        tnlh_curr = None
    
    # TODO: don't calculate PointVals for zero-p regions
    PointVals, PointCoords = getr4Values(vv, y, e, tnlh_curr, asdf1, C, p.contol, dataType, p) 

    if PointVals.size != 0:
        xk, Min = r2(PointVals, PointCoords, dataType)
    else: # all points have been removed by func7
        xk = p.xk
        Min = nan

    if r40 > Min:
        r40 = Min
        xRecord = xk.copy()# TODO: is copy required?
    if r41 > Min:
        r41 = Min
    
    fo = float(0 if isSNLE else min((r41, r40 - (fTol if maxSolutions == 1 else 0))))
        
    if p.solver.dataHandling == 'raw':
        
        if fo != fo_prev and not  isSNLE:
            fos = array([node.fo for node in an])
            
            #prev
            #ind_update = where(fos > fo + 0.01* fTol)[0]
            
            #new
            th_keys = array([node.th_key for node in an])
            delta_fos = fos - fo
            ind_update = where(10 * delta_fos > fos - th_keys)[0]
            
            nodesToUpdate = an[ind_update]
            update_nlh = True if ind_update.size != 0 else False
#                  print 'o MB:', float(o_tmp.nbytes) / 1e6
#                  print 'percent:', 100*float(ind_update.size) / len(an) 
            if update_nlh:
#                    from time import time
#                    tt = time()
                updateNodes(nodesToUpdate, fo)
#                    if not hasattr(p, 'Time'):
#                        p.Time = time() - tt
#                    else:
#                        p.Time += time() - tt
                    
            tmp = asarray([node.key for node in an])
            r10 = where(tmp > fo)[0]
            if r10.size != 0:
                mino = [an[i].key for i in r10]
                mmlf = nanmin(asarray(mino))
                g = nanmin((g, mmlf))

        NN = atleast_1d([node.tnlh_curr_best for node in an])
        r10 = logical_or(isnan(NN), NN == inf)
       
        if any(r10):
            ind = where(logical_not(r10))[0]
            an = an[ind]
            #tnlh = take(tnlh, ind, axis=0, out=tnlh[:ind.size])
            #NN = take(NN, ind, axis=0, out=NN[:ind.size])
            NN = NN[ind]

        if not isSNLE or p.maxSolutions == 1:
            #pass
            astnlh = argsort(NN)
            an = an[astnlh]
            
#        print(an[0].nlhc, an[0].tnlh_curr_best)
        # Changes
#        if NN.size != 0:
#            ind = searchsorted(NN, an[0].tnlh_curr_best+1)
#            tmp1, tmp2 = an[:ind], an[ind:]
#            arr = [node.key for node in tmp1]
#            Ind = argsort(arr)
#            an = hstack((tmp1[Ind], tmp2))
        #print [node.tnlh_curr_best for node in an[:10]]
    
    else: #if p.solver.dataHandling == 'sorted':
        if isSNLE and p.maxSolutions != 1: 
            an = hstack((nodes, _in))
        else:
            nodes.sort(key = lambda obj: obj.key)

            if len(_in) == 0:
                an = nodes
            else:
                arr1 = [node.key for node in _in]
                arr2 = [node.key for node in nodes]
                r10 = searchsorted(arr1, arr2)
                an = insert(_in, r10, nodes)
#                if p.debug:
#                    arr = array([node.key for node in an])
#                    #print arr[0]
#                    assert all(arr[1:]>= arr[:-1])

    if maxSolutions != 1:
        Solutions = r46(o, a, PointCoords, PointVals, fTol, varTols, Solutions)
        
        p._nObtainedSolutions = len(solutions)
        if p._nObtainedSolutions > maxSolutions:
            solutions = solutions[:maxSolutions]
            p.istop = 0
            p.msg = 'user-defined maximal number of solutions (p.maxSolutions = %d) has been exeeded' % p.maxSolutions
            return an, g, fo, None, Solutions, xRecord, r41, r40
    
    #p.iterfcn(xk, Min)
    p.iterfcn(xRecord, r40)
    if p.istop != 0: 
        return an, g, fo, None, Solutions, xRecord, r41, r40
    if isSNLE and maxSolutions == 1 and Min <= fTol:
        # TODO: rework it for nonlinear systems with non-bound constraints
        p.istop, p.msg = 1000, 'required solution has been obtained'
        return an, g, fo, None, Solutions, xRecord, r41, r40
    
    an, g = func9(an, fo, g, p)

    nn = maxNodes#1 if asdf1.isUncycled and all(isfinite(o)) and p._isOnlyBoxBounded and not p.probType.startswith('MI') else maxNodes

    an, g = func5(an, nn, g, p)
    nNodes.append(len(an))

    return an, g, fo, _s, Solutions, xRecord, r41, r40


def r46(o, a, PointCoords, PointVals, fTol, varTols, Solutions):
    solutions, coords = Solutions.solutions, Solutions.coords
    #n = o.shape[1] / 2
    
    #L1, L2 = o[:, :n], o[:, n:]
    #omin = where(logical_or(L1 > L2, isnan(L1)), L2, L1)
    #r5Ind =  where(logical_and(PointVals < fTol, nanmax(omin, 1) == 0.0))[0]
    
    r5Ind =  where(PointVals < fTol)[0]

    r5 = PointCoords[r5Ind]
    
    for c in r5:
        if len(solutions) == 0 or not any(all(abs(c - coords) < varTols, 1)): 
            solutions.append(c)
            #coords = asarray(solutions)
            Solutions.coords = append(Solutions.coords, c.reshape(1, -1), 0)
            
    return Solutions


def r45(y, e, vv, p, asdf1, dataType, r41, nlhc):
    o, a, definiteRange = func82(y, e, vv, asdf1, dataType, p, r41)
    
    if p.debug and any(a + 1e-15 < o):  
        p.warn('interval lower bound exceeds upper bound, it seems to be FuncDesigner kernel bug')
    if p.debug and any(logical_xor(isnan(o), isnan(a))):
        p.err('bug in FuncDesigner intervals engine')
    
    m, n = e.shape
    o, a = o.reshape(2*n, m).T, a.reshape(2*n, m).T

    if p.probType not in ('SNLE', 'NLSP') and asdf1.isUncycled and not p.probType.startswith('MI') \
    and len(p._discreteVarsList)==0:# for SNLE fo = 0
        # TODO: 
        # handle constraints with restricted domain and matrix definiteRange
        
        if all(definiteRange):
            # TODO: if o has at least one -inf => prob is unbounded
            tmp1 = o[nlhc==0] if nlhc is not None else o
            if tmp1.size != 0:
                tmp1 = nanmin(tmp1)
                
                ## to prevent roundoff issues ##
                tmp1 += 1e-14*abs(tmp1)
                if tmp1 == 0: tmp1 = 1e-300 
                ######################
                
                r41 = nanmin((r41, tmp1)) 
    else:
        pass
        
    return o, a, r41

def updateNodes(nodesToUpdate, fo):
    if len(nodesToUpdate) == 0: return
    a_tmp = array([node.a for node in nodesToUpdate])
    Tmp = a_tmp
    Tmp[Tmp>fo] = fo                

    o_tmp = array([node.o for node in nodesToUpdate])
    Tmp -= o_tmp
    Tmp[Tmp<1e-300] = 1e-300
    Tmp[o_tmp>fo] = nan
    tnlh_all_new =  - log2(Tmp)
    
    del Tmp, a_tmp
    
    tnlh_all_new += vstack([node.tnlhf for node in nodesToUpdate])#tnlh_fixed[ind_update]
    
    tnlh_curr_best = nanmin(tnlh_all_new, 1)

    o_tmp[o_tmp > fo] = -inf
    M = atleast_1d(nanmax(o_tmp, 1))
    for j, node in enumerate(nodesToUpdate): 
        node.fo = fo
        node.tnlh_curr = tnlh_all_new[j]
        node.tnlh_curr_best = tnlh_curr_best[j]
        node.th_key = M[j]

#    return tnlh_all_new, tnlh_curr_best, M


#from multiprocessing import Pool
#from numpy import array_split
#def updateNodes(nodesToUpdate, fo, p):
#    if p.nProc == 1:
#        Chunks = [nodesToUpdate]
#        result = [updateNodesEngine((nodesToUpdate, fo))]
#    else:
#        Chunks = array_split(nodesToUpdate, p.nProc)
#        if not hasattr(p, 'pool'):
#            p.pool = Pool(processes = p.nProc)
#        #result = p.pool.imap(updateNodesEngine, [(c, fo) for c in Chunks])
#        result = p.pool.map(updateNodesEngine, [(c, fo) for c in Chunks])
#    for i, elem in enumerate(result):
#        if elem is None: continue
#        tnlh_all_new, tnlh_curr_best, M = elem
#        for j, node in enumerate(Chunks[i]): 
#            node.fo = fo
#            node.tnlh_curr = tnlh_all_new[j]
#            node.tnlh_curr_best = tnlh_curr_best[j]
#            node.th_key = M[j]
