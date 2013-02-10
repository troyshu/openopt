from interalgLLR import *
from numpy import inf, prod, all, sum


def r14IP(p, nlhc, residual, definiteRange, y, e, vv, asdf1, C, CBKPMV, g, nNodes,  \
         frc, fTol, Solutions, varTols, _in, dataType, \
         maxNodes, _s, indTC, xRecord):

    required_sigma = p.ftol * 0.99 # to suppress roundoff effects
    
    m, n = y.shape
    
    if 1:
        ip = func10(y, e, vv)
        ip.dictOfFixedFuncs = p.dictOfFixedFuncs
        o, a, definiteRange = func8(ip, asdf1, dataType)
    else:
        vv = p._freeVarsList
        o, a, definiteRange = func82(y, e, vv, asdf1, dataType, p)
    
    if not all(definiteRange):
        p.err('''
        numerical integration with interalg is implemented 
        for definite (real) range only, no NaN values in integrand are allowed''')

    o, a = o.reshape(2*n, m).T, a.reshape(2*n, m).T

    nodes = func11(y, e, None, indTC, None, o, a, _s, p)

    #OLD
#    nodes.sort(key = lambda obj: obj.key)
#    #nodes.sort(key = lambda obj: obj.volumeResidual, reverse=True)
#
#    if len(_in) == 0:
#        an = nodes
#    else:
#        arr1 = [node.key for node in _in]
#        arr2 = [node.key for node in nodes]
##        arr1 = -array([node.volumeResidual for node in _in])
##        arr2 = -array([node.volumeResidual for node in nodes])
#        
#        r10 = searchsorted(arr1, arr2)
#        an = insert(_in, r10, nodes)
        
    #NEW
    #nodes.sort(key = lambda obj: obj.key)
    #nodes.sort(key = lambda obj: obj.volumeResidual, reverse=True)

    if len(_in) == 0:
        an = nodes
    else:
        an = hstack((_in, nodes)).tolist()
    if 1: 
        an.sort(key = lambda obj: obj.key, reverse=False)
        #an.sort(key = lambda obj: obj.minres, reverse=False)
    else:
        an.sort(key=lambda obj: obj.volumeResidual, reverse=False)

    ao_diff = array([node.key for node in an])
    volumes = array([node.volume for node in an])
    
    if 1:
        r10 = ao_diff <= 0.5*(required_sigma-p._residual) / (prod(p.ub-p.lb) - p._volume)
        #r10 = nanmax(a-o, 1) <= required_sigma / prod(p.ub-p.lb)
        
        ind = where(r10)[0]
        # TODO: use true_sum
        #print sum(array([an[i].F for i in ind]) * array([an[i].volume for i in ind]))
        #print 'p._F:', p._F, 'delta:', sum(array([an[i].F for i in ind]) * array([an[i].volume for i in ind]))
        v = volumes[ind]
        p._F += sum(array([an[i].F for i in ind]) * v)
        residuals = ao_diff[ind] * v
        p._residual += residuals.sum()
        p._volume += v.sum()
#        print '1: %e' %  v.sum()
        #volume_0 = p._volume
        #print 'iter:', p.iter, 'nNodes:', len(an), 'F:', p._F, 'div:', ao_diff / (required_sigma / prod(p.ub-p.lb))
        an = asarray(an, object)
        an = an[where(logical_not(r10))[0]]
        
        # changes
        if 0:
            minres = array([node.minres for node in an])
            r10 = minres <= 0.5*(required_sigma-p._residual) / (prod(p.ub-p.lb) - p._volume)
            ind = where(r10)[0]
            for i in ind:
                node = an[i]
                minres_ind = node.minres_ind
                minres = node.minres
                node_volume = prod(node.e-node.y) # TODO: get rid of cycle here
                minres_residual = node.a[minres_ind] - node.o[minres_ind]
                minres_F = 0.5*(node.a[minres_ind] + node.o[minres_ind])
                p._residual += 0.5 * minres_residual * node_volume
                p._F += 0.5 * minres_F * node_volume
                p._volume += 0.5 * node_volume
                node.volume *= 0.5
                node.minres = inf
                if minres_ind < n:
                    node.y[minres_ind] += 0.5 * (node.e[minres_ind] - node.y[minres_ind])
                    #node.key = min((node.key, node.a[minres_ind-n] - node.o[minres_ind-n]))
                else:
                    node.e[minres_ind-n] -= 0.5 * (node.e[minres_ind-n] - node.y[minres_ind-n])
                    #node.key = min((node.key, node.a[minres_ind] - node.o[minres_ind]))
                node.minres = node.complementary_minres
    #        if ind.size != 0 : print '2: %e' % (p._volume - volume_0)
            # changes end
    else:
        residuals = ao_diff * volumes
        p._residual = 0.5*sum(residuals) 
        #print 'sum(residuals): ',  sum(residuals) 
        if p._residual < required_sigma:
            p._F = sum(array([node.F for node in an]) * v)
            an = []
    #p._Residual = p._residual + sum(asarray([node.volume for node in an]) * asarray([node.key for node in an]))
    nNodes.append(len(an))
   
    p.iterfcn(xk=array(nan), fk=p._F, rk = 0)#TODO: change rk to something like p._r0 - p._residual
    if p.istop != 0: 
        ao_diff = array([node.key for node in an])
        volumes = array([node.volume for node in an])
        p._residual += sum(ao_diff * volumes)
        _s = None
 
    #an, g = func9(an, fo, g, 'IP')
    #nn = 1 if asdf1.isUncycled and all(isfinite(a)) and all(isfinite(o)) and p._isOnlyBoxBounded else maxNodes
    #an, g = func5(an, nn, g)

    return an, g, inf, _s, Solutions, xRecord, frc, CBKPMV
