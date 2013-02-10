from numpy import diag, array, sqrt,  eye, ones, inf, any, copy, zeros, dot, where, all, sum, isfinite, float64, isnan, log10, \
max, sign, array_equal, logical_and, matrix
from openopt.kernel.ooMisc import norm

try:
    from numpy.linalg import solve, LinAlgError
except ImportError:
    LinAlgError = Exception
    def solve(*args, **kw): 
        print('ralg with equality constraints is unimplemented yet')
        raise Exception('ralg with equality constraints is unimplemented yet')
from openopt.kernel.nonOptMisc import scipyAbsentMsg, scipyInstalled, isPyPy
import openopt

#try:
#    from numpy.linalg import cond
#except:
#    print 'warning: no cond in numpy.linalg, matrix B rejuvenation check will be omitted'
#    cond = lambda Matrix: 1

from openopt.kernel.baseSolver import *
#from openopt.kernel.Point import Point
from openopt.kernel.ooMisc import economyMult, Len
from openopt.kernel.setDefaultIterFuncs import *
from openopt.solvers.UkrOpt.UkrOptMisc import getBestPointAfterTurn

class ralg(baseSolver):
    __name__ = 'ralg'
    __license__ = "BSD"
    __authors__ = "Dmitrey"
    __alg__ = "Naum Z. Shor R-algorithm with adaptive space dilation & some modifications"
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub', 'c', 'h']
    iterfcnConnected = True
    _canHandleScipySparse = True

    #ralg default parameters
    B = None # if provided it should be square matrix of shape nVars x nVars
    alp, h0, nh, q1, q2  = 2.0, 1.0, 3, 'default:0.9 for NLP, 1.0 for NSP', 1.1
    hmult = 0.5
    S = 0
    T = float64
    dilationType = 'plain difference'

    showLS = False
    show_hs = False
    showRej = False
    showRes = False
    show_nnan = False
    doBackwardSearch = True
    approach = 'all active'
    newLinEq = True
    new_bs = True
    skipPrevIterNaNsInDilation = True
    innerState = None
    penalties = False # True means for nonlinear equalities only
    #new_s = False

    def needRej(self, p, b, g, g_dilated):
#        r = log10(1e15 * p.norm(g_dilated) / p.norm(g))
#        if isfinite(r):
#            p.debugmsg('%d' % int(r))
        #p.debugmsg('%0.2g'%(p.norm(g_dilated) / p.norm(g)))
        return 1e14 * p.norm(g_dilated) < p.norm(g)
    #checkTurnByGradient = True

    def __init__(self): pass
    def __solver__(self, p):

        alp, h0, nh, q1, q2 = self.alp, self.h0, self.nh, self.q1, self.q2
        
        if isPyPy:
            if p.nc != 0 or p.nh != 0:
                p.warn("in PyPy ralg may work incorrectly with nonlinear constraints yet")
            if p.nbeq != 0 or any(p.lb==p.ub):
                p.err('in PyPy ralg cannot handle linear equality constraints yet')
        
        if type(q1) == str:
            if p.probType== 'NLP' and p.isUC: q1 = 0.9
            else: q1 = 1.0
        T = self.T
        # alternatively instead of alp=self.alp etc you can use directly self.alp etc

        n = p.n
        x0 = p.x0
        
        if p.nbeq == 0 or any(abs(p._get_AeqX_eq_Beq_residuals(x0))>p.contol): # TODO: add "or Aeqconstraints(x0) out of contol"
            x0[x0<p.lb] = p.lb[x0<p.lb]
            x0[x0>p.ub] = p.ub[x0>p.ub]
        
        ind_box_eq = where(p.lb==p.ub)[0]
        nEQ = ind_box_eq.size
        if nEQ != 0:
            initLenBeq = p.nbeq
            Aeq, beq, nbeq = copy(p.Aeq), copy(p.beq), p.nbeq
            p.Aeq = zeros([Len(p.beq) + nEQ, p.n])
            p.beq = zeros(Len(p.beq) + nEQ)
            p.beq[:Len(beq)] = beq
            p.Aeq[:Len(beq)] = Aeq
            for i in range(len(ind_box_eq)):
                p.Aeq[initLenBeq+i, ind_box_eq[i]] = 1
                p.beq[initLenBeq+i] = p.lb[ind_box_eq[i]] # = p.ub[indEQ[i]], because they are the same
            p.nbeq += nEQ
            
        if not self.newLinEq or p.nbeq == 0:
            needProjection = False
            B0 = eye(n,  dtype=T)
            restoreProb = lambda *args: 0
            Aeq_r, beq_r, nbeq_r = None, None, 0
        else:
            needProjection = True
            B0 = self.getPrimevalDilationMatrixWRTlinEqConstraints(p)
            #Aeq, beq, nbeq = p.Aeq, p.beq, p.nbeq
            
            if any(abs(p._get_AeqX_eq_Beq_residuals(x0))>p.contol/16.0):
                #p.debugmsg('old point Aeq residual:'+str(norm(dot(Aeq, x0)-beq)))
                try:
                    x0 = self.linEqProjection(x0, p.Aeq, p.beq)
                except LinAlgError:
                    s = 'Failed to obtain projection of start point to linear equality constraints subspace, probably the system is infeasible'
                    p.istop, p.msg = -25,  s
                    return
                    
                #p.debugmsg('new point Aeq residual:'+str(norm(dot(Aeq, x0)-beq)))
            if nEQ == 0:
                Aeq_r, beq_r, nbeq_r = p.Aeq, p.beq, p.nbeq
            else:
                Aeq_r, beq_r, nbeq_r = Aeq, beq, nbeq
            
            p.Aeq, p.beq, p.nbeq = None, None, 0
            
            # TODO: return prob with unmodified Aeq, beq
            
            def restoreProb():
                p.Aeq, p.beq, p.nbeq = Aeq_r, beq_r, nbeq_r
                #if nEQ != 0: restore lb, ub
                    
            
        b = B0.copy() if self.B is None else self.B
#        B_f = diag(ones(n))
#        B_constr = diag(ones(n))
        hs = asarray(h0, T)
        
        if self.innerState is not None:
            hs = self.innerState['hs']
            b = self.innerState['B']
        
        ls_arr = []
        w = asarray(1.0/alp-1.0, T)

        """                            Shor r-alg engine                           """
        bestPoint = p.point(array(copy(x0).tolist(), T)) # tolist() for PyPy compatibility
        prevIter_best_ls_point = bestPoint
        prevIter_PointForDilation = bestPoint

        g = bestPoint._getDirection(self.approach)
        prevDirectionForDilation = g
        moveDirection = g
        if not any(g) and all(isfinite(g)):
            # TODO: create ENUMs
            p.iterfcn(bestPoint)
            restoreProb()
            p.istop = 14 if bestPoint.isFeas(False) else -14
            p.msg = 'move direction has all-zero coords'
            return

        HS = []
        LS = []
        
        SwitchEncountered = False
        selfNeedRej = False
        doScale = False
        
        #directionVectorsList = []
#        #pass-by-ref! not copy!
#        if p.isFeas(p.x0): b = B_f
#        else: b = B_constr

#        if p.debug and hasattr(p, 'x_opt'):
#            import scipy
#            exactDirection = x0-p.x_opt
#            asdf_0 = exactDirection * (0.2+scipy.rand(n))
#            #asdf = asdf_0.copy()


        fTol = p.fTol if p.fTol is not None else 15*p.ftol
        
        # CHANGES
        if self.penalties:
            oldVal = p.f(p.x0)
            newVal = inf
            x = p.x0
            
            #H,  DH = p.h, p.dh
            if p.nh != 0:
                #S = 1.0
            
                _Aeq = p.dh(x)
                _beq = -p.h(x)
                df = p.df(x)
                if n>=150 and not scipyInstalled:
                    p.pWarn(scipyAbsentMsg)
                if n>100 and scipyInstalled:
                    from scipy.sparse import eye as Eye # to prevent numpy.eye overwrite
                    HH = Eye(n, n)
                else:
                    HH = eye(n)
                qp = openopt.QP(H=HH, f=df, Aeq=_Aeq, beq=_beq)
    #                print ('len(_beq): %d' % len(_beq))
    #                assert len(_beq) != 0
                QPsolver = openopt.oosolver('cvxopt_qp', iprint=-1)
                if not QPsolver.isInstalled:
                    #p.pWarn('to use ')
                    S = None
                else:
                    r = qp.solve(QPsolver)
                    #S = 2.0*abs(r.duals).sum() if r.istop > 0 else 0
                    S = 10.0*sum(abs(r.duals)) if r.istop > 0 else None
                
                while any(p.h(x)) > p.contol:
                    if S is not None:
                        p2 = getattr(openopt, p.probType)(p.f, x)
                        p.inspire(p2)
                        p2.x0 = x
                        p2.h = p2.dh = None
                        p2.userProvided.h = p2.userProvided.dh = False
                        p2.nh = 0
                        p2.f = lambda *args, **kwargs: p.f(*args, **kwargs) + sum(abs(S * p.h(*args, **kwargs)))
                        p2.df = lambda *args, **kwargs: p.df(*args, **kwargs) + dot(S * sign(p.h(*args, **kwargs)), p.dh(*args, **kwargs))
                        #p2.iterfcn = p.iterfcn
    #                    def df2(*args, **kwargs):
    #                        r1 = p.df(*args, **kwargs)
    #                        r2 = S * dot(p.dh(*args, **kwargs).reshape(-1, 1), sign(p.h(*args, **kwargs))).flatten()
    #                        #raise 0
    #                        return r1+r2
    #                    #p2.df = lambda *args, **kwargs: p.df(*args, **kwargs) + S * dot(p.dh(x).reshape(-1, 1), sign(p.h(*args, **kwargs))).flatten()
    #                    p2.df = df2
    #                    #raise 0
                        r2 = p2.solve(p.solver, iprint=10)
                        if r2.stopcase >= 0:
                            x = r2.xf
                            p.solver.innerState = r2.extras['innerState']
                            oldVal, newVal = newVal, r2.ff
                        else:
                            if r2.istop == IS_LINE_SEARCH_FAILED:
                                # TODO: custom S as raising penalties
                                pass
                        
                        if p.isFeas(p2.xk):
                            p.xf = p.xk = p2.xk
                            p.istop, p.msg = p2.istop, p2.msg
                            return
                        else:
                            S *= 50
                            #print('max residual:%0.2e'% r2.rf)
                        
                    else: # failed to solve QP
                        break
                    
        #print 'b:', b, '\nhs:', hs
        # CHANGES END

        """                           Ralg main cycle                                    """

        for itn in range(p.maxIter+10):
            doDilation = True
            lastPointOfSameType = None # to prevent possible bugs
            alp_addition = 0.0
            
            iterStartPoint = prevIter_best_ls_point
            x = iterStartPoint.x.copy()

            g_tmp = economyMult(b.T, moveDirection)
            if any(g_tmp): g_tmp /= p.norm(g_tmp)
            g1 = p.matmult(b, g_tmp)
            
#            norm_moveDirection = p.norm(g1)
#            if doScale:
#                g1 *= (norm_moveDirection_prev/norm_moveDirection) ** 0.5
#            norm_moveDirection_prev = norm_moveDirection

#            if p.debug and hasattr(p, 'x_opt'):
#                cos_phi_0 = p.matmult(moveDirection,  prevIter_best_ls_point.x - p.x_opt)/p.norm(moveDirection)/p.norm(prevIter_best_ls_point.x - p.x_opt)
#                cos_phi_1 = p.matmult(g1,  prevIter_best_ls_point.x - p.x_opt)/p.norm(g1)/p.norm(prevIter_best_ls_point.x - p.x_opt)
#                print('beforeDilation: %f  afterDilation: %f' % (cos_phi_0, cos_phi_1) )
#                asdf = asdf_0.copy()
#                g_tmp = economyMult(b.T, asdf)
#                
#                #g_tmp = p.matmult(b.T, asdf)
#                
#                if any(g_tmp): g_tmp /= p.norm(g_tmp)
#                asdf = p.matmult(b, g_tmp)
#                cos_phi = dot(asdf, exactDirection) / p.norm(asdf) / p.norm(exactDirection)
#                p.debugmsg('cos_phi:%f' % cos_phi)
#                assert cos_phi >0


            """                           Forward line search                          """

            hs_cumsum = 0
            hs_start = hs
            for ls in range(p.maxLineSearch):
                hs_mult = 1.0
                if ls > 20:
                    hs_mult = 2.0
                elif ls > 10:
                    hs_mult = 1.5
                elif ls > 2:
                    hs_mult = 1.05
                hs *= hs_mult

                x -= hs * g1
                hs_cumsum += hs

                newPoint = p.point(x) if ls == 0 else iterStartPoint.linePoint(hs_cumsum/(hs_cumsum-hs), oldPoint) #  TODO: take ls into account?
                
                if not p.isUC:
                    if newPoint.isFeas(True) == iterStartPoint.isFeas(True):
                        lastPointOfSameType = newPoint
              
                if self.show_nnan: p.info('ls: %d nnan: %d' % (ls, newPoint.__nnan__()))

                
                if ls == 0:
                    oldPoint = prevIter_best_ls_point#prevIterPoint
                    oldoldPoint = oldPoint
                    
                #if not self.checkTurnByGradient:
                if newPoint.betterThan(oldPoint, altLinInEq=True):
                    if newPoint.betterThan(bestPoint, altLinInEq=False): bestPoint = newPoint
                    oldoldPoint = oldPoint
                    oldPoint, newPoint = newPoint,  None
                else:
                    if not itn % 4: 
                        for fn in ['_lin_ineq', '_lin_eq']:
                            if hasattr(newPoint, fn): delattr(newPoint, fn)
                    break
                    
            hs /= hs_mult
            
            if ls == p.maxLineSearch-1:
                p.istop,  p.msg = IS_LINE_SEARCH_FAILED,  'maxLineSearch (' + str(p.maxLineSearch) + ') has been exceeded, the problem seems to be unbounded'
                restoreProb()
                return

            #iterPoint  = newPoint
            PointForDilation = newPoint
            #best_ls_point = newPoint if ls == 0 else oldPoint
            #if p.debug and ls != 0: assert not oldPoint.betterThan(best_ls_point)

            """                          Backward line search                          """
            mdx = max((150, 1.5*p.n))*p.xtol
            if itn == 0:  mdx = max((hs / 128.0, 128*p.xtol )) # TODO: set it after B rej as well
            ls_backward = 0
            maxLS = 3 if ls == 0 else 1
#            if ls <=3 or ls > 20:
            if self.doBackwardSearch:
                if self.new_bs:
                    best_ls_point,  PointForDilation, ls_backward = \
                    getBestPointAfterTurn(oldoldPoint, newPoint, maxLS = maxLS, maxDeltaF = 150*p.ftol, \
                                          maxDeltaX = mdx, altLinInEq = True, new_bs = True)
                    if PointForDilation.isFeas(True) == iterStartPoint.isFeas(True):
                        lastPointOfSameType = PointForDilation
#                        elif best_ls_point.isFeas(altLinInEq=True) == iterStartPoint.isFeas(altLinInEq=True):
#                            lastPointOfSameType = best_ls_point
                else:
                    best_ls_point, ls_backward = \
                    getBestPointAfterTurn(oldoldPoint, newPoint, maxLS = maxLS, altLinInEq = True, new_bs = False)
                    PointForDilation = best_ls_point

                # TODO: extract last point from backward search, that one is better than iterPoint
                if best_ls_point.betterThan(bestPoint): bestPoint = best_ls_point
                #p.debugmsg('ls_backward:%d' % ls_backward)
                if ls == 0 and ls_backward == -maxLS:
                    #pass
                    alp_addition += 0.25
                    #hs *= 0.9
                
                if ls_backward <= -1 and itn != 0:  # TODO: mb use -1 or 0 instead?
                    pass
                    #alp_addition -= 0.25*ls_backward # ls_backward less than zero
                
                #hs *= 2 ** min((ls_backward+1, 0))
            else:
                pass
                #hs *= 0.95
            
            best_ls_point = PointForDilation # elseware lots of difficulties
            
            """                                 Updating hs                                 """
            step_x = p.norm(PointForDilation.x - prevIter_PointForDilation.x)
            step_f = abs(PointForDilation.f() - prevIter_PointForDilation.f())
            HS.append(hs_start)
            assert ls >= 0
            LS.append(ls)
            if itn > 3:
                mean_ls = (3*LS[-1] + 2*LS[-2]+LS[-3]) / 6.0
                j0 = 3.3
                if mean_ls > j0:
                    hs = (mean_ls - j0 + 1)**0.5 * hs_start
                else:
                    #hs = (ls/j0) ** 0.5 * hs_start
                    hs = hs_start
                    if ls == 0 and ls_backward == -maxLS:
                        shift_x = step_x / p.xtol
                        RD = log10(shift_x+1e-100)
                        if PointForDilation.isFeas(True) or prevIter_PointForDilation.isFeas(True):
                            RD = min((RD, asscalar(asarray(log10(step_f / p.ftol + 1e-100)))))
                        if RD > 1.0:
                            mp = (0.5, (ls/j0) ** 0.5, 1 - 0.2*RD)
                            hs *= max(mp)
                            #from numpy import argmax
                            #print argmax(mp), mp

            """                            Handling iterPoints                            """
               
            best_ls_point = PointForDilation
            
            #if not SwitchEncountered and p.nh != 0 and PointForDilation.isFeas(altLinInEq=False) != prevIter_PointForDilation.isFeas(altLinInEq=False):
                #SwitchEncountered = True
                #selfNeedRej = True
            
            involve_lastPointOfSameType = False
            if lastPointOfSameType is not None and PointForDilation.isFeas(True) != prevIter_PointForDilation.isFeas(True):
                # TODO: add middle point for the case ls = 0
                assert self.dilationType == 'plain difference'
                #directionForDilation = lastPointOfSameType._getDirection(self.approach) 
                PointForDilation = lastPointOfSameType
                involve_lastPointOfSameType = True
                
            
           
           
            #directionForDilation = newPoint.__getDirection__(self.approach) # used for dilation direction obtaining
            
#            if not self.new_bs or ls != 0:
#                moveDirection = iterPoint.__getDirection__(self.approach)
#            else:
#                moveDirection = best_ls_point.__getDirection__(self.approach)
                
                #directionForDilation = pointForDilation.__getDirection__(self.approach) 
                
                
#                cos_phi = -p.matmult(moveDirection, prevIterPoint.__getDirection__(self.approach))
#                assert cos_phi.size == 1
#                if cos_phi> 0:
#                    g2 = moveDirection#pointForDilation.__getDirection__(self.approach) 
#                else:
#                    g2 = pointForDilation.__getDirection__(self.approach) 
                
            if itn == 0:
                p.debugmsg('hs: ' + str(hs))
                p.debugmsg('ls: ' + str(ls))
            if self.showLS: p.info('ls: ' + str(ls))
            if self.show_hs: p.info('hs: ' + str(hs))
            if self.show_nnan: p.info('nnan: ' + str(best_ls_point.__nnan__()))
            if self.showRes:
                r, fname, ind = best_ls_point.mr(True)
                p.info(fname+str(ind))

            """                         Set dilation direction                            """

            #if sum(p.dotmult(g, g2))>0:
                #p.debugmsg('ralg warning: slope angle less than pi/2. Mb dilation for the iter will be omitted.')
                #doDilation = False



                    
            # CHANGES
#            if lastPointOfSameType is None:
#                if currIterPointIsFeasible and not prevIterPointIsFeasible:
#                    alp_addition += 0.1
#                elif prevIterPointIsFeasible and not currIterPointIsFeasible:
#                    alp_addition -= 0.0
                
            # CHANGES END
            
#            r_p, ind_p, fname_p = prevIter_best_ls_point.mr(1)
#            r_, ind_, fname_ = PointForDilation.mr(1)


            #else:
            

            #print itn,'>>>>>>>>>', currIterPointIsFeasible
            
            """                                    Excluding derivatives switched to/from NaN                                    """
            
            if self.skipPrevIterNaNsInDilation:
                c_prev, c_current = prevIter_PointForDilation.c(), PointForDilation.c()
                h_prev, h_current = prevIter_PointForDilation.h(), PointForDilation.h()
                
            """                                             Handling switch to NaN                                            """
            NaN_derivatives_excluded = False
            if self.skipPrevIterNaNsInDilation:
                assert self.approach == 'all active'
                
                if not prevIter_PointForDilation.isFeas(True):
                    """                          processing NaNs in nonlin inequality constraints                          """
                    ind_switch_ineq_to_nan = where(logical_and(isnan(c_current), c_prev>0))[0]              
                    if len(ind_switch_ineq_to_nan) != 0:
                        NaN_derivatives_excluded = True
                        tmp = prevIter_PointForDilation.dc(ind_switch_ineq_to_nan)
                        if hasattr(tmp, 'toarray'):
                            tmp = tmp.A
                        if len(ind_switch_ineq_to_nan)>1:
                            tmp *= (c_prev[ind_switch_ineq_to_nan] /sqrt((tmp**2).sum(1))).reshape(-1, 1)
                        else:
                            tmp *= c_prev[ind_switch_ineq_to_nan] / norm(tmp)
                        if tmp.ndim>1: tmp = tmp.sum(0)
                        if not isinstance(tmp, ndarray) or isinstance(tmp, matrix): tmp = tmp.A.flatten() # dense or sparse matrix
                        #print '1: excluded:', norm(tmp), norm(prevDirectionForDilation)
                        prevDirectionForDilation -= tmp
                        #print '1: result=', norm(prevDirectionForDilation)
                        
                    """                           processing NaNs in nonlin equality constraints                           """
                    ind_switch_eq_to_nan = where(logical_and(isnan(h_current), h_prev>0))[0]       
                    if len(ind_switch_eq_to_nan) != 0:
                        NaN_derivatives_excluded = True
                        tmp = prevIter_PointForDilation.dh(ind_switch_eq_to_nan)                        
                        if tmp.ndim>1: tmp = tmp.sum(0)
                        if not isinstance(tmp, ndarray) or isinstance(tmp, matrix): tmp = tmp.A.flatten() # dense or sparse matrix
                        prevDirectionForDilation -= tmp

                    ind_switch_eq_to_nan = where(logical_and(isnan(h_current), h_prev<0))[0]                
                    if len(ind_switch_eq_to_nan) != 0:
                        NaN_derivatives_excluded = True
                        tmp = prevIter_PointForDilation.dh(ind_switch_eq_to_nan)
                        if tmp.ndim>1: tmp = tmp.sum(0)
                        if not isinstance(tmp, ndarray) or isinstance(tmp, matrix): tmp = tmp.A.flatten() # dense or sparse matrix
                        prevDirectionForDilation += tmp
                
            directionForDilation = PointForDilation._getDirection(self.approach) 
            
            """                                            Handling switch from NaN                                           """
            if self.skipPrevIterNaNsInDilation:
                
                if not PointForDilation.isFeas(True):
                    
                    """                          processing NaNs in nonlin inequality constraints                          """
                    ind_switch_ineq_from_nan = where(logical_and(isnan(c_prev), c_current>0))[0]
                    if len(ind_switch_ineq_from_nan) != 0:
                        NaN_derivatives_excluded = True
                        tmp = PointForDilation.dc(ind_switch_ineq_from_nan)
                        if hasattr(tmp, 'toarray'):
                            tmp = tmp.A                        
                        if len(ind_switch_ineq_from_nan)>1:
                            tmp *= (c_current[ind_switch_ineq_from_nan] /sqrt((tmp**2).sum(1))).reshape(-1, 1)
                        else:
                            tmp *= c_current[ind_switch_ineq_from_nan] / norm(tmp)                        
                        if tmp.ndim>1: tmp = tmp.sum(0)
                        if not isinstance(tmp, ndarray) or isinstance(tmp, matrix): tmp = tmp.A.flatten() # dense or sparse matrix
                        #print '2: excluded:', norm(tmp), norm(directionForDilation)
                        directionForDilation -= tmp
                        #print '2: result=', norm(directionForDilation)
                        
                        
                    """                           processing NaNs in nonlin equality constraints                           """
                    ind_switch_eq_from_nan = where(logical_and(isnan(h_prev), h_current>0))[0]
                    if len(ind_switch_eq_from_nan) != 0:
                        NaN_derivatives_excluded = True
                        tmp = PointForDilation.dh(ind_switch_eq_from_nan)
                        if tmp.ndim>1: tmp = tmp.sum(0)
                        if not isinstance(tmp, ndarray) or isinstance(tmp, matrix): tmp = tmp.A.flatten() # dense or sparse matrix
                        directionForDilation -= tmp

                    ind_switch_eq_from_nan = where(logical_and(isnan(h_prev), h_current<0))[0]
                    if len(ind_switch_eq_from_nan) != 0:
                        NaN_derivatives_excluded = True
                        tmp = PointForDilation.dh(ind_switch_eq_from_nan)
                        if tmp.ndim>1: tmp = tmp.sum(0)
                        if not isinstance(tmp, ndarray) or isinstance(tmp, matrix): tmp = tmp.A.flatten() # dense or sparse matrix
                        directionForDilation += tmp

#            # CHANGES
#            gn = g2/norm(g2)
#            if len(directionVectorsList) == 0 or n < 3: pass
#            else:
#                if len(directionVectorsList) == 1 or abs(dot(directionVectorsList[-1], directionVectorsList[-2]))>0.999:
#                    projectionComponentLenght = abs(dot(directionVectorsList[-1], gn))
#                    restLength = sqrt(1 - min((1, projectionComponentLenght))**2)
#                else: 
#                    e1 = directionVectorsList[-1]
#                    e2 = directionVectorsList[-2] - dot(directionVectorsList[-1], directionVectorsList[-2]) * directionVectorsList[-1]
#                    e2 /= norm(e2)
#                   
#                    proj1, proj2 = dot(e1, gn), dot(e2, gn)
#                    rest = gn - proj1 * e1 - proj2 * e2
#                    restLength = norm(rest)
#                if restLength > 1+1e-5: p.pWarn('possible error in ralg solver: incorrect restLength, exceeds 1.0')
#                
#                # TODO: make it parameters of ralg
#                commonCoeff, alp_add_coeff = 0.5, 1.0
#                
#                if restLength < commonCoeff * (n - 2.0) / n:
#                    #pass
#                    alpAddition = 0.5+(arctan((n - 2.0) / (n * restLength)) - pi / 4.0) / (pi / 2.0) * alp_add_coeff
#                    #p.debugmsg('alpAddition:' + str(alpAddition))
#                    assert alpAddition > 0 # if someone incorrectly modifies commonCoeff it can be less than zero
#                    alp_addition += alpAddition
#                    #p.debugmsg('alp_addition:' + str(alp_addition))
#                    
#            directionVectorsList.append(gn)
#            if len(directionVectorsList) > 2: directionVectorsList = directionVectorsList[:-2]
#            # CHANGES END

                
            if self.dilationType == 'normalized' and (not fname_p in ('lb', 'ub', 'lin_eq', 'lin_ineq') \
                                                      or not fname_ in ('lb', 'ub', 'lin_eq', 'lin_ineq')) and (fname_p != fname_  or ind_p != ind_):
                G2,  G = directionForDilation/norm(directionForDilation), prevDirectionForDilation/norm(prevDirectionForDilation)
            else:
                G2,  G = directionForDilation, prevDirectionForDilation            
           
            if prevIter_PointForDilation.isFeas(True) == PointForDilation.isFeas(True):
                g1 = G2 - G
            elif prevIter_PointForDilation.isFeas(True):
                g1 = G2.copy()
            else:
                g1 = G.copy()
                alp_addition += 0.05
                
            #print p.getMaxResidual(PointForDilation.x, 1)
            ##############################################
            # the case may be occured when 
            #  1) lastPointOfSameType is used 
            # or
            #  2) some NaN from constraints have been excluded
            if norm(G2 - G) < 1e-12 * min((norm(G2), norm(G))) and (involve_lastPointOfSameType or NaN_derivatives_excluded):
                p.debugmsg("ralg: 'last point of same type gradient' is used")
                g1 = G2
            ##############################################


                #g1 = -G.copy() # signum doesn't matter here


            # changes wrt infeas constraints
#            if prevIterPoint.nNaNs() != 0:
#                cp, hp = prevIterPoint.c(), prevIterPoint.h()
#                ind_infeas_cp, ind_infeas_hp = isnan(cp), isnan(hp)
#                
#                c, h = iterPoint.c(), iterPoint.h()
#                ind_infeas_c, ind_infeas_h = isnan(c), isnan(h)
#                
#                ind_goodChange_c = logical_and(ind_infeas_cp,  logical_not(ind_infeas_c))
#                ind_goodChange_h = logical_and(ind_infeas_hp,  logical_not(ind_infeas_h))
#                
#                any_c, any_h = any(ind_goodChange_c), any(ind_goodChange_h)
#                altDilation = zeros(n)
#                if any_c:
#                    altDilation += sum(atleast_2d(iterPoint.dc(where(ind_goodChange_c)[0])), 0)
#                    assert not any(isnan(altDilation))
#                if any_h:
#                    altDilation += sum(atleast_2d(iterPoint.dh(where(ind_goodChange_h)[0])), 0)
#                if any_c or any_h:
#                    #print '!>', altDilation
#                    #g1 = altDilation
#                    pass
            # changes end


            """                             Perform dilation                               """

            # CHANGES
#            g = economyMult(b.T, g1)
#            gn = g/norm(g)
#            if len(directionVectorsList) == 0 or n < 3 or norm(g1) < 1e-20: pass
#            else:
#                if len(directionVectorsList) == 1 or abs(dot(directionVectorsList[-1], directionVectorsList[-2]))>0.999:
#                    projectionComponentLenght = abs(dot(directionVectorsList[-1], gn))
#                    restLength = sqrt(1 - min((1, projectionComponentLenght))**2)
#                else: 
#                    e1 = directionVectorsList[-1]
#                    e2 = directionVectorsList[-2] - dot(directionVectorsList[-1], directionVectorsList[-2]) * directionVectorsList[-1]
#                    print dot(directionVectorsList[-1], directionVectorsList[-2])
#                    e2 /= norm(e2)
#                    proj1, proj2 = dot(e1, gn), dot(e2, gn)
#                    rest = gn - proj1 * e1 - proj2 * e2
#                    restLength = norm(rest)
#                assert restLength < 1+1e-5, 'error in ralg solver: incorrect restLength'
#                
#                # TODO: make it parameters of ralg
#                commonCoeff, alp_add_coeff = 0.5, 1.0
#                
#                if restLength < commonCoeff * (n - 2.0) / n:
#                    #pass
#                    alpAddition = 0.5+(arctan((n - 2.0) / (n * restLength)) - pi / 4.0) / (pi / 2.0) * alp_add_coeff
#                    #p.debugmsg('alpAddition:' + str(alpAddition))
#                    assert alpAddition > 0 # if someone incorrectly modifies commonCoeff it can be less than zero
#                    alp_addition += alpAddition
#                    #p.debugmsg('alp_addition:' + str(alp_addition))
#                    
#            directionVectorsList.append(gn)
#            if len(directionVectorsList) > 2: directionVectorsList = directionVectorsList[:-2]
            # CHANGES END

            if doDilation:
                g = economyMult(b.T, g1)
                ng = p.norm(g)

                if self.needRej(p, b, g1, g) or selfNeedRej:
                    selfNeedRej = False
                    if self.showRej or p.debug:
                        p.info('debug msg: matrix B restoration in ralg solver')
                    b = B0.copy()
                    hs = p.norm(prevIter_best_ls_point.x - best_ls_point.x)
                    # TODO: iterPoint = projection(iterPoint,Aeq) if res_Aeq > 0.75*contol

                if ng < 1e-40: 
                    hs *= 0.9
                    p.debugmsg('small dilation direction norm (%e), skipping' % ng)
                if all(isfinite(g)) and ng > 1e-50 and doDilation:
                    g = (g / ng).reshape(-1,1)
                    vec1 = economyMult(b, g).reshape(-1,1)# TODO: remove economyMult, use dot?
                    #if alp_addition != 0: p.debugmsg('alp_addition:' + str(alp_addition))
                    w = asarray(1.0/(alp+alp_addition)-1.0, T) 
                    vec2 = w * g.T
                    b += p.matmult(vec1, vec2)
            

            """                               Call OO iterfcn                                """
            if hasattr(p, '_df'): delattr(p, '_df')
            if best_ls_point.isFeas(False) and hasattr(best_ls_point, '_df'): 
                p._df = best_ls_point.df().copy()           
                
            p.iterfcn(best_ls_point)
            

            """                             Check stop criteria                           """

            cond_same_point = array_equal(best_ls_point.x, prevIter_best_ls_point.x)
            if cond_same_point and not p.istop:
                p.istop = 14
                p.msg = 'X[k-1] and X[k] are same'
                p.stopdict[SMALL_DELTA_X] = True
                restoreProb()
                self.innerState = {'B': b, 'hs': hs}
                return
            
            s2 = 0
            if p.istop and not p.userStop:
                if p.istop not in p.stopdict: p.stopdict[p.istop] = True # it's actual for converters, TODO: fix it
                if SMALL_DF in p.stopdict:
                    if best_ls_point.isFeas(False): s2 = p.istop
                    p.stopdict.pop(SMALL_DF)
                if SMALL_DELTA_F in p.stopdict:
                    # TODO: implement it more properly
                    if best_ls_point.isFeas(False) and prevIter_best_ls_point.f() != best_ls_point.f(): s2 = p.istop
                    p.stopdict.pop(SMALL_DELTA_F)
                if SMALL_DELTA_X in p.stopdict:
                    if best_ls_point.isFeas(False) or not prevIter_best_ls_point.isFeas(False) or cond_same_point: s2 = p.istop
                    p.stopdict.pop(SMALL_DELTA_X)
#                if s2 and (any(isnan(best_ls_point.c())) or any(isnan(best_ls_point.h()))) \
#                and not p.isNaNInConstraintsAllowed\
#                and not cond_same_point:
#                    s2 = 0
                    
                if not s2 and any(p.stopdict.values()):
                    for key,  val in p.stopdict.items():
                        if val == True:
                            s2 = key
                            break
                p.istop = s2
                
                for key,  val in p.stopdict.items():
                    if key < 0 or key in set([FVAL_IS_ENOUGH, USER_DEMAND_STOP, BUTTON_ENOUGH_HAS_BEEN_PRESSED]):
                        p.iterfcn(bestPoint)
                        self.innerState = {'B': b, 'hs': hs}
                        return
            """                                If stop required                                """
            
            if p.istop:
#                if self.needRej(p, b, g1, g) or not feasiblePointWasEncountered:
#                    b = B0.copy()
#                    hs = max((p.norm(prevIter_best_ls_point.x - best_ls_point.x) , 128*p.xtol))
#                    p.istop = 0
#                else:
                    restoreProb()
                    p.iterfcn(bestPoint)
                    #p.istop, p.msg = istop, msg
                    self.innerState = {'B': b, 'hs': hs}
                    return


            """                Some final things for ralg main cycle                """
#            p.debugmsg('new point Aeq residual:'+str(norm(dot(Aeq, iterPoint.x)-beq)))
#            if needProjection and itn!=0:
#                #pass
#                x2 = self.linEqProjection(iterPoint.x, Aeq, beq)
#                p.debugmsg('norm(delta):' + str(norm(iterPoint.x-x2))) 
#                iterPoint = p.point(x2)
#                p.debugmsg('2: new point Aeq residual:'+str(norm(dot(Aeq, iterPoint.x)-beq)))
            #p.hs.append(hs)
            #g = moveDirection.copy()
            
            #prevDirectionForDilation = directionForDilation

            #iterPoint = None
            #doScale = self.new_s and prevIter_PointForDilation.isFeas(True) !=  best_ls_point.isFeas(True)
            #print doScale
            prevIter_best_ls_point = best_ls_point
            prevIter_PointForDilation = best_ls_point
            prevDirectionForDilation = best_ls_point._getDirection(self.approach)
            moveDirection = best_ls_point._getDirection(self.approach)


    def getPrimevalDilationMatrixWRTlinEqConstraints(self, p):
        n, Aeq, beq = p.n, p.Aeq, p.beq
        nLinEq = len(p.beq)
        ind_fixed = where(p.lb==p.ub)[0]
        arr=ones(n, dtype=self.T)
        arr[ind_fixed] = 0
        b = diag(arr)
        
        if hasattr(Aeq, 'tocsc'):Aeq = Aeq.tocsc()
        
        for i in range(nLinEq):
            vec = Aeq[i]
            #raise 0
            if hasattr(vec, 'toarray'): vec = vec.toarray().flatten()
            g = economyMult(b.T, vec)
            if not any(g): continue
            #ind_nnz = nonzero(g)[0]
            ng = norm(g)
            g = (g / ng).reshape(-1,1)
            
            vec1 = p.matmult(b, g)# TODO: remove economyMult, use dot?
            vec2 = -g.T
            
            b += p.matmult(vec1, vec2)
            
#            if len(ind_nnz) > 0.7 * g.size:
#                b += p.matmult(vec1, vec2)
#            else:
#                ind_nnz1 = nonzero(vec1)[0]
#                ind_nnz2 = nonzero(vec2)[1]
#                r = dot(vec1[ind_nnz1, :], vec2[:, ind_nnz2])
#                if p.debug: 
#                    assert abs(norm(p.matmult(vec1, vec2).flatten()) - norm(r.flatten())) < 1e-5
#                b[ix_(ind_nnz1, ind_nnz2)] += r
        return b
 
    def linEqProjection(self, x, Aeq, beq):
        # TODO: handle case nbeq = 1 ?
        if hasattr(Aeq, 'toarray'):Aeq = Aeq.toarray()
        AeqT = Aeq.T
        AeqAeqT = dot(Aeq, AeqT)
        Aeqx = dot(Aeq, x)
        AeqT_AeqAeqT_inv_Aeqx = dot(AeqT, ravel(solve(AeqAeqT, Aeqx)))
        AeqT_AeqAeqT_inv_beq = dot(AeqT, ravel(solve(AeqAeqT, beq)))
        xf = x - AeqT_AeqAeqT_inv_Aeqx + AeqT_AeqAeqT_inv_beq
        return xf
