from numpy import dot, zeros, float64, diag, ones, ndarray, isfinite, all, asarray
from numpy.linalg import norm

class Dilation():
    def __init__(self, p): 
        self.T = float64
#        if hasattr(numpy, 'float128'):
#            pass
#            self.T = numpy.float128        
        self.b = diag(ones(p.n, dtype = self.T))
        
    def getDilatedVector(self, vec):
        assert type(vec) == ndarray and all(isfinite(vec))
        tmp = dot(self.b.T, vec)
        if any(tmp): tmp /= norm(tmp)
        return dot(self.b, tmp)
    
    def updateDilationMatrix(self, vec, alp=2.0):
        assert type(vec) == ndarray and all(isfinite(vec))
        g = dot(self.b.T, vec)
        ng = norm(g)

#        if self.needRej(p, b, g1, g) or selfNeedRej:
#            selfNeedRej = False
#            if self.showRej or p.debug:
#                p.info('debug msg: matrix B restoration in ralg solver')
#            b = B0.copy()
#            hs = p.norm(prevIter_best_ls_point.x - best_ls_point.x)
#            # TODO: iterPoint = projection(iterPoint,Aeq) if res_Aeq > 0.75*contol

#        if ng < 1e-40: 
#            hs *= 0.9
#            p.debugmsg('small dilation direction norm (%e), skipping' % ng)

        if all(isfinite(g)) and ng > 1e-50:
            g = (g / ng).reshape(-1,1)
            vec1 = dot(self.b, g)
            w = asarray(1.0/alp-1.0, self.T) 
#            w = asarray(1.0/(alp+alp_addition)-1.0, T) 
            vec2 = w * g.T
            self.b += dot(vec1, vec2)
            
