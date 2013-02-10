__docformat__ = "restructuredtext en"
from numpy import asfarray, array, asarray, argmax, zeros, isfinite, all, isnan, arange

empty_arr = asfarray([])

try:
    from scipy.sparse import csr_matrix
except:
    pass
    
#def MULT(A, x):
#    if isinstance(A, ndarray):
#        return dot(A, x)
#    else:
#        t2 = csc_matrix(x)
#        if t2.shape[0] != A.shape[1]:
#            if t2.shape[1] == A.shape[1]:
#                t2 = t2.T
#        return A._mul_sparse_matrix(t2).toarray()         


class residuals:
    def __init__(self):
        pass
    def _get_nonLinInEq_residuals(self, x):
        if hasattr(self.userProvided, 'c') and self.userProvided.c: return self.c(x)
        else: return empty_arr.copy()

    def _get_nonLinEq_residuals(self, x):
        if hasattr(self.userProvided, 'h') and self.userProvided.h: return self.h(x)
        else: return empty_arr.copy()

    def _get_AX_Less_B_residuals(self, x):
        if self.A is not None and self.A.size > 0: 
            if x.ndim > 1: # multiarray
                return (self.matmult(self.A, x.T) - self.b.reshape(-1, 1)).T if not hasattr(self, '_A') else \
                (self._A._mul_sparse_matrix(csr_matrix(x.T)).toarray().T - self.b.reshape(-1, 1)).T
            return self.matmult(self.A, x).flatten() - self.b if not hasattr(self, '_A') else \
            self._A._mul_sparse_matrix(csr_matrix((x, (arange(self.n), zeros(self.n))), shape=(self.n, 1))).toarray().flatten() - self.b
            #return self.matmult(self.A, x).flatten() - self.b if not hasattr(self, '_A') else self._A._mul_sparse_matrix(csr_matrix(x).reshape((self.n, 1))).toarray().flatten() - self.b
        else: return empty_arr.copy()

    def _get_AeqX_eq_Beq_residuals(self, x):
        if self.Aeq is not None and self.Aeq.size>0 : 
            if x.ndim > 1: # multiarray
                return (self.matmult(self.Aeq, x.T) - self.beq.reshape(-1, 1)).T if not hasattr(self, '_Aeq') else \
                (self._Aeq._mul_sparse_matrix(csr_matrix(x.T)).toarray().T - self.beq.reshape(-1, 1)).T
            return self.matmult(self.Aeq, x).flatten() - self.beq if not hasattr(self, '_Aeq') else \
            self._Aeq._mul_sparse_matrix(csr_matrix((x, (arange(self.n), zeros(self.n))), shape=(self.n, 1))).toarray().flatten() - self.beq
        else: return empty_arr.copy()

    def _getLbresiduals(self, x):
        return self.lb - x

    def _getUbresiduals(self, x):
        return x - self.ub

    def _getresiduals(self, x):
#        if 'x' in self.prevVal['r'].keys() and all(x == self.prevVal['r']['x']):
#            return self.prevVal['r']['Val']
        # TODO: add quadratic constraints
        r = EmptyClass()
        # TODO: simplify it!
        if self._baseClassName == 'NonLin':
            r.c = self._get_nonLinInEq_residuals(x)
            r.h = self._get_nonLinEq_residuals(x)
        else:
            r.c = r.h = 0
        r.lin_ineq = self._get_AX_Less_B_residuals(x)
        r.lin_eq= self._get_AeqX_eq_Beq_residuals(x)
        r.lb = self._getLbresiduals(x)
        r.ub = self._getUbresiduals(x)
#        self.prevVal['r']['Val'] = deepcopy(r)
#        self.prevVal['r']['x'] = copy(x)
        return r

    def getMaxResidual(self, x, retAll = False):
        """
        if retAll:  returns
        1) maxresidual
        2) name of residual type (like 'lb', 'c', 'h', 'Aeq')
        3) index of the constraint of given type
        (for example 15, 'lb', 4 means maxresidual is equal to 15, provided by lb[4])
        don't forget about Python indexing from zero!
        if retAll == False:
        returns only r
        """

        residuals = self._getresiduals(x)
        r, fname, ind = 0, None, None
        for field in ('c',  'lin_ineq', 'lb', 'ub'):
            fv = asarray(getattr(residuals, field)).flatten()
            if fv.size>0:
                ind_max = argmax(fv)
                val_max = fv[ind_max]
                if r < val_max:
                    r, ind, fname = val_max, ind_max, field
        for field in ('h', 'lin_eq'):
            fv = asarray(getattr(residuals, field)).flatten()
            if fv.size>0:
                fv = abs(fv)
                ind_max = argmax(fv)
                val_max = fv[ind_max]
                if r < val_max:
                    r, ind, fname = val_max, ind_max, field
#        if self.probType == 'NLSP':
#            fv = abs(self.f(x))
#            ind_max = argmax(fv)
#            val_max = fv[ind_max]
#            if r < val_max:
#                r, ind, fname = val_max, ind_max, 'f'
        if retAll:
            return r, fname, ind
        else:
            return r

    def _getMaxConstrGradient2(self, x):
        g = zeros(self.n)
        mr0 = self.getMaxResidual(x)
        for j in range(self.n):
            x[j] += self.diffInt
            g[j] = self.getMaxResidual(x)-mr0
            x[j] -= self.diffInt
        g /= self.diffInt
        return g

    def getMaxConstrGradient(self, x,  retAll = False):
        g = zeros(self.n)
        maxResidual, resType, ind = self.getMaxResidual(x, retAll=True)
        if resType == 'lb':
            g[ind] -= 1 # N * (-1), -1 = dConstr/dx = d(lb-x)/dx
        elif resType == 'ub':
            g[ind] += 1 # N * (+1), +1 = dConstr/dx = d(x-ub)/dx
        elif resType == 'A':
            g += self.A[ind]
        elif resType == 'Aeq':
            rr = self.matmult(self.Aeq[ind], x)-self.beq[ind]
            if rr < 0:  g -= self.Aeq[ind]
            else:  g += self.Aeq[ind]
        elif resType == 'c':
            dc = self.dc(x, ind).flatten()
            g += dc
        elif resType == 'h':
            dh = self.dh(x, ind).flatten()
            if self.h(x, ind) < 0:  g -= dh#CHECKME!!
            else: g += dh#CHECKME!!
        if retAll:
            return g,  resType,  ind
        else:
            return g

#    def _getLagrangeresiduals(self, x, lm):
#        #lm is Lagrange multipliers
#        residuals = self.getresiduals(x)
#        r = 0
#
#        for field in ['c', 'h', 'A', 'Aeq', 'lb', 'ub']:
#            fv = getattr(residuals, field)
#            if fv not in ([], ()) and fv.size>0: r += self.dotwise(fv, getattr(lm, field))
#        return r
#        #return r.nonLinInEq * lm.nonLinInEq + r.nonLinEq * lm.nonLinEq + \
#                   #r.aX_Less_b * lm.aX_Less_b + r.aeqX_ineq_beq * lm.aeqX_ineq_beq + \
#                   #r.res_lb * lm.res_lb + r.res_ub * lm.res_ub

    def isFeas(self, x):
        if any(isnan(self._get_nonLinEq_residuals(x))) or any(isnan(self._get_nonLinInEq_residuals(x))):
            return False
        is_X_finite = all(isfinite(x))
        is_ConTol_OK = self.getMaxResidual(x) <= self.contol
        cond1 = is_ConTol_OK and is_X_finite and all(isfinite(self.objFunc(x)))
        if self.probType in ('NLSP', 'SNLE'): return cond1 and self.F(x) < self.ftol
        else: return cond1

    def discreteConstraintsAreSatisfied(self, x):
        k = -1
        A = array([0, 1])
        for i in self.discreteVars.keys():#range(m):	# check x-vector
            # TODO: replace it by "for i, val in dict.itervalues()"
            s = self.discreteVars[i] if self.discreteVars[i] is not bool and self.discreteVars[i] is not 'bool' else A
            if not any(abs(x[i] - s) < self.discrtol):
                k=i	# Violation of this set constraint.
                break # Go and split for this x-component
        if k == -1:
            return True
        else:
            return False
        
class EmptyClass:
    pass

