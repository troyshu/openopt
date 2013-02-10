'''
Translated from Octave code at: http://www.ecs.shimane-u.ac.jp/~kyoshida/lpeng.htm
and placed under MIT licence by Enzo Michelangeli with permission explicitly
granted by the original author, Prof. Kazunobu Yoshida  

-----------------------------------------------------------------------------
Copyright (c) 2010, Kazunobu Yoshida, Shimane University, and Enzo Michelangeli, 
IT Vision Limited

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
-----------------------------------------------------------------------------

Usage:
 
 optx,zmin,is_bounded,sol,basis = lp(c,A,b)
 
  This program finds a solution of the standard linear programming problem:
    minimize    z = c'x
    subject to  Ax = b, x >= 0
  using the two phase method, where the simplex method is used at each stage.
  Returns the tuple:
    optx: an optimal solution.
    zmin: the optimal value. 
    is_bounded: True if the solution is bounded; False if unbounded.
    sol: True if the problem is solvable; False if unsolvable.
    basis: indices of the basis of the solution.
    
'''
from numpy import *
#from numpy.linalg import norm
#from numpy import dot, asfarray, atleast_1d,  zeros, ones, int, float64, where, inf, ndarray
from openopt.kernel.baseSolver import baseSolver
from openopt.kernel.nonOptMisc import isspmatrix, scipyInstalled, scipyAbsentMsg
try:
    from openopt.kernel.nonOptMisc import isPyPy
except:
    pass
from openopt.kernel.ooMisc import xBounds2Matrix
from openopt.kernel.nonOptMisc import Hstack, Vstack, SparseMatrixConstructor, Eye, Diag, DenseMatrixConstructor
#try:
#    from scipy.sparse import csc_matrix, csr_matrix
#except:
#    pass

class pclp(baseSolver):
    __name__ = 'pclp'
    __license__ = "MIT"
    __authors__ = ''
    __alg__ = 'a simplex method implementation'
    __info__ = """    """
       
    __optionalDataThatCanBeHandled__ = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub']
    _canHandleScipySparse = True

    
    def __init__(self): pass

    def __solver__(self, p):
        xBounds2Matrix(p)
        n = p.n
        
        
        Ind_unbounded = logical_and(isinf(p.lb), isinf(p.ub))
        ind_unbounded = where(Ind_unbounded)[0]
        n_unbounded = ind_unbounded.size
        
        # Cast linear inequality constraints into linear equality constraints using slack variables
        nLinInEq, nLinEq = p.b.size, p.beq.size
        
        #p.lb, p.ub = empty(n+nLinInEq+n_unbounded), empty(n+nLinInEq+n_unbounded)
        #p.lb.fill(-inf)
        #p.ub.fill(inf)
        
#        for fn in ['_A', '_Aeq']:
#            if hasattr(p, fn): delattr(p, fn)

        # TODO: handle p.useSparse parameter
        
#        if n_unbounded != 0:
#            R = SparseMatrixConstructor((n_unbounded, n))
#            R[range(n_unbounded), ind_unbounded] = 1.0
#            R2 = Diag([1]*nLinInEq+[-1]*n_unbounded)
#            _A = Hstack((Vstack((p.A, R)), R2))
#        else:
#            _A = Hstack((p.A, Eye(nLinInEq)))

        _A = Hstack((p.A, Eye(nLinInEq), -Vstack([p.A[:, i] for i in ind_unbounded]).T if isPyPy else -p.A[:, ind_unbounded]))
        

#        if isspmatrix(_A): 
#            _A = _A.A
        
        # add lin eq cons
        if nLinEq != 0:
            Constructor = SparseMatrixConstructor if scipyInstalled and nLinInEq > 100 else DenseMatrixConstructor
            _A = Vstack((_A, Hstack((p.Aeq, Constructor((nLinEq, nLinInEq)), \
                                                         -Vstack([p.Aeq[:, i] for i in ind_unbounded]).T if isPyPy else -p.Aeq[:, ind_unbounded]))))
            
        if isspmatrix(_A): 
            if _A.size > 0.3 * prod(_A.shape):
                _A = _A.A
            else:
                _A = _A.tolil()
            #_A = _A.A
        
        #p.A, p.b = zeros((0, p.n)), array([])
        #_f = hstack((p.f, zeros(nLinInEq+n_unbounded)))
        
        #_f = Hstack((p.f, zeros(nLinInEq), -p.f[ind_unbounded]))
        #_b = hstack((p.b, [0]*n_unbounded, p.beq))
        
        _f = hstack((p.f, zeros(nLinInEq), -p.f[Ind_unbounded]))
        
        _b = hstack((p.b, p.beq))
        
        
        if p.useSparse is False and isspmatrix(_A): _A = _A.A
        p.debugmsg('handling as sparse: ' + str(isspmatrix(_A)))
        optx,zmin,is_bounded,sol,basis = lp_engine(_f, _A, _b)
        p.xf = optx[:n] -optx[-n:] if len(optx) != 0 else nan
        p.istop = 1000 if p.xf is not nan else -1000
        
#        from openopt  import LP
#        p2 = LP(_f, Aeq=_A.copy(), beq=_b, lb=[0]*_f.size)
#        r=p2.solve('lpSolve', maxIter = 1e4)
#        x_opt_2= r.xf[:n] - r.xf[-n:]
        
        #assert len(optx)>0
        #print linalg.norm(p.xf - x_opt_2)
        #raise 0
        

def lp_engine(c, A, b):
#    c = asarray(c)
#    A = asarray(A)
#    b = asarray(b)

    m,n = A.shape
    ind = b < 0
    if any(ind):
        b = abs(b)
        #A[ind, :] = -A[ind, :] doesn't work for sparse
        if type(A) == ndarray and not isPyPy:
            A[ind] = -A[ind]
        else:
            for i in where(ind)[0]:
                A[i,:] = -A[i,:]
            
    d = -A.sum(axis=0)
    if not isscalar(d) and type(d) != ndarray:
        d = d.A.flatten()
    if not isinstance(d, ndarray): d = d.A.flatten() # d may be dense matrix
    w0 = sum(b)
    # H = [A b;c' 0;d -w0];
    #H = bmat('A b; c 0; d -w0') 
    ''''''
    H = Vstack([     #  The initial _simplex table of phase one
         Hstack([A, atleast_2d(b).T]), # first m rows
         hstack([c, 0.]),   # last-but-one
         hstack([d, -asfarray(w0)])]) # last
    #print sum(abs(Hstack([A, array([b]).T])))

    if isspmatrix(H): H = H.tolil()
    ''''''
    indx = arange(n)
    basis = arange(n, n+m)
    #H, basis, is_bounded = _simplex(H, basis, indx, 1)
    is_bounded = _simplex(H, basis, indx, 1)
    if H[m+1,n] < -1e-10:   # last row, last column
        sol = False
        #print('unsolvable')
        optx = []
        zmin = []
        is_bounded = False
    else:
        sol = True
        j = -1
        
        #NEW
        tmp = H[m+1,:]
        if type(tmp) != ndarray: tmp = tmp.A.flatten()
        ind = tmp > 1e-10
        if any(ind):
            j = where(logical_not(ind))[0]
            H = H[:, j]
            indx = indx[j]
        #Old
#        for i in range(n):
#            j = j+1
#            if H[m+1,j] > 1e-10:
#                H[:,j] = []
#                indx[j] = []
#                j = j-1
        #H(m+2,:) = [] % delete last row
        H = H[0:m+1,:]
        if size(indx) > 0:
        # Phase two
            #H, basis, is_bounded = _simplex(H,basis,indx,2);
            is_bounded = _simplex(H,basis,indx,2)
            if is_bounded:
                optx = zeros(n+m)
                n1,n2 = H.shape
                for i in range(m):
                    optx[basis[i]] = H[i,n2-1]
                # optx(n+1:n+m,1) = []; % delete last m elements
                optx = optx[0:n] 
                zmin = -H[n1-1,n2-1]    #  last row, last column
            else:
                optx = []
                zmin = -Inf
        else:
            optx = zeros(n+m)
            zmin = 0
    return (optx, zmin, is_bounded, sol, basis)  

def _simplex(H,basis,indx,s):
    '''
      [H1,basis,is_bounded] = _simplex(H,basis,indx,s)
      H: simplex table (MODIFIED).
      basis: the indices of basis (MODIFIED).
      indx: the indices of x.
      s: 1 for phase one; 2 for phase two.
      H1: new simplex table.
      is_bounded: True if the solution is bounded; False if unbounded.
    '''
    if s == 1:
        s0 = 2
    elif s == 2:
        s0 = 1
    n1, n2 = H.shape
    sol = False
    while not sol:
        #print 'new Iter'
        # [fm,jp] = min(H(n1,1:n2-1));
        q = H[n1-1, 0:n2-1] # last row, all columns but last
        if type(q) != ndarray: q = q.toarray().flatten()
        jp = argmin(q)
        fm = q[jp]
        if fm >= 0:
            is_bounded = True    # bounded solution
            sol = True
        else:
            # [hm,ip] = max(H(1:n1-s0,jp));
            q = H[0:n1-s0,jp]
            if type(q) != ndarray: q = q.toarray().flatten()
            ip = argmax(q)
            hm = q[ip]
            if hm <= 0:
                is_bounded = False # unbounded solution
                sol = True
            else:
                h1 = empty(n1-s0)
                h1.fill(inf)
                
                # NEW
                tmp = H[:n1-s0,jp]
                if isspmatrix(tmp): tmp = tmp.A.flatten()
                ind = tmp>0
                
                #tmp2 = H[ind,n2-1]/H[ind,jp]
                tmp2 = hstack([H[i,n2-1]/H[i, jp] for i in where(ind)[0]]) if 1 or isPyPy else H[ind,n2-1]/H[ind,jp]
                
                #assert hstack([H[i,n2-1]/H[i, jp] for i in where(ind)[0]]).shape == (H[ind,n2-1]/H[ind,jp]).shape
                if isspmatrix(tmp2): tmp2 = tmp2.A
                
                if isPyPy:
                    for i, val in enumerate(where(ind)[0]):
                        h1[val] = tmp2[i]
                else:
                    h1[atleast_1d(ind)] = tmp2
                
                #OLD
#                for i in range(n1-s0):
#                    if H[i,jp] > 0:
#                        h1[i] = H[i,n2-1]/H[i,jp]
                        
                ip = argmin(h1)
                minh1 = h1[ip]
                basis[ip] = indx[jp]
                if not _pivot(H,ip,jp):
                    raise ValueError("the first parameter is a Singular matrix") 
    return is_bounded

def _pivot(H,ip,jp):
    # H is MODIFIED
    n, m = H.shape
    piv = H[ip,jp]
    if piv == 0:
        #print('singular')
        return False
    else:
        #NEW
        H[ip,:] /= piv
        tmp2 = H[:,jp]*H[ip,:] if isspmatrix(H) else H[:,jp].reshape(-1, 1)*H[ip,:]
        if isspmatrix(tmp2): tmp2 = tmp2.tolil()
        tmp2[ip, :] = 0
        H -= tmp2
        #OLD
#        for i in range(n):
#            if i != ip:
#                H[i,:] -= H[i,jp]*H[ip,:]
    return True


######### Unit test section #########
#
#from numpy.testing import *
#
#def test_lp():
#    probs = [
#        {
#            'A': array([
#                [2.,  5., 3., -1.,  0.,  0.],
#                [3., 2.5, 8.,  0., -1.,  0.],
#                [8.,10.,  4.,  0.,  0., -1.]]),
#            'b': array([185., 155., 600.]),
#            'c': array([4., 8., 3., 0., 0., 0.]),
#            'result': [
#                    array([ 66.25, 0., 17.5, 0., 183.75, 0.]),
#                    317.5,
#                    True,
#                    True,
#                    array([2, 0, 4])            
#                ]
#        }, # add other test cases here...
#    ]
#
#    for prob in probs:
#        optx, zmin, bounded, solvable, basis = lp_engine(prob['c'],prob['A'],prob['b'])
#        expected_res = prob['result']
#        assert_almost_equal(optx, expected_res[0])
#        assert_almost_equal(zmin, expected_res[1])
#        assert_equal(bounded, expected_res[2])
#        assert_equal(solvable, expected_res[3])
#        assert_equal(basis, expected_res[4])
#
#if __name__ == "__main__":
#    run_module_suite()
