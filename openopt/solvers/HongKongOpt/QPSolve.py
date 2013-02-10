'''
Copyright (c) 2010 Enzo Michelangeli and IT Vision Ltd

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
'''

from numpy import *
from LCPSolve import LCPSolve

def QPSolve(Q, e, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None):
    '''
    Note: if lb == None lower bounds are assumed to be -Inf
          if ub == None upper bounds are assumed to be +Inf
    i.e., x is NOT assumed to be non-negative by default
    This quadratic solver works by converting the QP problem 
    into an LCP problem. It does well up to few hundred variables
    and dense problems (it doesn't take advantage of sparsity).
    It fails if Aeq is not full row rank, or if Q is singular.
    '''
    nvars = Q.shape[0] # also e.shape[0]
    # convert lb and ub (if present) into Ax <=> b conditions, but
    # skip redundant rows: the ones where lb[i] == -Inf or ub[i] == Inf
    if lb != None:
        delmask = (lb != -Inf)
        addA = compress(delmask, eye(nvars), axis=0)
        addb = compress(delmask, lb, axis=0)
        A = vstack([A, -addA]) if A != None else -addA
        b = concatenate([b, -addb]) if b != None else -addb
    if ub != None:
        delmask = (ub != Inf)
        addA = compress(delmask, eye(nvars), axis=0)
        addb = compress(delmask, ub, axis=0)
        A = vstack([A, addA]) if A != None else addA
        b = concatenate([b, addb]) if b != None else addb

    n_ineq = A.shape[0]
    #print "nett ineq cons:", n_ineq

    # if there are equality constraints, it's equiv to particular MLCP that 
    # can anyway be converted to LCP 
    '''
    minimize:
        e' x + 1/2 x' Q x 
    s.t.:
        A x <= b 
        Aeq x = beq
        
    Upper and lower bounds, if presents, are automatically added to (A, b)

    The Karush-Kuhn-Tucker first-order conditions (being mu and lambda the
    KKT multipliers) are:

        (1.1) e + Q x + Aeq' mu + A' lambda = 0
        (1.2) Aeq x = beq
        (1.3) s = b - A x
        (1.4) s >= 0
        (1.5) lambda >= 0   
        (1.6) s' lambda = 0

    lambda are multipliers of inequality constr., mu of equality constr.,
    s are the slack variables for inequalities.

    This is a MLCP, where s and lambda are complementary. However, 
    we can re-write (1.1) and (1.2) as:

        | Q  Aeq'| * | x |  =  - | e + A' lambda |
        |-Aeq  0 |   |mu |       |       beq       |

    ...and, as long as 

        | Q  Aeq'|
        |-Aeq  0 |

    ...is non-singular (TODO: this should be checked!), we can solve:

        | x | = inv(| Q  Aeq'|) * | -e - A' lambda |
        |mu |       |-Aeq  0 |    |       -beq       |

    Then, if we define:

        M =     | A 0 | * inv( | Q  Aeq'| ) * | A'|
                               |-Aeq  0 |     | 0 |    

        q = b + | A 0 | * inv( | Q  Aeq'| ) * | e |
                               |-Aeq  0 |     | beq |

    ...(1.3) can be rewritten as an LCP problem in (s, lambda):

        s = M lambda + q

    (proof: replace M and q in the eq. above, and simplify remembering
    that | A'| lmbd == | A' lmbd |  , finally reobtaining  s = b - Ax )
         | 0 |         |    0    |

    Now, as we saw,    

        | Q  Aeq'| * | x |  = - | e + A' lambda | 
        |-Aeq  0 |   |mu |      |      beq      |

    ...so we can find

        | x | = inv( | Q  Aeq'| ) * - | e + A' lambda |
        |mu |        |-Aeq    |       |      beq      |

    ...and x will be in the first nvar elements of the solution.
    (we can also verify that b - A x == s)

    The advantage of having an LCP in lambda and s alone is also greater efficiency
    (no mu's and x's are calculated by the Lemke solver). Also, x's are not
    necessarily positive (unlike s's and lambda's), unless there are eplicit A,b
    conditions about it. However, the matrix inversion does cause a loss of accuracy
    (which could be estimated through B's condition number: should this value be 
    returned with the status?).
    
    '''
    if Aeq != None:
        n_eq = Aeq.shape[0]
        B = vstack([
                hstack([ Q, Aeq.T]),
                hstack([-Aeq, zeros((n_eq, n_eq))])
             ])
        A0 = hstack([A, zeros((n_ineq, n_eq))])
    else:
        B = Q
        A0 = A

    #print "det(M):", linalg.det(B)
    #print "B's log10(condition number):", log10(linalg.cond(B))
    BI = linalg.inv(B)

    A0BI = dot(A0, BI)

    M = dot(A0BI, A0.T)
    q = b + dot(A0BI, concatenate((e, beq))) if Aeq != None else b + dot(A0BI, e)

    # LCP: s = M lambda + q, s >= 0, lambda >= 0, s'lambda = 0
    # print "M is",M.shape,", q is ",q.shape
    s, lmbd, retcode = LCPSolve(M,q) 

    if retcode[0] == 1:
        xmu = dot(BI, -concatenate([e + dot(A.T, lmbd), beq])) if Aeq != None else dot(BI, -(e + dot(A.T, lmbd)))
        x = xmu[:nvars]
    else:
        x = None

    return (x, retcode)
    
