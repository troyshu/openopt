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
from qlcp import qlcp
try:
    import openopt
except:
    pass # if OpenOpt is not installed, the qpsolver kwarg can't be specified
    
def _simple_grad(f, x, delta = 1e-8):
    nvars = x.shape[0]
    Id = eye(nvars)*delta
    grad = array([(f(x+Id[i,:]) - f(x-Id[i,:]))/(2*delta) for i in range(nvars)])
    return grad

def _simple_hessian(f, x, delta = 1e-4):    # generally too slow for use
    g = lambda x: _simple_grad(f, x, delta = delta) # g(x) is the gradient of f
    return _simple_grad(g, x, delta=delta)

def _simple_hessdiag(f, x, delta = 1e-4):
    nvars = x.shape[0]
    Id = eye(nvars)*delta
    hd = array([(f(x+Id[i,:]) + f(x-Id[i,:]) - 2*f(x))/delta**2 for i in range(nvars)]).flatten()
    return diag(hd)

def sqlcp(f, x0, df=None, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None, minstep=1e-15, minfchg=1e-15, qpsolver=None, callback = None):
    '''
    SQP solver. Approximates f in x0 with paraboloid with same gradient and hessian,
    then finds its minimum with a quadratic solver (qlcp by default) and uses it as new point, 
    iterating till changes in x and/or f drop below given limits. 
    Requires the Hessian to be definite positive.
    The Hessian is initially approximated by its principal diagonal, and then
    updated at every step with the BFGS method.
    f:        objective function of x to be minimized
    x0:       initial value for f
    df:       gradient of f: df(f) should return a function of such as f(x) would
              return the gradient of f in x. If missing or None, an approximation 
              will be calculated with an internal finite-differences procedure.
    A:        array of inequality constraints (A x >= b)
    b:        right-hand side of A x >= b
    Aeq:      array of equality constraints (Aeq x = beq)
    beq:      right-hand side of Aeq x >= beq
    lb:       lower bounds for x (assumed -Inf if missing)
    ub:       upper bounds for x (assumed +Inf if missing)
    minstep:  iterations terminate when updates to x become < minstep (default: 1e-15)
    minfchg:  iterations terminate when RELATIVE changes in f become < minfchg (default: 1e-15)
    qpsolver: if None, qlcp; else a solver accepted by openopt.QP (if OpenOpt and 
              that particular solver are installed)
    '''
       
    nvars = x0.shape[0]
    x = x0.copy()
    niter = 0
    deltah = 1e-4
    deltag = deltah**2
    
    if df == None:  # df(x) is the gradient of f in x
        df = lambda x: _simple_grad(f, x, deltag)
    
    twoI = 2.*eye(nvars)
    oldfx = f(x)
    gradfx = df(x)  # return the gradient of f() at x
    hessfx = _simple_hessdiag(f,x,delta=deltah) # good enough, and much faster, but only works if REAL Hessian is DP!
    invhessfx = linalg.inv(hessfx)
    while True:
        niter += 1
        
        # compute the b, beq, lb and ub for the QP sub-problem (as bx, beqx, lbx, ubx)
        bx = b if b == None else b-dot(A,x)
        beqx = beq if beq == None else beq-dot(Aeq,x)
        lbx = lb if lb == None else lb - x
        ubx = ub if ub == None else ub - x

        if qpsolver == None:
            deltax = qlcp(hessfx, gradfx, A=A, b=bx, Aeq=Aeq, beq=beqx, lb=lbx, ub=ubx, QI=invhessfx)
        else:
            p = openopt.QP(hessfx, gradfx, A=A, b=bx, Aeq=Aeq, beq=beqx, lb=lbx, ub=ubx)
            p.ftol = 1.e-10
            r = p.solve(qpsolver, iprint = -1)
            deltax = p.xf
        
        if deltax == None:
            #print("Cannot converge, sorry.")
            x = None
            break
        
        x += deltax
        if linalg.norm(deltax) < minstep:
            break
        fx = f(x)
        if abs(fx-oldfx) < minfchg*abs(fx):
            break
        if callback is not None and callback(x):
            break
            
        oldfx = fx
        oldgradfx = gradfx.copy()
        gradfx = df(x)  # return the gradient of f() at the new x
        # we might also put a termination test on the norm of grad...
        
        '''
        # recalc hessian afresh would be sloooow...
        #hessfx = _simple_hessian(f,x,delta=deltah)  # return the hessian of f() at x
        hessfx = _simple_hessdiag(f,x,delta=deltah)  # return the hessian (diag only) of f() at x
        invhessfx = linalg.inv(hessfx)
        '''
        # update Hessian and its inverse with BFGS based on current Hessian, deltax and deltagrad    
        # See http://en.wikipedia.org/wiki/BFGS
        deltagrad = gradfx - oldgradfx
        hdx = dot(hessfx, deltax)
        dgdx = dot(deltagrad,deltax)
        #if dgdx < 0.:
        #    print "deltagrad * deltax < 0!" # a bad sign
        hessfx += ( outer(deltagrad,deltagrad) / dgdx - 
                    outer(hdx, hdx) / dot(deltax, hdx) )
        # now update inverse of Hessian  
        '''
        invhessfx = linalg.inv(hessfx)
        '''
        hidg = dot(invhessfx,deltagrad)
        oIdgdeltax = outer(hidg,deltax)
        invhessfx += ( (dgdx+dot(deltagrad,hidg))*outer(deltax,deltax)/(dgdx**2) -
                (oIdgdeltax+oIdgdeltax.T)/dgdx ) # just because invhessfx is symmetric, or else:
                #(oIdgdeltax+outer(deltax,dot(invhessfx.T,deltagrad)))/dgdx )
    return x, niter
    
