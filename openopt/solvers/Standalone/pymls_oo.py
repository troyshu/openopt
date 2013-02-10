# -*- coding: utf-8 -*-
"""
Created on Thu May 05 20:02:00 2011

@author: Tillsten
"""
#from numpy.linalg import norm
#from numpy import dot, asfarray, atleast_1d,  zeros, ones, int, float64, where, inf
import numpy as np
from openopt.kernel.baseSolver import baseSolver

class pymls(baseSolver):
    __name__ = 'pymls'
    __license__ = "BSD"
    __authors__ = 'Till Stensitzki, based directly on code from qcat from Ola Harkegard (http://research.harkegard.se/)'
    __alg__ = ''
    __info__ = ''
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']

    #__bvls_inf__ = 1e100
    T = np.float64
    __pymls_inf__ = 1e9
    def __init__(self): pass

    def __solver__(self, p):
        T = self.T
        A, b = T(p.C),  T(p.d).copy().reshape(-1, 1)
        lb,  ub = p.lb.copy().reshape(-1, 1), p.ub.copy().reshape(-1, 1)
        lb[lb==-np.inf] = -self.__pymls_inf__
        ub[ub==np.inf] = self.__pymls_inf__
        xf = bounded_lsq(A,b,lb,ub)
        p.xf = p.xk = xf.flatten()



from scipy.linalg import qr
eps=np.finfo(float).eps

def mls(B,v,umin,umax,Wv=None,Wu=None,ud=None,u=None,W=None,imax=100):
    """
mls - Control allocation using minimal least squares.

[u,W,iter] = mls_alloc(B,v,umin,umax,[Wv,Wu,ud,u0,W0,imax])

 Solves the bounded sequential least-squares problem

   min ||Wu(u-ud)||   subj. to   u in M

 where M is the set of control signals solving

   min ||Wv(Bu-v)||   subj. to   umin <= u <= umax

 using a two stage active set method. Wu must be diagonal since the
 problem is reformulated as a minimal least squares problem. The
 implementation does not handle the case of coplanar controls.

  Inputs:
  -------
 B     control effectiveness matrix (k x m)
 v     commanded virtual control (k x 1)
 umin  lower position limits (m x 1)
 umax  upper position limits (m x 1)
 Wv    virtual control weighting matrix (k x k) [I]
 Wu    control weighting matrix (m x m), diagonal [I]
 ud    desired control (m x 1) [0]
 u0    initial point (m x 1)
 W0    initial working set (m x 1) [empty]
 imax  max no. of iterations [100]
 
  Outputs:
  -------
 u     optimal control
 W     optimal active set
 iter  no. of iterations (= no. of changes in the working set + 1)

                           0 if u_i not saturated
 Active set syntax: W_i = -1 if u_i = umin_i
                          +1 if u_i = umax_i

Directly Based on the code from:
     Ola Härkegård, www.control.isy.liu.se/~ola   
    see licsence. 
     
"""

  
 #k = number of virtual controls
 #m = number of variables (actuators)

    k,m=B.shape    
    if u==None:
        u=np.mean(umin+umax,0)[:,None]                
    if W==None:
        W=np.zeros((m,1))
    if ud==None:
        ud=np.zeros((m,1))
    if Wu==None:
        Wu=np.eye(m)
    if Wv==None:
        Wv=np.eye(k)
    
    phase=1
    
    #Reformulate as a minimal least squares problem. See 2002-03-08 (1).    
    A=Wv.dot(B).dot(np.linalg.pinv(Wu))    
    #print B, v
    #A=B    
    b = Wv.dot(v-B.dot(ud))
    #b=v
    #print b
    xmin = (umin-ud)
    xmax = (umax-ud)
    # Compute initial point and residual.
    x = Wu.dot(u-ud)
    #x#=umin-umax
    r = A.dot(x)-b
#    print x.shape, r.shape, b.shape,x,r
    #Determine indeces of free variables
    i_free = W==0    
    m_free = np.sum(i_free)
    
    for i in range(imax):  
        #print 'Iter: ', i 
        if phase==1:
            A_free = A[:,i_free.squeeze()]
            if m_free<=k:                
                if m_free>0:
                    p_free=np.linalg.lstsq(-A_free,r)[0]                
            else:                
                q1,r1=qr(A_free.T)
                p_free=-q1.dot(np.solve(r1.T,r))
            p=np.zeros((m,1))
            p[i_free.squeeze()]=p_free
            
        
        else:            
            i_fixed=np.logical_not(i_free)
            m_fixed=m-m_free
            
            if m_fixed>0:            
                HT=U[i_fixed.squeeze(),:].T                      
                V,Rtot= qr(np.atleast_2d(HT))
                V1=V[:,:m_fixed]
                V2=V[:,m_fixed+1:]
                R=Rtot[:,m_fixed]      
            else:
                V,Rtot=np.array([[]]),np.array([[]])
                V1=V2=R=V.T
            
            s=-V2.T.dot(z)
            pz=V2.dot(s)
            p=U.dot(pz)
            
        x_opt=x+p
        infeasible=np.logical_or(x_opt<xmin,x_opt>xmax)
        if not np.any(infeasible[i_free]):
            x=x_opt
            
            if phase==1:
                r=r+A.dot(p)                
            else:
                z=z+pz
            if phase==1 and m_free>=k:
                phase=2
                Utot, Stot=qr(A.T)
                U=Utot[:,k:]
                z=U.T.dot(x)
                
                
            else:            
                lam=np.zeros((m,1))
                if m_free<m:
                    if phase==1:
                        g=A.T.dot(r)
                        lam=-W*g
                    else:
                        lam[i_fixed]=-W[i_fixed]*np.solve(R,V1.T.dot(z))
                    
                    if np.all(lam>= -eps):
                        u=np.linalg.solve(Wu,x)+ud
                        return u
                
                lambda_neg,i_neg=np.min(lam),np.argmin(lam)
                W[i_neg]=0
                i_free[i_neg]=1
                m_free+=1
        else:
            dist=np.ones((m,1))            
            
            i_min=np.logical_and(i_free,p<0)
            i_max=np.logical_and(i_free,p>0)
            
            
                
            dist[i_min]=(xmin[i_min]-x[i_min])/p[i_min]
            dist[i_max]=(xmax[i_max]-x[i_max])/p[i_max]
                
            
            alpha,i_alpha=np.min(dist),np.argmin(dist)
            
            x = x + alpha*p
            if phase==1:
                r=r+A.dot(alpha*p) #!!                
            else:
                z=z+alpha*pz
            W[i_alpha]=np.sign(p[i_alpha])
            i_free[i_alpha]=0
            m_free-=1
    u=np.linalg.solve(Wu,x)+ud
    return u
    
def bounded_lsq(A,b,lower_lim,upper_lim):
    """
    Minimizes:
    
    |Ax-b|_2 

    for lower_lim<x<upper_lim.
    """
    return mls(A,b,lower_lim,upper_lim)

#if __name__=='__main__':
#    from numpy.core.umath_tests import matrix_multiply
#    import matplotlib.pyplot as plt
#    plt.rcParams['font.family']='serif'
#    A=np.array([[1,-3],[5,7]])
#    b=np.array([[-50],[50]])
#    ll=np.array(([[-10],[-10]]))
#    ul=np.array(([[10],[10]]))
#    
#    Ud=np.array(([0,0]))
#    gamma=1000
#    
#    
#    x0=bounded_lsq(A,b,ll,ul)
#    x=np.linspace(-30,30,500)
#    y=np.linspace(-30,30,500)
#    
#    
#    X,Y=np.meshgrid(x,y)
#    
#    S=np.dstack((X,Y))
#    SN=matrix_multiply(S,A.T)
#    plt.clf()
#   
#    plt.contourf(x,y,np.sqrt(((SN-b.T)**2).sum(-1)),30,cmap=plt.cm.PuBu)
#    plt.colorbar()
#    #plt.axhline(ll[0])
#    #plt.axhline(ul[0])
#    #plt.axvline(ll[1])
#    #plt.axvline(ul[1])
#    rect=np.vstack((ll,ul-ll))
#    patch=plt.Rectangle(ll,*(ul-ll),facecolor=(0.0,0.,0.,0))
#    plt.gca().add_patch(patch)
#    plt.annotate("Bounded Min",
#                xy=x0, xycoords='data',
#                xytext=(-5, 5), textcoords='data',
#                arrowprops=dict(arrowstyle="->",
#                                connectionstyle="arc3"),
#                )
#    
#    plt.annotate("Lsq Min",
#                xy=np.linalg.lstsq(A,b)[0], xycoords='data',
#                xytext=(20, 10), textcoords='offset points',
#                arrowprops=dict(arrowstyle="->",
#                                connectionstyle="arc3"),
#                )
#                
#    plt.scatter(*x0)
#    plt.scatter(*np.linalg.lstsq(A,b)[0])
#    plt.show()
#    
    
            
    
        
        
