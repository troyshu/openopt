from openopt import NLSP
from numpy import asfarray, zeros, cos, sin, inf

def test(complexity=0, **kwargs):

    f = lambda x: (x[0]**3+x[1]**3-9, x[0]-0.5*x[1], cos(x[2])+x[0]-1.5)

    #optional: gradient
    def df(x):
        df = zeros((3,3))
        df[0,0] = 3*x[0]**2
        df[0,1] = 3*x[1]**2
        df[1,0] = 1
        df[1,1] = -0.5
        df[2,0] = 1
        df[2,2] = -sin(x[2])
        return df

    x0 = [8,15, 80]

    #w/o gradient:
    #p = NLSP(f, x0)

    p = NLSP(f, x0, df = df, maxFunEvals = 1e5, iprint = -1, ftol = 1e-6, contol=1e-35)

    p.lb = [-inf, -inf, 150]
    p.ub = [inf, inf, 158]

    # you could try also comment/uncomment nonlinear constraints:
    p.c = lambda x: (x[2] - 150.8)**2-1.5
    # optional: gradient
    p.dc = lambda x: asfarray((0, 0, 2*(x[2]-150.8)))
    # also you could set it via p=NLSP(f, x0, ..., c = c, dc = dc)

    r = p.solve('nssolve', **kwargs)
    return r.istop>0, r, p


if __name__ == '__main__':
    isPassed, r, p = test()
    
    #assert r.istop > 0
