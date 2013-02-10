"""
The test is related to obsolete OpenOpt version and doesn't work for now
It was used to write an article related to numerical optimization.
"""

from numpy import *
from openopt import *
#from pylab import *

TestCollection = xrange(5, 26)
#TestCollection = xrange(5, 11)
#TestCollection = [5]
TestCollection = [21]

P = 0
#P = 5e0
xyCoordsAlwaysExist = False

solvers = ['ralg', 'ipopt','scipy_cobyla']
solvers = ['ralg', 'ipopt']
solvers = ['ralg']
#solvers = ['ipopt']
PLOT = 0
Results = {}

contol = 1e-8
xl, yl = 0, 15
yThreshold = -1.0


LeftPointForceX  = 10.00
LeftPointForceY = -3000.00

#maxY = array([yl, 10.0912237188, 5.81736169843, 2.38892359538,-0.528509686607, -3.61027100044,-7.21539449067,-11.0833587513,-14.5626399121,-17.2632362225,-19.3451872935,-21.3081000933,-23.4618325667,-25.641568098,-27.3915251977,-28.4811563214,-29.0663497915,-29.387570995,-29.4874690862,-29.2098877982,-28.3834712745,-27.140937007,-25.7876146522,-24.3568516911,-22.5652835804,-20.1292934475]) + yThreshold

maxY = array([inf, 10.0399786566, 5.61033453009, 2.06002552258, -0.96112448303, -4.13674736433, -7.84375429882, -11.8094052553, -14.9106560481, -17.1598341467, -18.7959512268, -20.240402618, -21.6919090157, -22.9270315304, -23.5750132311, -23.5920241002, -23.2285382362, -22.6103728598, -21.5974976575, -19.9739484665, -17.7441144032, -15.2828386661, -12.9735192373, -10.7235600474, -8.05824588682, -4.62542867851]) + yThreshold
#maxY[1] -= 100

for n in TestCollection:

    oovarInit = oovar('leftPointForces', v0 = [LeftPointForceX, LeftPointForceY], lb=[0, -inf])

    MaxForces = 100*sin(arange(n)) + 5000*ones(n)\
    + array([ -1.37163831e+03,  -1.60694848e+03,  -1.74685759e+03,\
            -1.77099023e+03,  -1.76295446e+03,  -1.83232862e+03,\
            -2.01327466e+03,  -2.23292838e+03,  -2.37990432e+03,\
            -2.40786079e+03,  -2.37604148e+03,  -2.39220174e+03,\
            -2.51227773e+03,  -2.68825834e+03,  -2.81448159e+03,\
            -2.82791948e+03,  -2.76385327e+03,  -2.71873406e+03,\
            -2.75969731e+03,  -2.86237323e+03,  -2.93613565e+03,\
            -2.91260196e+03,  -2.80726185e+03,  -2.69990883e+03,\
            -2.65982371e+03,  -2.68285823e+03,  -2.69793719e+03,\
            -2.63722331e+03,  -2.49886567e+03,  -2.34512448e+03,\
            -2.24405190e+03,  -2.20790597e+03,  -2.18342816e+03,\
            -2.10368051e+03,  -1.95071662e+03,  -1.77036769e+03,\
            -1.62911926e+03,  -1.55345727e+03,  -1.50621766e+03,\
            -1.42212572e+03,  -1.26885753e+03,  -1.07614103e+03,\
            -9.07638583e+02,  -8.02510393e+02,  -7.39726398e+02,\
            -6.58001243e+02,  -5.12877756e+02,  -3.17504802e+02,\
            -1.30014677e+02,   9.99898976e-03])[:n]

    lengths = 5*ones(n)+cos(arange(n))#array([4, 3, 1, 2])
    masses = 15*ones(n)+4*cos(arange(n))#array([10, 15, 20])

    g = 10
    Fm = masses * g

    ########################################################
    s = [20, 20]
    AdditionalMasses = oovar('AdditionalMasses', v0=s + [(100.0-sum(s))/(n-len(s))]*(n-len(s)), lb=zeros(n))
    ########################################################

    ########################################################
    from blockMisc import *

    #def blockEngineFunc(inp, AdditionalMasses, y_limit, blockID):
    def blockEngineFunc(inp, AdditionalMasses, blockID):
        if blockID == 0:
            lFx, lFy, lx, ly, prevBlockForceThreshold, prev_yLimit = inp[0], inp[1], xl, yl, 0, 0
        else:
            lFx, lFy, lx, ly, prevBlockForceThreshold, prev_yLimit = inp[0], inp[1], inp[2], inp[3], inp[4], inp[5]

        prevBlockBroken = blockID>=0 and isnan(prevBlockForceThreshold)\
        or \
        (prevBlockForceThreshold > 0 and P == 0)\
        or \
        (prev_yLimit > 0 or isnan(prev_yLimit) and P == 0 and not xyCoordsAlwaysExist)

        # calculate output
        Fwhole = sqrt(lFx**2+lFy**2)
        ForceThreshold = (Fwhole - MaxForces[blockID]) / 1e4

        CurrentAdditionalMass = AdditionalMasses[blockID]
        rFy = lFy +Fm[blockID] + CurrentAdditionalMass*g # TODO : store Fm[i] inside blocks
        rFx = lFx# Fx are same along whole chain

        dx, dy = lengths[blockID] * lFx/ Fwhole, lengths[blockID] * lFy/ Fwhole
        rx = lx + dx
        ry = ly + dy

        yLimit = ly - maxY[blockID]

        if P != 0:
            projection, distance = project2ball(x = [lFx, lFy], radius=MaxForces[blockID], center = 0)
            ForceThreshold += P * distance / 1e4

            projection, distance = project2box(ly, -inf, maxY[blockID])
            if distance > 0:
                yLimit = P/1e4 * distance

        if prevBlockBroken or (prev_yLimit>0 and P == 0): #and not xyCoordsAlwaysExist:
            rx, ry, ForceThreshold, rFx, rFy, yLimit = nan, nan, nan, nan, nan, nan


        r = array((rFx, rFy, rx, ry, ForceThreshold, yLimit))

        return r

    #def derivative_blockEngineFunc(inp, AdditionalMasses, y_limit, blockID):
    def derivative_blockEngineFunc(inp, AdditionalMasses, blockID):
        # TODO: return nans if prev block is broken
        if blockID == 0:
            lFx, lFy, lx, ly, prevBlockForceThreshold, prev_yLimit = inp[0], inp[1], xl, yl, 0, 0
        else:
            lFx, lFy, lx, ly, prevBlockForceThreshold, prev_yLimit = inp[0], inp[1], inp[2], inp[3], inp[4], inp[5]

        if blockID == 0:
            nVars = 2 + len(AdditionalMasses)
        else:
            nVars = 6 + len(AdditionalMasses)

        #r = zeros((len(inp)+1, nVars))
        r = zeros((6, nVars))

        # d_ rFx /
        r[0, 0] = 1 # d_lFx

        # d_rFy/
        r[1, 1] = 1 # d_lFy
        r[1, len(inp) + blockID] = g
        # TODO: Check it (below)
        #elif blockID < n-1:
            #r[3, 5 + blockID] = g

        # d_rx /
        if blockID!=0: r[2, 2] = 1 # dlx
        Fwhole = sqrt(lFx**2+lFy**2)
        #r[2, 0] = lengths[blockID] / Fwhole # dlFx
        r[2, 0] = lengths[blockID] * lFy**2 / (Fwhole ** 3) # d_lFx
        r[2, 1] = - lengths[blockID] * lFx * lFy / (Fwhole ** 3) # d_lFy

        # d_ry /
        if blockID!=0: r[3, 3] = 1 # dly
        r[3, 0] = - lengths[blockID] * lFx * lFy / (Fwhole ** 3) # d_lFx, and is same to r[2,1]
        r[3, 1] = lengths[blockID] * lFx**2 / (Fwhole ** 3) # d_lFy

        # dForceThreshold
        r[4, 0] = lFx / Fwhole / 1e4 # / dlFx
        r[4, 1] = lFy / Fwhole / 1e4 # / dlFy

        # dYlimit
        if blockID != 0: r[5, 3] = 1.0 #d_Ylimit / d_yl

        if P != 0:
            projection, distance = project2ball(x = [lFx, lFy], radius=MaxForces[blockID], center = 0)
            if distance != 0:
                penalty_derivative = P/1e4 * project2ball_derivative(x = [lFx, lFy], radius=MaxForces[blockID], center = 0)
                r[4, 0] += penalty_derivative[0]
                r[4, 1] += penalty_derivative[1]

            projection, distance = project2box(ly, -inf, maxY[blockID])
            if distance > 0:
                r[5, 3] = P/1e4 * project2box_derivative(ly, -inf, maxY[blockID])

            #projection, distance = project2box(ly, maxY[blockID], inf)
            #yLimit = P * distance
        prevBlockBroken = blockID>=0 and isnan(prevBlockForceThreshold)\
        or \
        (prevBlockForceThreshold > 0 and P == 0)\
        or \
        (prev_yLimit > 0 or isnan(prev_yLimit) and P == 0 and not xyCoordsAlwaysExist)
        if prevBlockBroken:
            r *= nan
        return r

    ooFuncs, c = [], []
    constrYmax = []

    for i in xrange(n):
        oof = oofun(blockEngineFunc, args = copy(i), name = 'blockEngine'+str(i))

        if i == 0:
            oof.input = (oovarInit, AdditionalMasses)
        else:
            oof.input = (ooFuncs[i-1], AdditionalMasses)
        oof.d = derivative_blockEngineFunc
        ooFuncs.append(oof)
        # TODO: replace "4" by named output "ForceThreshold"

        c.append(oolin([0, 0, 0, 0, 1, 0], input = ooFuncs[copy(i)], name='maxForce'+str(i)))
        c.append(oolin([0, 0, 0, 0, 0, 1], input = ooFuncs[copy(i)], name='Ylimit'+str(i)))
        #c.append(y_limit)


#    y_limit = oofun(lambda *inputs: [inp[1]-maxY[i] for i, inp in enumerate(inputs)], input = ooFuncs, name='maxY')
#    def d_y_limit(*inputs):
#        r = zeros((n, len(inputs[0])*n))
#        for i in xrange(n):
#            r[i, len(inputs[0]) * i + 1] = 1
#        return r
#    y_limit.d = d_y_limit
#
#    c.append(y_limit)

    f = oofun(lambda z: z[0]**1.5, input = ooFuncs[-1], d = lambda z:[0, 0, 1.5*z[0]**0.5, 0, 0, 0], name = 'objFunc')
    #f = oofun(lambda z: z[0], input = ooFuncs[-1], d = lambda z:[0, 0, 1, 0, 0,0], name = 'objFunc')
    #f = oolin(array([0, 0, 1, 0, 0, 0]), input = ooFuncs[-1], name = 'objFunc')

    sumOfMasses = oofun(lambda z: 1-z.sum()/100.0, input=AdditionalMasses, d = lambda z: -ones(n)/100.0, name='sOm')
    c.append(sumOfMasses)

    colors = ['b', 'r', 'g', 'y', 'm', 'c']

    for j, solver in enumerate(solvers):

        p = NLP(f, c=c, goal = 'max', gtol = 1e-6,  plot=0, contol = contol, maxFunEvals = 1e10)
        def callback(p):
            print p.c(p.xk)
            return 0
        #p.callback = callback
        if solver == 'scipy_cobyla':
            p.f_iter = max((int(n/2), 5))

        if solver == 'ipopt':
            p.maxIter = 1500 - 40*n
        else:
            p.maxIter = 15000

        r = p.solve(solver,  plot=0, showFeas=1, maxTime = 150, iprint = -1, ftol=1e-6, xtol=1e-6)

        Results[(n, p.solver.__name__)] = r
        if r.isFeasible: msgF = '+'
        else: msgF = '-'
        print 'n=%d' % n, ('f=%3.2f'% r.ff)+'['+msgF+']','Time=%3.1f' % r.elapsed['solver_time']


        if PLOT:
            hold(1)
            for i, oof in enumerate(ooFuncs):
                if i == 0:
                    plot([xl, ooFuncs[0](p.xk)[0]], [yl, ooFuncs[0](p.xk)[1]], colors[j])
                    plot([xl, ooFuncs[0](p.x0)[0]], [yl, ooFuncs[0](p.x0)[1]], 'k')
                else:
                    plot([ooFuncs[i-1](p.xk)[0], ooFuncs[i](p.xk)[0]], [ooFuncs[i-1](p.xk)[1], ooFuncs[i](p.xk)[1]], colors[j])
                    plot([ooFuncs[i-1](p.x0)[0], ooFuncs[i](p.x0)[0]], [ooFuncs[i-1](p.x0)[1], ooFuncs[i](p.x0)[1]], 'k')
    if PLOT: show()
