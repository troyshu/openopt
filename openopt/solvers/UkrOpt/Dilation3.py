from numpy import dot, sign, zeros, all, isfinite, array, sqrt, any, isnan, pi, sin, arccos, inf, argmax, asfarray
import numpy
from numpy.linalg import norm

class DilationUnit():
    
    #vectorNorm = 0 # TODO: remove it?
    #dilationCoeff = 1.0
    maxScalarComponentsLength = 2
    
    
    def __init__(self, vector, dilationCoeff):
        #self.vectorDirection = None
        self.scalarComponents = []
        nv = norm(vector)
        assert nv != 0
        #self.vectorDirection, self.vectorNorm, self.dilationCoeff = vector/nv, nv, dilationCoeff
        self.vectorDirection, self.dilationCoeff = vector/nv, dilationCoeff
        
class Dilation():
    #maxUnits = 10
    #treshhold = 1.01
    th_phi = 0.1
    dilationCoeffThreshold = 0.999999
    prevRest = None
    #maxVectorNum = 50
    
    def __init__(self, maxUnitsNum): 
        self.maxUnitsNum = maxUnitsNum
        self.units = []
        self.unitsNum = 0
        self.T = numpy.float64
        if hasattr(numpy, 'float128'):
            pass
            self.T = numpy.float128
    
    def addDilationUnit(self, vector, dilationCoeff = 0.99999):
        assert all(isfinite(vector))
        self.unitsNum += 1
        v = self.T(vector.copy())
        nv = norm(v)
        v /= nv
        
        # TODO: COMMENT IT OUT
#        M = 0
#        for i, unit in enumerate(self.units):
#            M = max((M, abs(dot(unit.vectorDirection, v))))
#        if M > 1e-2: 
#            print 'warning: max M=', M, 'nv=', nv
#            return
        
        #self.units.add(DilationUnit(vector.copy(), dilationCoeff))
        self.units.append(DilationUnit(v, dilationCoeff))
        print 'add new dilation vector; curr num: ', len(self.units)
        
    def getProjections(self, vv):
        #assert len(self.units) != 0
        V = self.T(vv)
        NV = norm(V)
        V /= NV
        r= []
        #print 'norm(V):', norm(V)
        for unit in self.units:
            # TODO: try to involve less multiplication ops
            scalarComponent = dot(unit.vectorDirection, V)
#            print 'norm(unit.vectorDirection):', norm(unit.vectorDirection)
#            print 'scalarComponent>>>', scalarComponent
            component =  unit.vectorDirection * scalarComponent#V
            r.append((scalarComponent, component, unit))
        for scalarComponent, component, unit in r: 
            V -= component
        return r, V#*NV
    
    def getDilatedDirection(self, direction):
        projectionsInfo, rest = self.getProjections(direction)
        dilatedDirection = zeros(direction.size)
        for scalarComponent, component, unit in projectionsInfo:
            dilatedDirection += component * unit.dilationCoeff
        return projectionsInfo, dilatedDirection+rest

    def getRestrictedDilatedDirection(self, direction):
        projectionsInfo, rest = self.getProjections(direction)
        dilatedDirection = zeros(direction.size)
        s, ns = [], []
        for scalarComponent, component, unit in projectionsInfo:
            t = component * unit.dilationCoeff
            s.append(t)
            ns.append(norm(t))
            dilatedDirection += t
        r = dilatedDirection+rest
        nr = norm(r)
        for i in xrange(len(s)):
            if ns[i] < 1e-10*nr:
                r += 1e-10*nr*s[i]/ns[i]-s[i]
        return projectionsInfo, r

    def getMostInsufficientUnit(self, scalarComponents):
        assert self.unitsNum != 0 
        ind, miUnit, miValue = 0, self.units[0], self.units[0].dilationCoeff#abs(scalarComponents[0])*(1-self.units[0].dilationCoeff)
        for i, unit in enumerate(self.units):
            #newValue = unit.dilationCoeff*abs(scalarComponents[i])
            newValue = unit.dilationCoeff#abs(scalarComponents[i])*(1-unit.dilationCoeff)
            if newValue > miValue:
                ind, miUnit, miValue = i, unit, newValue
        return ind, miUnit
        
    def updateDilationCoeffs2(self, scalarComponents, rest):
        arr_u = array([unit.dilationCoeff for unit in self.units])
        if self.unitsNum == 1: 
            self.units[0].dilationCoeff /= 2.0
            return
        m = self.unitsNum
        n = rest.size
        #th = norm(rest) * sqrt(n-m)
        
        for i, unit in enumerate(self.units):
            c = unit.dilationCoeff * abs(scalarComponents[i]) / n
            
            if c < 0.125:
                unit.dilationCoeff  *= 2.0
            elif c > 0.25 :
                unit.dilationCoeff  /= 2.0
            print i, unit.dilationCoeff
            
            
    def updateDilationCoeffs(self, scalarComponents, rest):
        arr_u = array([unit.dilationCoeff for unit in self.units])
#        if self.unitsNum == 1: 
#            self.units[0].dilationCoeff /= 2.0
#            return
        Ui2 = arr_u ** 2
        UiSCi = abs(array([unit.dilationCoeff*scalarComponents[i] for i, unit in enumerate(self.units)]))
        Ui2SCi2 = array(UiSCi) ** 2
        S, S2 = sum(Ui2), sum(Ui2SCi2)
        SCi = abs(array(scalarComponents))
        SCi2 = SCi ** 2
        
        alp = 2.0
        beta = 1.0 / alp
        
        #k = sqrt(S2 / (alp*sum((1.0-UiSCi)**2 * UiSCi2)))
        #rr = k * sqrt(1.0-UiSCi)**2
        m, n = self.unitsNum, rest.size
        b = abs(beta)
        #b = min((abs(beta), m/(16*n*sqrt(sum(UiSCi)))))
        #b = m/(n*sqrt(sum(UiSCi)))
        
        nr2 = norm(rest) ** 2
        k = b*sqrt(S2 / sum(Ui2SCi2*(1.0-UiSCi)))
        #k = sqrt(((b2-1)*nr2 + b2 * S2) / sum(Ui2SCi2*(1.0-UiSCi)))
#        k1 = sqrt(b2 * S2 / sum(Ui2SCi2*(1.0-UiSCi)))
#        m, n = self.unitsNum, rest.size
#        rr1 = k1 * (1-UiSCi)
#        u1 = rr1 * arr_u
        
        
        
        rr = k * (1-UiSCi)
        assert k > 0
        

        #k = sqrt(S2 / (alp*sum((1.0-SCi)**2 * Ui2SCi2)))
        #rr = k * sqrt(1.0-SCi)**2
       
        rr[rr>4.0] = 4.0
        rr[rr<0.25] = 0.25
        r = rr * arr_u
        #r[r<1e-20] = 1e-20
        assert len(r) == self.unitsNum == len(self.units)
        #print '--------------------'
        
        for i, unit in enumerate(self.units):
            unit.dilationCoeff = r[i]
        print 'nU=%d k=%0.1g r_min=%0.1g r_max=%0.1g' % (self.unitsNum, k, min(r), max(r))
        #print r
            #print i, unit.dilationCoeff
#        print 'old sum:', S2,'new sum:', sum(array([unit.dilationCoeff*scalarComponents[i] for i, unit in enumerate(self.units)])**2)
#        print '====================='
   
    def _updateDilationInfo(self, _dilationDirection_, ls, _moveDirection_):
        r = {'increased':0, 'decreased':0}
        #projectionsInfo, dilatedDirectionComponent, rest = self.getDilatedDirection(_moveDirection_)
        
        projectionsInfo1, rest1 = self.getProjections(_dilationDirection_)
        
        print 'norm(rest1):', norm(rest1)
        #cond_add = norm(rest1) > 1e-2
        s = abs(asfarray([scalarComponent for (scalarComponent, component, unit) in projectionsInfo1]))
        #cond_add = self.unitsNum == 0 or any(norm(rest1) > 64.0*asfarray([unit.dilationCoeff * s[i] for i, unit in enumerate(self.units)]))
        cond_add = norm(rest1) > 1e-3 #or (self.prevRest is not None and dot(self.prevRest, rest1) <= 0)
        if cond_add: 
            self.addDilationUnit(rest1)
            projectionsInfo1, rest1 = self.getProjections(_dilationDirection_)
            print 'norm(rest11):', norm(rest1)
        
        #print '!>', dot(dilatedDirectionComponent1, rest1) / norm(dilatedDirectionComponent1) / norm(rest1)
        #print '!!>', dilatedDirectionComponent1, rest1
        #assert norm(dilatedDirectionComponent1) > 1e-10
        
#        mostUnusefulUnitNumber = -1
#        mostUnusefulUnitCoeff = -1
#        for i, u in enumerate(self.units):
#            if u.dilationCoeff > mostUnusefulUnitCoeff:
#                mostUnusefulUnitNumber, mostUnusefulUnitCoeff = i, u.dilationCoeff
        
        #print 'norm(_dilationDirection_) :', norm(_dilationDirection_) 
        #s = norm(rest1) / norm(dilatedDirectionComponent1)       
        #print 's:', s
        
        scalarComponents = [scalarComponent for (scalarComponent, component, unit) in projectionsInfo1]
        #print argmax(scalarComponents)
        
        #m = len(self.units)
        #n = _dilationDirection_.size
        
        #print 'norm(rest1), norm(scalarComponents, inf):', norm(rest1) ,  norm(scalarComponents, inf)
        #print ''
        #condReasonableBigRest = self.unitsNum == 0  or norm(rest1) > norm(scalarComponents, inf)#miUnit.dilationCoeff* abs(scalarComponents[ind_mi])   # s > 0.05
        #print 'norm(rest1):', norm(rest1), 'miUnit.dilationCoeff:', miUnit.dilationCoeff, 'miUnit.sc:', scalarComponents[ind_mi]
        #if 1 or norm(rest1) > 0.9:
            #print '>', rest1/norm(rest1), norm(rest1), [Unit.dilationCoeff for Unit in self.units], [Unit.vectorDirection for Unit in self.units]
        #and self.prevRest is not None and (True or dot(self.prevRest, rest) <= 0)
        
        
        #projectionsInfo2, dilatedDirectionComponent2, rest2 = self.getDilatedDirection( dilatedDirectionComponent1 + rest1)
        #assert norm(dilatedDirectionComponent2) > 1e-10
        #projectionsInfo3, dilatedDirectionComponent, rest = self.getDilatedDirection( dilatedDirectionComponent + rest)
        
        projectionsInfo,  rest = projectionsInfo1, rest1
        #print 'norm(r1), norm(d1):', norm(rest1), norm(dilatedDirectionComponent1)
        #abs_tan_phi = norm(rest1) / norm(dilatedDirectionComponent1)
    
        #projectionsInfo,  dilatedDirectionComponent, rest = projectionsInfo2, dilatedDirectionComponent2, rest2
        #print 'norm(r2), norm(d2):', norm(rest2), norm(dilatedDirectionComponent2)
        
        #cond_drift = self.prevRest is not None and dot(self.prevRest, rest1) > 0
        # TODO: what if self.prevRest is None?
        
        #haveToAdd = condReasonableBigRest and any(rest1!=0) #and not cond_drift



            
        self.updateDilationCoeffs(scalarComponents, rest)



        self.prevRest = rest.copy()
        #print 'self.unitsNum, self.maxUnitsNum:', self.unitsNum,  self.maxUnitsNum
        if self.unitsNum >= self.maxUnitsNum:
            self.unitsNum = 0
            self.units = []
#            ind_mi, miUnit = self.getMostInsufficientUnit(scalarComponents)
#            self.units.pop(ind_mi)
#            self.unitsNum -= 1

#            for unit in self.units:
#                unit.dilationCoeff /= miUnit.dilationCoeff
            #print 'mi removed:', ind_mi
        nRemoved = self.cleanUnnessesaryDilationUnits()     
        if nRemoved: print 'nRemoved:', nRemoved
        return r            
            #d = 1.0
        
        #projectionsInfo, dilatedDirectionComponent, rest = self.getDilatedDirection(_dilationDirection_)
        
#        for scalarComponent, component, unit in projectionsInfo:
#            # angle between ort and dilation direction
#            angle = arccos(scalarComponent) # values from 0 to pi
#            if angle > pi/2: angle = pi - angle
#            if d < 1.0: 
#                unit.dilationCoeff *= max((0.5, sin(d*angle)))
#            elif d > 1.0:
#                unit.dilationCoeff *= max((2.0, d/sqrt(1-scalarComponent**2)))
#
#        if cond_overdilated:
#            #TODO - omit repeated calculations
#            nRemoved = self.cleanUnnessesaryDilationUnits()        
#            print 'REMOVED: ', nRemoved#, 'increaseMultiplier:', increaseMultiplier



#            if sign(dot(_moveDirection_, component)) == sign(scalarComponent):
#                #print 'case 1'
#                unit.dilationCoeff *=  multiplier
#                if unit.dilationCoeff  > 1.0: unit.dilationCoeff  = 1.0
#            else:
#                reduceMultiplier = max((0.5, sqrt(1 - scalarComponent**2)))
#                unit.dilationCoeff *= reduceMultiplier
                #unit.dilationCoeff /=  2.0#multiplier
                #print 'case 2'

        
        
        
#        #cond_overDilated = abs_tan_phi > 0.5#min((2.0, 1.0 / self.th_phi))
#        if abs_tan_phi < self.th_phi: #abs_tan_phi  < 0.1 * len(self.units) / (_dilationDirection_.size-len(self.units)): 
#            koeff = 0.5
#            for scalarComponent, component, unit in projectionsInfo:
#                unit.dilationCoeff *=  koeff * max((0.1, sqrt(1 - scalarComponent**2)))
#        #elif cond_overDilated: # TODO: use separate parameter instead of (1.0 / self.th_phi)
#        else:
#        #elif abs_tan_phi > 
#            multiplier = self.th_phi / abs_tan_phi
#            if multiplier > 2.0: multiplier = 2.0
#            elif multiplier < 1.3: multiplier = 1.3           
#            for scalarComponent, component, unit in projectionsInfo:
#                if sign(dot(_moveDirection_, component)) == sign(scalarComponent):
#                    unit.dilationCoeff *=  multiplier
#                    #pass
                
        
#        for scalarComponent, component, unit in projectionsInfo:
#            pass
            #reduceMultiplier = max((0.25, sqrt(1 - scalarComponent**2)))
            #unit.dilationCoeff *= reduceMultiplier        

        ##########################################
        
        
        # get NEW rest
        # TODO - omit repeated calculations
        #projectionsInfo, dilatedDirectionComponent, rest = self.getDilatedDirection(_dilationDirection_)
        ##########################################
        
        #if self.prevRest is not None:print 'sign dot:', sign(dot(self.prevRest, rest))
        #print 'norm(rest) / norm(dilatedDirectionComponent:', norm(rest)/ norm(dilatedDirectionComponent)
        
        #haveToAdd = True
        



    def cleanUnnessesaryDilationUnits(self):
        indUnitsToRemove = []
        for i, unit in enumerate(self.units):
            if unit.dilationCoeff > self.dilationCoeffThreshold:
                print '>>', unit.dilationCoeff ,  self.dilationCoeffThreshold
                indUnitsToRemove.append(i)
                #unitsToRemove.add(unit)
        for j in xrange(len(indUnitsToRemove)):
            self.units.pop(indUnitsToRemove[-1-j])
        nRemoved = len(indUnitsToRemove)
        self.unitsNum -= nRemoved
        #if nRemoved!=0: print 'dilation units: removed = %d left=%d' %  (nRemoved, len(self.units))
        return nRemoved
        
#    def cleanUnnessesaryDilationUnits(self):
#        unitsToRemove = set()
#        for unit in self.units:
#            if unit.dilationCoeff >= self.treshhold:
#                unitsToRemove.add(unit)
#        self.units = self.units.difference(unitsToRemove)


#    def updateDilationInfo(self, projectionsInfo, ls):
#        r = {'increased':0, 'decreased':0}
#        
#        for scalarComponent, component, unit in projectionsInfo:
#            scs = unit.scalarComponents
#            if len(scs) >= unit.maxScalarComponentsLength:
#                scs.pop(0)
#            scs.append(scalarComponent)
#            signs = sign(scs)
#            
#            #cond_1 = norm(unit.vectorDirection - array([1.0]+[0]*(unit.vectorDirection.size-1)))<1e-4
#            #cond_1 = any(unit.vectorDirection[:10]==1)
#            
#            # TODO: remove cycles
#            
#            if len(signs)>1: # TODO: handle signs[-1-i] == 0 case
#                if ls <= 1:
#                    if signs[-1] == signs[-2]:
#                        pass
#                    else:
#                        unit.dilationCoeff /= 2.0
#                        r['decreased'] += 1
#                elif ls > 1:
#                    if signs[-1] == signs[-2]:
#                        unit.dilationCoeff *= 1 + 0.1#ls/3.0
#                        r['increased'] += 1
#                    else:
#                        pass
#        return r
