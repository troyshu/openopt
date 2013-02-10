#from numpy import asfarray, argmax, sign, inf, log10
from openopt.kernel.baseSolver import baseSolver
#from numpy import asfarray,  inf,  atleast_1d
from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F
import numpy as np
nanPenalty = 1e10

# In PyPy np.random is unimplemented yet
#global asdf
#asdf = 0
try:
    from np import random
    Rand = random.rand
    Seed = random.seed
    Randint = random.randint
except:# ImportError, AttributeError
 
    import random
    Seed = random.seed
    def Rand(*shape):
        
        r = np.empty(np.prod(shape))
        for i in range(r.size):
            r[i] = random.random()
        
#        #change
#        global asdf
#        for i in range(r.size):
#            asdf += 1
#            r[i] = np.sin(asdf)
        
        return r.reshape(shape)
    def Randint(low, high=None, size = None):
        assert high is None, 'unimplemented yet'
        if size is None:
            
            #change
#            global asdf
#            asdf += 1
#            return asdf % low
            
            return random.randint(0, low-1)
        a = np.empty(np.prod(size) if not np.isscalar(size) else size, dtype=int)
        for i in range(a.size):
            a[i] = random.randint(0, low-1)
            
            #change
#            global asdf
#            asdf += 1
#            a[i] = asdf % low
            
            
        return a.reshape(size)


class de(baseSolver):
    __name__ = 'de'
    __license__ = "BSD"
    __authors__ = "Stepan Hlushak, stepanko - at - gmail - dot - com, connected to OO by Dmitrey"
    __alg__ = "Two array differential evolution algorithm, Feoktistov V. Differential Evolution. In Search of Solutions (Springer, 2006)(ISBN 0387368965)."
    iterfcnConnected = True
    __homepage__ = ''
    __isIterPointAlwaysFeasible__ = lambda self, p: p.__isNoMoreThanBoxBounded__()
    __optionalDataThatCanBeHandled__ = ['lb', 'ub', 'A', 'b', 'Aeq','beq','c','h']
    _requiresFiniteBoxBounds = True
    
    """Strategy for selecting base vector.
    'random' - random base vector.
    'best' - base vector is chosen to be the best one.
    """
    baseVectorStrategy = 'random'
    
    """Strategy of evaluating difference vector.
    'random' - random direction.
    'best' - move into direction of best vectors.
    """
    searchDirectionStrategy = 'random'
    
    """Strategy of choosing F factor.
    'constant' - F factor is constant.
    'random' - F is vector of random numbers, that are 
             less or equal than F.
    """
    differenceFactorStrategy = 'random'
    
    population = 'default: 10*nVars will be used' 
    differenceFactor = 0.8
    crossoverRate = 0.5
    hndvi = 1
    seed = 150880
    

    __info__ = """
        This is two array differential evolution algorithm.
        
        Can handle constraints Ax <= b and c(x) <= 0 
        
        Finite box-bound constraints lb <= x <= ub are required.
        
        Parameters:
            population - population number (should be ~ 10*nVars),
            differenceFactor - difference factor (should be in (0,1] range),
            crossoverRate - constant of crossover (should be ~ 0.5, in (0,1] range),
            baseVectorStrategy - strategy of selection of base vector, can 
                                   be 'random' (default) or 'best' (best vector).
            searchDirectionStrategy - strategy for calculation of the difference vector,
                                    can be 'random' (random direction, default) or 
                                    'best' (direction of the best vectors).
            differenceFactorStrategy - strategy for difference factror (F), can be
                                    'constant' (constant factor, default) or
                                    'random' (F is chosen random for each vector and 
                                    each component of vector).
            hndvi -   half of number of individuals that take part in creation
                         of difference vector.
        """

    def __init__(self):pass
    def __solver__(self, p):

        #def DE2arr(func, lb, ub, itermax=1000, NP=100, F=0.8, Cr=0.5, strategy=(1, 1, 1, 1), constraints=None):
        
        if not p.__isFiniteBoxBounded__(): p.err('this solver requires finite lb, ub: lb <= x <= ub')
        
        p.kernelIterFuncs.pop(SMALL_DELTA_X, None)
        p.kernelIterFuncs.pop(SMALL_DELTA_F, None)
        
        lb, ub = p.lb, p.ub
        D = p.n#dimension

        if isinstance(self.population,str):
            NP = 10*D
        else:
            NP = self.population
            
        F = self.differenceFactor
        
        Cr = self.crossoverRate
        
        #func = p.f
        
        #constraints = lambda x: np.hstack((p.c(x), p._get_AX_Less_B_residuals(x)))
        
        #################################################
        Seed(self.seed)
        
        #initialize population
        pop = Rand(NP,D)*(ub-lb) + lb
        
        if np.any(np.isfinite(p.x0)):
            pop[0] = np.copy(p.x0)

        #evaluate  population 
        best, vals, constr_vals = _eval_pop(pop, p)

        Best = p.point(best[2], f=best[0], mr = best[1], mrName = None, mrInd = 0)

        if self.baseVectorStrategy == 'random':
            useRandBaseVectStrat = True 
        elif self.baseVectorStrategy == 'best':
            useRandBaseVectStrat = False
        else:
            p.err('incorrect baseVectorStrategy, should be "random" or "best", got ' + str(self.baseVectorStrategy))
        
        if self.searchDirectionStrategy == 'random':
            useRandSearchDirStrat = True
        elif self.searchDirectionStrategy == 'best':
            useRandSearchDirStrat = False
        else:
            p.err('incorrect searchDirectionStrategy, should be "random" or "best", got ' + str(self.searchDirectionStrategy))
            
        if self.differenceFactorStrategy == 'random':
            useRandDiffFactorStrat = True 
        elif self.differenceFactorStrategy == 'constant':
            useRandDiffFactorStrat = False
        else:
            p.err('incorrect differenceFactorStrategy, should be "random" or "constant", got ' + str(self.differenceFactorStrategy))
        
        for i in range(p.maxIter+10):
            
            old_pop = pop
            old_vals = vals
            old_constr_vals = constr_vals
            Old_best = Best

            #BASE VECTOR
            if useRandBaseVectStrat: #random base vector
                try:
                    beta = old_pop[Randint(NP, size=NP)]
                except:
                    beta = np.array([old_pop[Randint(NP)] for j in range(NP)])
            else: #best vector
                beta = np.ones((NP,D),'d')
                beta = beta*best[2] #TODO: I think there is a better way to create such matrix

            num_ind = self.hndvi #half of the number of individuals that take part
            #                      in differential creation
            
            #DIFFERENCE
            if useRandSearchDirStrat: #random search
                r1_ints = Randint(NP, size=(num_ind,NP))
                r2_ints = Randint(NP, size=(num_ind,NP))
                try:
                    barycenter1 = (old_pop[r1_ints]).sum(0)
                    barycenter2 = (old_pop[r2_ints]).sum(0)
                except: # PyPy
                    assert self.hndvi == 1, 'unimplemented for PyPy yet'
                    barycenter1 = np.array([old_pop[j] for j in r1_ints[0]])
                    barycenter2 = np.array([old_pop[j] for j in r2_ints[0]])
            else: #directed search
                r_ints = Randint(NP, size=(2*num_ind))
                list = [ (j,constr_vals[j],vals[j]) for j in r_ints]
                list_arr = np.array(list, dtype=[('i', int),
                                                  ('constr_val', float),
                                                  ('val', float)])
                list_arr.sort(order=['constr_val','val'])
                best_list = list_arr[0:num_ind]
                best_arr = np.array([j for (j,c,f) in best_list], 'i')
                worst_list = list_arr[num_ind:2*num_ind]
                worst_arr = np.array([j for (j,c,f) in worst_list], 'i')
                
                try:
                    barycenter1 = ((old_pop[worst_arr]).sum(0))/num_ind
                    barycenter2 = ((old_pop[best_arr]).sum(0))/num_ind
                except: # PyPy
                    assert self.hndvi == 1, 'unimplemented for PyPy yet'
                    barycenter1 = np.array([old_pop[j] for j in worst_arr[0]])
                    barycenter2 = np.array([old_pop[j] for j in best_arr[0]])

            if num_ind != 1:
                barycenter1 /= num_ind
                barycenter2 /= num_ind

            delta = barycenter2 - barycenter1 #should be (NP,D)-shape array

            if useRandDiffFactorStrat:
                Ft = Rand(NP,D)
                Ft = Ft*F
            else:
                Ft = F
            
            pop = beta + Ft*delta
            
            #CROSSOVER
            cross_v = np.ones((NP,D))
            const_v = Rand(NP,D)

            const_v = np.ceil(const_v - Cr) 
            cross_v = cross_v - const_v
            
            pop = old_pop*const_v + pop*cross_v

            #CHECK CONSTRAINTS
            
            pop = _correct_box_constraints(lb,ub,pop)
            
            best, vals, constr_vals = _eval_pop(pop, p)
            Best = p.point(best[2], f=best[0], mr = best[1], mrName = None, mrInd = 0)
            
            #SELECTION
            bool_constr_v = old_constr_vals < constr_vals #true when the old individual is better
            bool_v = old_vals < vals #also true when the old individual is better

            # Stephan, I dislike all these integer operations, I'm sure bool operations are faster and clearer, but it's your right // Dmitrey
            zero_constr_v =  1*((constr_vals > 0) + (old_constr_vals > 0))  
            # 0 - when all constrains in old and new pops are satisfied
            # 1 - when not

            bool_constr_v = (bool_constr_v*4 - 2)*zero_constr_v
            bool_v = bool_v*2 - 1
            
            bool_v = bool_constr_v + bool_v
            
            bool_v = bool_v > 0 
            old_sel_v = (bool_v*1).reshape(NP,1) #converting from bool to int array
            sel_v = np.ones((NP,1)) - old_sel_v
            
            pop = old_pop*old_sel_v + pop*sel_v
            
            old_sel_v = old_sel_v.reshape(NP)
            sel_v = sel_v.reshape(NP)
            vals = old_vals*old_sel_v + vals*sel_v
            constr_vals = old_constr_vals*old_sel_v + constr_vals*sel_v
            #END SELECTION
            
            if Old_best.betterThan(Best):
                Best = Old_best
                
            p.iterfcn(Best)
            if p.istop: 
                return

def _eval_pop(pop, p):
    
    NP = pop.shape[0]
    
    constr_vals = np.zeros(NP)
    vals = p.f(pop).flatten()
    
    if vals.size == 1:
        vals = np.array([vals]*NP)
    vals[np.isnan(vals)] = np.inf

    if p.__isNoMoreThanBoxBounded__():
        best_i = vals.argmin()        
        best = (vals[best_i], 0, pop[best_i])
    else:
#        new = 1
#        
#        if new:# and p.isFDmodel:
            # TODO: handle nanPenalty * newPoint.nNaNs()
            #vals = np.empty(pop.shape[0])
            #vals.fill(np.nan)
        
        P = p.point(pop)

        constr_vals = P.mr(checkBoxBounds = False) + nanPenalty * P.nNaNs()

        ind = constr_vals < p.contol

        if not np.any(ind):
            j = np.argmin(constr_vals)
            bestPoint = p.point(pop[j])
            #bestPoint.i = j
        else:
            IND = np.where(ind)[0]
            #print(IND, pop.shape)
            try:
                P2 = np.atleast_2d(pop[IND])
                F = np.atleast_1d(vals[IND])
            except: # PyPy
                P2 = np.atleast_2d([pop[j] for j in IND])
                F = np.atleast_1d([vals[j] for j in IND])
            J = np.nanargmin(F)
            bestPoint = p.point(P2[J], f=F[J])# TODO: other fields
            #bestPoint.i = IND[J]
            #bestPoint.i = np.where(ind)[0]

        best = (bestPoint.f(), bestPoint.mr() + nanPenalty * bestPoint.nNaNs(), bestPoint.x)
#    print(vals, constr_vals)
#    from openopt.kernel.ooMisc import isSolved 
#    raise isSolved

    return best, vals, constr_vals
   
    
def _correct_box_constraints(lb, ub, pop):
    diff_lb = pop - lb
    check_lb = diff_lb < 0.0
    scale = 1.0
    while np.any(check_lb):
        check_lb = check_lb*1
        pop = pop - (1+scale)*check_lb*diff_lb
        diff_lb = pop - lb
        check_lb = diff_lb < 0.0 
        scale /= 2

    diff_ub = pop - ub
    check_ub = diff_ub > 0.0 
    scale = 1.0
    while np.any(check_ub):
        check_ub = check_ub*1
        pop = pop - (1+scale)*check_ub*diff_ub
        diff_ub = pop - ub
        check_ub = diff_ub > 0.0 
        scale /= 2

    # temporary fix
    ind = np.where(pop<lb)
    if ind[0].size:
        pop[ind] = lb[ind[1]]
    ind = np.where(pop>ub)
    if ind[0].size:
        pop[ind] = ub[ind[1]]

    return pop
