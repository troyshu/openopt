import os
from oologfcn import OpenOptException
from numpy import zeros, hstack, vstack, ndarray, copy, where, prod, isscalar, atleast_2d, eye, diag, asfarray
import sys
syspath = sys.path
Sep = os.sep

try:
    import scipy
    scipyInstalled = True
    scipyAbsentMsg = ''
    from scipy.sparse import isspmatrix, csr_matrix, coo_matrix
    from scipy.sparse import hstack as HstackSP, vstack as VstackSP, find as Find
    
    def Hstack(Tuple):
        ind = where([isscalar(elem) or prod(elem.shape)!=0 for elem in Tuple])[0].tolist()
        elems = [Tuple[i] for i in ind]
        if any([isspmatrix(elem) for elem in elems]):
            return HstackSP(elems)
        
        s = set([(0 if isscalar(elem) else elem.ndim) for elem in elems])
        ndim = max(s)
        if ndim <= 1:  return hstack(elems)
        #assert ndim <= 2 and 1 not in s, 'bug in OpenOpt kernel, inform developers'
        return hstack(elems) if 0 not in s else hstack([atleast_2d(elem) for elem in elems])
        
    def Vstack(Tuple):
        ind = where([prod(elem.shape)!=0 for elem in Tuple])[0].tolist()
        elems = [Tuple[i] for i in ind]
        if any([isspmatrix(elem) for elem in elems]):
            return VstackSP(elems)
        
        s = set([(0 if isscalar(elem) else elem.ndim) for elem in elems])
        ndim = max(s)
        if ndim <= 1:  return vstack(elems)
        #assert ndim <= 2 and 1 not in s, 'bug in OpenOpt kernel, inform developers'
        return vstack(elems) if 0 not in s else vstack([atleast_2d(elem) for elem in elems])
        
    #Hstack = lambda Tuple: HstackSP(Tuple) if any([isspmatrix(elem) for elem in Tuple]) else hstack(Tuple)
    #Vstack = lambda Tuple: VstackSP(Tuple) if any([isspmatrix(elem) for elem in Tuple]) else vstack(Tuple)
    SparseMatrixConstructor = lambda *args, **kwargs: scipy.sparse.lil_matrix(*args, **kwargs)
except:
    scipyInstalled = False
    csr_matrix = None
    coo_matrix = None
    scipyAbsentMsg = 'Probably scipy installation could speed up running the code involved'
    isspmatrix = lambda *args,  **kwargs:  False
    Hstack = hstack
    Vstack = vstack
    def SparseMatrixConstructor(*args, **kwargs): 
        raise OpenOptException('error in OpenOpt kernel, inform developers')
    def Find(*args, **kwargs): 
        raise OpenOptException('error in OpenOpt kernel, inform developers')

try:
    import numpypy
    isPyPy = True
except ImportError:
    isPyPy = False

DenseMatrixConstructor = lambda *args, **kwargs: zeros(*args, **kwargs)

pwSet = set()
def pWarn(msg):
    if msg in pwSet: return
    pwSet.add(msg)
    print('Warning: ' + msg)

def Eye(n):
    if not scipyInstalled and n>150:
        pWarn(scipyAbsentMsg)
    if n == 1:
        return 1.0
    elif n <= 16 or not scipyInstalled:
        return eye(n)
    else:
        return scipy.sparse.identity(n)

def Diag(x):
    if not scipyInstalled and len(x)>150: 
        pWarn(scipyAbsentMsg)
    if isscalar(x): return x
    elif len(x) == 1: return asfarray(x)
    elif len(x) < 16 or not scipyInstalled: return diag(x)
    else: return scipy.sparse.spdiags(x, [0], len(x), len(x)) 


##################################################################
solverPaths = {}
from os import path as os_path
FILE = os_path.realpath(__file__)
for root, dirs, files in os.walk(''.join([elem + os.sep for elem in FILE.split(os.sep)[:-2]+ ['solvers']])):
    rd = root.split(os.sep)
    if '.svn' in rd or '__pycache__' in rd: continue
    rd = rd[rd.index('solvers')+1:]
    for file in files:
        if file.endswith('_oo.py'):
            solverPaths[file[:-6]] = ''.join(rd+['.',file[:-3]])


def getSolverFromStringName(p, solver_str):
    if solver_str not in solverPaths:
        p.err('''
        incorrect solver is called, maybe the solver "%s" is misspelled 
        or requires special installation and is not installed, 
        check http://openopt.org/%s''' % (solver_str, p.probType))
    if p.debug:
        solverClass =  solver_import(solverPaths[solver_str], solver_str)
    else:
        try:
            solverClass = solver_import(solverPaths[solver_str], solver_str)
        except ImportError:
            p.err('incorrect solver is called, maybe the solver "' + solver_str +'" require its installation, check http://www.openopt.org/%s or try p._solve() for more details' % p.probType)
    r = solverClass()        
    if not hasattr(r, 'fieldsForProbInstance'): r.fieldsForProbInstance = {}
    return r

##################################################################
importedSet = set()
ooPath = ''.join(elem+Sep for elem in __file__.split(Sep)[:-3])
def solver_import(solverPath, solverName):
    if solverPath not in importedSet:
        importedSet.add(solverPath)
        syspath.append(ooPath+'openopt'+Sep + 'solvers'+''.join(Sep+elem for elem in solverPath.split('.')[:-1]))
    name = 'openopt.solvers.' + solverPath
    mod = __import__(name)
    components = name.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return getattr(mod, solverName)

def oosolver(solverName, *args,  **kwargs):
    if args != ():
        raise OpenOptException("Error: oosolver() doesn't consume any *args, use **kwargs only")
    from openopt.kernel.baseSolver import baseSolver
    if isinstance(solverName, baseSolver):
        return solverName
    try:
        if ':' in solverName:
            # TODO: make it more properly
            # currently it's used for to get filed isInstalled value
            # from ooSystem
            solverName = solverName.split(':')[1]
        solverClass = solver_import(solverPaths[solverName], solverName)
        solverClassInstance = solverClass()
        solverClassInstance.fieldsForProbInstance = {}
        for key, value in kwargs.items():
            if hasattr(solverClassInstance, key):
                setattr(solverClassInstance, key, value)
            else:
                solverClassInstance.fieldsForProbInstance[key] = value
        solverClassInstance.isInstalled = True
    except ImportError:
        
        solverClassInstance = baseSolver()
        solverClassInstance.__name__ = solverName
        solverClassInstance.fieldsForProbInstance = {}
        solverClassInstance.isInstalled = False
    #assert hasattr(solverClassInstance, 'fieldsForProbInstance')
    return solverClassInstance

def Copy(arg): 
    return arg.copy() if isinstance(arg, ndarray) or isspmatrix(arg) else copy(arg)

class EmptyClass: pass
