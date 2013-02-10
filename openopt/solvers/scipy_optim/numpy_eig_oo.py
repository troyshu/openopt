from openopt.kernel.baseSolver import baseSolver
from openopt.kernel.nonOptMisc import Vstack, isspmatrix
from numpy.linalg import eig

#from nonOptMisc import Hstack

class numpy_eig(baseSolver):
    __name__ = 'numpy_eig'
    __license__ = "BSD"
    __authors__ = ''
    __alg__ = ''
    __info__ = """    """

    __optionalDataThatCanBeHandled__ = []
    _canHandleScipySparse = True
    
    #def __init__(self): pass

    def __solver__(self, p):
        A = p.C
        if isspmatrix(A):
            p.warn('numpy.linalg.eig cannot handle sparse matrices, cast to dense will be performed')
            A = A.A
        #M = p.M
        
        if p._goal != 'all':
            p.err('numpy_eig cannot handle the goal "%s" yet' % p._goal)
        
        eigenvalues, eigenvectors = eig(A)
        p.xf = p.xk = Vstack((eigenvalues, eigenvectors))
        p.eigenvalues = eigenvalues
        p.eigenvectors = eigenvectors
        p.ff = 0
