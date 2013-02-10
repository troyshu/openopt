from baseProblem import MatrixProblem
from numpy import asarray, ones, inf, dot, zeros, asfarray, atleast_1d
import NLP

class LP(MatrixProblem):
    _optionalData = ['A', 'Aeq', 'b', 'beq', 'lb', 'ub']
    expectedArgs = ['f', 'x0']
    probType = 'LP'
    allowedGoals = ['minimum', 'min', 'max', 'maximum']
    showGoal = True
    _lp_prepared = False

    def __init__(self, *args, **kwargs):
        self.goal = 'minimum'
        MatrixProblem.__init__(self, *args, **kwargs)
        if len(args) > 1 and not hasattr(args[0], 'is_oovar'):
            self.err('No more than 1 argument is allowed for classic style LP constructor')

    def _Prepare(self):
        if self._lp_prepared: return
        self._lp_prepared = True
        MatrixProblem._Prepare(self)
        if self.x0 is None: self.x0 = zeros(self.n)
        if hasattr(self.f, 'is_oovar'): # hence is oofun or oovar
            # freeVars and fixedVars are already rendered to both not-None here (if it's the case)
            # but current .D() implementation doesn't allow to use both arguments
            # BTW translator ignores fixed vars in derivatives, so passing fixedVars here can be omitted 
            # but is performed for more safety wrt future changes
            _f = self._point2vector(self.f.D(self._x0, fixedVars = self.fixedVars)) 
            self.f, self._f = _f, self.f
            self._init_f_vector = _f # we don't take p.goal into account here
            _c = self._f(self._x0) - dot(self.f, self.x0)
            self._c = _c
        else:
            self._init_f_vector = self.f # we don't take p.goal into account here
            self._c = 0
        self.f = atleast_1d(self.f)
        if not hasattr(self, 'n'): self.n = len(self.f)
        #print 'lb:', self.lb, 'ub:', self.ub
        if not hasattr(self, 'lb'): self.lb = -inf * ones(self.n)
        if not hasattr(self, 'ub'): self.ub = inf * ones(self.n)
#        if any(isnan(self.lb)): 
#            if self.lb.size != 1: self.err('NaN in lower bound for a variable from the problem')
#            self.lb = -inf * ones(self.n)
#        if any(isnan(self.ub)): 
#            if self.ub.size != 1: self.err('NaN in upper bound for a variable from the problem')
#            self.ub = inf * ones(self.n)
        
        if self.goal in ['max', 'maximum']:
            self.f = -asfarray(self.f)
            
    # TODO: handle this and SDP finalize in single func finalize_for_max
    def __finalize__(self):
        MatrixProblem.__finalize__(self)
        if self.goal in ['max', 'maximum']:
            self.f = -self.f
            for fn in ['fk', ]:#not ff - it's handled in other place in RunProbSolver.py
                if hasattr(self, fn):
                    setattr(self, fn, -getattr(self, fn))
        if hasattr(self, '_f'):
            self.f = self._f
            

            
    def objFunc(self, x):
        if self.isFDmodel:
            # TODO: fix it
            r = self._f(self._vector2point(x))
            return -r if self.goal in ['max', 'maximum'] else r
        else:
            return dot(self.f, x) + self._c

    def lp2nlp(self, solver, **solver_params):
        if self.isConverterInvolved and self.goal in ['max', 'maximum']:
            self.err('maximization problems are not implemented lp2nlp converter')
        ff = lambda x: dot(x, self.f)+self._c
        dff = lambda x: self.f
        if hasattr(self,'x0'): p = NLP.NLP(ff, self.x0, df=dff)
        else: p = NLP.NLP(ff, zeros(self.n), df=dff)
        self.inspire(p)
        self.iprint = -1

        # for LP plot is via NLP
        p.show = self.show
        p.plot, self.plot = self.plot, 0
        if self.isFDmodel:
            p._x0 = self._x0 # may be used in interalg 
        r = p.solve(solver, **solver_params)
        self.xf, self.ff, self.rf = r.xf, r.ff, r.rf

        return r

    def exportToMPS(self, filename, format='fixed', startIndex=0):
        try: from lp_solve import lpsolve
        except ImportError: self.err('To export LP/MILP in files you should have lpsolve and its Python binding properly installed')
        
        maxNameLength = 8 if format != 'free' else 255
        handler = self.get_lpsolve_handler(maxNameLength, startIndex)
        
        # TODO: uncomment it
        ext = 'mps' if not filename.endswith('MPS') and not filename.endswith('mps') else ''
        if ext != '': filename += '.' + ext
        
        if format=='fixed':
            r = bool(lpsolve('write_mps', handler, filename) )
        elif format=='free':
            r = bool(lpsolve('write_freemps', handler, filename) )
        else:
            self.err('incorrect MPS format, should be "fixed" or "free"')
        if r != True: 
            self.warn('Failed to write MPS file, maybe read-only filesystem, incorrect path or write access is absent')
            
        lpsolve('delete_lp', handler) 
        return r

    def get_lpsolve_handler(self, maxNameLength=255, startIndex=0):
        try: from lp_maker import lp_maker, lpsolve
        except ImportError: self.err('To export LP/MILP in files you should have lpsolve and its Python binding properly installed')
        self._Prepare()
        from ooMisc import LinConst2WholeRepr
        LinConst2WholeRepr(self)
        
        # set goal to  min/max
        minim = 0 if self.goal in ['max', 'maximum'] else 1
        
        # objective
        f = self._init_f_vector
        
        lp_handle = lp_maker(List(asarray(f).flatten()), List(self.Awhole), List(asarray(self.bwhole).flatten()), List(asarray(self.dwhole).flatten()), \
        List(self.lb), List(self.ub), (1+asarray(self._intVars_vector)).tolist(), 0,minim)
       
        #lp_handle = lpsolve('make_lp', len(self.beq)+len(self.b), self.n) 
        L = lambda action, *args: lpsolve(action, lp_handle, *args)
        
        # set name
        L('set_lp_name', self.name)
        
        # set boolean values if present
        #if len(self.boolVars)>0:
            #assert self.boolVars[0] in [True, False]
            #L('set_binary', self.boolVars)
        
        # set variables names
        if self.isFDmodel:
            assert not isinstance(self.freeVars, set), 'error in openopt kernel, inform developers'
            #assert len(self.freeVars) == self.n, 'not implemented yet for oovars of size > 1'
            #names = [oov.name for oov in self.freeVars]
            x0 = self._x0
            names = []
            #assert not isinstance(self.freeVars,  set), 'error in openopt kernel, inform developers'
            for oov in self.freeVars:
                if oov.name.startswith('unnamed'):
                    L('delete_lp')
                    self.err('For exporting FuncDesigner models into MPS files you cannot have variables with names starting with "unnamed"')
                if ' ' in oov.name:
                    L('delete_lp')
                    self.err('For exporting FuncDesigner models into MPS files you cannot have variables with spaces in names')
                Size = asarray(x0[oov]).size
                if Size == 1:
                    Name = oov.name
                    names.append(Name)
                else:
                    tmp = [(oov.name + ('_%d' % (startIndex+j))) for j in range(Size)]
                    names += tmp
                    Name = tmp[-1]
                if maxNameLength < len(Name):
                    L('delete_lp')
                    self.err('incorrect name "%s" - for exporting FuncDesigner models into MPS files you cannot have variables with names of length > maxNameLength=%d'% maxNameLength)
                    
            # TODO: check are names unique
            L('set_col_name', names) 
        return lp_handle
        
def List(x):
    if isinstance(x, list): return x
    elif x == None or x.size == 0: return None
    else: return x.tolist()        

