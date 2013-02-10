from baseProblem import NonLinProblem
from numpy import inf

from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F
class MOP(NonLinProblem):
    _optionalData = ['lb', 'ub', 'A', 'Aeq', 'b', 'beq', 'c', 'h']
    showGoal = True
    goal = 'weak Pareto front'
    probType = 'MOP'
    allowedGoals = ['weak Pareto front', 'strong Pareto front', 'wpf', 'spf']
    isObjFunValueASingleNumber = False
    expectedArgs = ['f', 'x0']
    _frontLength = 0
    _nIncome = 0
    _nOutcome = 0
    
    iprint = 1
    
    def __init__(self, *args, **kwargs):
        NonLinProblem.__init__(self, *args, **kwargs)
        self.nSolutions = 'all'
        self.kernelIterFuncs.pop(SMALL_DELTA_X, None)
        self.kernelIterFuncs.pop(SMALL_DELTA_F, None)
        self.data4TextOutput = ['front length', 'income', 'outcome', 'log10(maxResidual)']
        f = self.f
        i = 0
        targets = []
        while True:
            if len(f[i:]) == 0: break
            func = f[i]
            if type(func) in (list, tuple):
                F, tol, val = func
                i += 1
            else:
                F, tol, val = f[i], f[i+1], f[i+2]
                i += 3
            t = target()
            t.func, t.tol = F, tol
            t.val = val if type(val) != str \
            else inf if val in ('max', 'maximum') \
            else -inf if val in ('min', 'minimum') \
            else self.err('incorrect MOP func target')
            targets.append(t)
        self.targets = targets
        self.f = [t.func for t in targets]
        self.user.f = self.f

    def objFuncMultiple2Single(self, fv):
        return 0#(fv ** 2).sum()

    def solve(self, *args, **kw):
#        if self.plot or kw.get('plot', False):
#            self.warn('\ninteractive graphic output for MOP is unimplemented yet and will be turned off')
#            kw['plot'] = False
        self.graphics.drawFuncs = [mop_iterdraw]
        r = NonLinProblem.solve(self, *args, **kw)
        r.plot = lambda *args, **kw: self._plot(**kw)
        r.__call__ = lambda *args, **kw: self.err('evaluation of MOP result on arguments is unimplemented yet, use r.solutions')
        r.export = lambda *args, **Kw: _export_to_xls(self, r, *args, **kw)
        T0 = self.targets[0]
        if T0.val == -inf:
            keyfunc = lambda elem: elem[T0.func]
        elif T0.val == inf:
            keyfunc = lambda elem: -elem[T0.func]
        else:
            keyfunc = lambda elem: abs(T0.val - elem[T0.func])
        r.solutions.sort(key=keyfunc)
        for v in self._categoricalVars:
            for elem in r.solutions:
                elem.useAsMutable = True
                elem[v] = v.aux_domain[elem[v]]
                elem.useAsMutable = False
        return r

    def _plot(self, **kw):
        from numpy import asarray, atleast_1d, array_equal
        S = self.solutions
        if type(S)==list and len(S) == 0: return
        tmp = asarray(self.solutions.F if 'F' in dir(self.solutions) else self.solutions.values)
        from copy import deepcopy
        kw2 = deepcopy(kw)
        useShow = kw2.pop('show', True)
        if not useShow and hasattr(self, '_prev_mop_solutions') and array_equal(self._prev_mop_solutions, tmp):
            return
        self._prev_mop_solutions = tmp.copy()
        if tmp.size == 0:
            if self.isFinished:
                self.disp('no solutions, nothing to plot')
            return
        try:
            import pylab
        except:
            self.err('you should have matplotlib installed')
        pylab.ion()
        if self.nf != 2:
            self.err('MOP plotting is implemented for problems with only 2 goals, while you have %d' % self.nf)
        X, Y = atleast_1d(tmp[:, 0]), atleast_1d(tmp[:, 1])

        useGrid = kw2.pop('grid', 'on')
        
        if 'marker' not in kw2: 
            kw2['marker'] = (5, 1, 0)
        if 's' not in kw2:
            kw2['s']=[150]
        if 'edgecolor' not in kw2:
            kw2['edgecolor'] = 'b'
        if 'facecolor' not in kw2:
            kw2['facecolor'] = '#FFFF00'#'y'
            
        pylab.scatter(X, Y, **kw2)
        
        pylab.grid(useGrid)
        t0_goal = 'min' if self.targets[0].val == -inf else 'max' if self.targets[0].val == inf else str(self.targets[0].val)
        t1_goal = 'min' if self.targets[1].val == -inf else 'max' if self.targets[1].val == inf else str(self.targets[1].val)
        
        pylab.xlabel(self.user.f[0].name + ' (goal: %s    tolerance: %s)' %(t0_goal, self.targets[0].tol))
        pylab.ylabel(self.user.f[1].name + ' (goal: %s    tolerance: %s)' %(t1_goal, self.targets[1].tol))
        
        pylab.title('problem: %s    goal: %s' %(self.name, self.goal))
        figure = pylab.gcf()
        from openopt import __version__ as ooversion
        figure.canvas.set_window_title('OpenOpt ' + ooversion)
        
        pylab.hold(0)
        pylab.draw()
        if useShow: 
            pylab.ioff()
            pylab.show()

def mop_iterdraw(p):
    p._plot(show=False)

TkinterIsInstalled = True
import platform
if platform.python_version()[0] == '2': 
    # Python2
    try:
        from Tkinter import Tk
        from tkFileDialog import asksaveasfilename
    except:
        TkinterIsInstalled = False
else: 
    # Python3
    try:
        from tkinter import Tk
        from tkinter.filedialog import asksaveasfilename
    except:
        TkinterIsInstalled = False

def _export_to_xls(p, r, *args, **kw):
    try:
        import xlwt
    except:
        s = '''
        To export OpenOpt MOP result into xls file
        you should have Python module "xlwt" installed,
        (http://www.python-excel.org),
        available via easy_install xlwt 
        or Linux apt-get python-xlwt
        '''
        p.err(s)
    if len(args) != 0:
        xls_file = args[0]
    elif TkinterIsInstalled:
        root = Tk()
        root.withdraw()
        import os
        hd = os.getenv("HOME")
        xls_file = asksaveasfilename(defaultextension='.xls', initialdir = hd, filetypes = [('xls files', '.xls')])
        root.destroy()
        if xls_file in (None, ''): 
            return
    else:
        p.err('''
        you should either provide xls file name for data output 
        or have tkinter installed to set it via GUI window''')
        
#    xls_file = asksaveasfilename(defaultextension='.xls', initialdir = self.hd, filetypes = [('xls files', '.xls')])
#    if xls_file in (None, ''):
#        return

    nf = p.nf
    target_funcs = [t.func for t in p.targets]
    vars4export = set(p._freeVarsList).difference(target_funcs)
    vars4export = list(vars4export)
    vars4export.sort(key = lambda v: v._id)
    nv = len(vars4export)
    R = [[] for i in range(nv + nf)]
    Names = [t.name for t in target_funcs] + [v.name for v in vars4export]
    Keys = target_funcs + vars4export
    for elem in r.solutions:
        for i, key in enumerate(Keys):
            R[i].append(elem[key])
    from numpy import asarray
    R = asarray(R)
    
    L = len(r.solutions)
    
    wb = xlwt.Workbook()
    

    ws = wb.add_sheet('OpenOpt MOP result')
    from openopt import __version__ as ver
    i = 0
    ws.write(i, 0, 'OpenOpt ver')
    ws.write(i, 1, ver)
    i += 1
    ws.write(i, 0, 'Solver')
    ws.write(i, 1, p.solver.__name__)
    i += 1
    ws.write(i, 0, 'Prob name')
    ws.write(i, 1, p.name)
    i += 1
    ws.write(i, 0, 'Prob type')
    ws.write(i, 1, p.probType)
    i += 1
    ws.write(i, 0, 'Time, s')
    ws.write(i, 1, str(int(r.elapsed['solver_time'])))
    i += 1
    ws.write(i, 0, 'CPU Time, s')
    ws.write(i, 1, str(int(r.elapsed['solver_cputime'])))
    i += 1
    ws.write(i, 0, 'N solutions')
    ws.write(i, 1, str(L))
       
    style1 = xlwt.easyxf("""
         font:
             name Times New Roman,
             colour_index black;
         pattern:
             back_colour yellow,
             pattern thick_forward_diag,
             fore-colour yellow
         """) 
    for i in range(nf):
        ws.write(0, 3+i, Names[i], style1)
        for j in range(L):
            ws.write(1+j, 3+i, R[i, j], style1)

    for i in range(nf, nf + nv):
        ws.write(0, 3+i, Names[i])
        for j in range(L):
            ws.write(1+j, 3+i, R[i, j])


    wb.save(xls_file)
    p.disp('export MOP %s result to xls file finished' % p.name)
    
class target:
    pass
