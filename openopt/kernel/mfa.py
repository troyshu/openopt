#!/usr/bin/env python

from numpy import inf, copy, floor, log10, asfarray, asscalar, argsort

TkinterIsInstalled = True
try:
    from Tkinter import Tk, Button, Entry, Label, Frame, StringVar, DISABLED, END, IntVar, \
    Radiobutton, Canvas
    from tkFileDialog import asksaveasfilename, askopenfile 
    from tkMessageBox import showerror
except ImportError:
    TkinterIsInstalled = False

xtolScaleFactor = 1e-5

class mfa:
    filename = None # Python pickle file where to save results
    x0 = None
    solved = False
    def startSession(self):
        assert TkinterIsInstalled, '''
        Tkinter is not installed. 
        If you have Linux you could try using 
        "apt-get install python-tk"'''
        try:
            import nlopt
        except ImportError:
            s = '''
            To use OpenOpt multifactor analysis tool 
            you should have nlopt with its Python API installed,
            see http://openopt.org/nlopt'''
            print(s)
            showerror('OpenOpt', s)
            raw_input()
            return
        import os
        hd = os.getenv("HOME")
        self.hd = hd
        root = Tk()
        self.root = root
        from openopt import __version__ as oover
        root.wm_title(' OpenOpt %s Multifactor analysis tool for experiment planning ' % oover)
        SessionSelectFrame = Frame(root)
        SessionSelectFrame.pack(side='top', padx=230, ipadx = 40, fill='x', expand=True)
        var = StringVar()
        var.set('asdf')
        Radiobutton(SessionSelectFrame, variable = var, text = 'New', indicatoron=0, \
                    command=lambda: (SessionSelectFrame.destroy(), self.create())).pack(side = 'top', fill='x', pady=5)
        Radiobutton(SessionSelectFrame, variable = var, text = 'Load', indicatoron=0,  \
                    command=lambda:self.load(SessionSelectFrame)).pack(side = 'top', fill='x', pady=5)
        
        root.protocol("WM_DELETE_WINDOW", self.exit)

        root.mainloop()
        self.exit()

    
    def create(self, S={}):
        root = self.root
        
#        #BackgroundFile = Image(file='/home/dmitrey/tmp_i/Backgd01.jpg')
#        bgfile = '/home/dmitrey/tmp_i/Backgd01.gif'
#        #bgfile = '/home/dmitrey/IP.png'
#        BackgroundFile = PhotoImage(file=bgfile)
#        #RootFrame.create_image(0, 0, image=BackgroundFile)
        RootFrame = Canvas(root)#, image=BackgroundFile)
#        RootFrame.create_image(0, 0, image=BackgroundFile)
#        RootFrame.image = BackgroundFile
        RootFrame.pack()
        
        self.NameEntriesList, self.LB_EntriesList, self.UB_EntriesList, self.TolEntriesList, self.ValueEntriesList = [], [], [], [], []
        self.calculated_points = S.get('calculated_points', [])

        # Title
        #root.wm_title(' FuncDesigner ' + fdversion + ' Manager')
        
        C = Canvas(root)
        
        """                                              Buttons                                               """
        Frame(RootFrame).pack(ipady=4)
        #Label(root, text=' FuncDesigner ' + fdversion + ' ').pack()


        #                                                   Upper Frame
        UpperFrame = Frame(RootFrame)
        
        
        ProjectNameFrame = Frame(UpperFrame)#, relief = 'ridge', bd=2)
        Label(ProjectNameFrame,  text = 'Project name:').pack(side = 'left')
        ProjectNameEntry = Entry(ProjectNameFrame)
        ProjectNameEntry.pack(side = 'left')
        self.ProjectNameEntry = ProjectNameEntry
        ProjectNameFrame.pack(side = 'left')
        
        GoalSelectFrame = Frame(UpperFrame, relief = 'ridge', bd=2)
        GoalSelectText = StringVar(value = 'Goal:')
        Label(GoalSelectFrame, textvariable = GoalSelectText).pack(side = 'left')
        goal = StringVar()
        r1 = Radiobutton(GoalSelectFrame, text = 'Minimum', value = 'min', variable=goal)
        r1.pack(side = 'left')
        r2 = Radiobutton(GoalSelectFrame, text = 'Maximum', value = 'max', variable=goal)
        r2.pack(side = 'left')
        goal.set('min')    
        GoalSelectFrame.pack(side = 'left', padx = 10)
        self.goal = goal
        
        ObjectiveToleranceFrame = Frame(UpperFrame, relief = 'ridge', bd=2)
        ObjectiveToleranceFrame.pack(side='left')
        Label(ObjectiveToleranceFrame, text='Objective function tolerance:').pack(side = 'left')
        ObjTolEntry = Entry(ObjectiveToleranceFrame)
        ObjTolEntry.pack(side='left')
        self.ObjTolEntry = ObjTolEntry
        
        UpperFrame.pack(side = 'top', expand=True, fill = 'x')
        
        #                                                   Variables Frame
        varsRoot = Frame(RootFrame)
       
       
        #                                                    Lower frame
        LowerFrame = Frame(varsRoot)
        LowerFrame.pack(side = 'bottom', expand=True, fill = 'x')

        from webbrowser import open_new_tab
        About = Button(LowerFrame, text = 'About', command = lambda: open_new_tab('http://openopt.org/MultiFactorAnalysis'))
        About.pack(side='left')
        
        SaveButton = Button(LowerFrame, text = 'Save', command = self.save)
        SaveButton.pack(side='left', padx = 15)
        SaveAsButton = Button(LowerFrame, text = 'Save As ...', command = self.save)
        SaveAsButton.pack(side='left')
        Write_xls_Button = Button(LowerFrame, text = 'Write xls report', command = self.write_xls_report)
        Write_xls_Button.pack(side='left', padx = 15)
        
        
        
       
    #    PlotButton = Button(LowerFrame, text = 'Plot', command = lambda: Plot(C, self.prob))
    #    PlotButton.pack(side='left')
     
        ExperimentNumber = IntVar()
        ExperimentNumber.set(1)
        self.ExperimentNumber = ExperimentNumber
       
        ObjVal = StringVar()
        ObjEntry = Entry(LowerFrame, textvariable = ObjVal)
        self.ObjEntry = ObjEntry

        NN = StringVar(LowerFrame)
        NN_Label = Label(LowerFrame, textvariable = NN)
        
        
        names, lbs, ubs, tols, currValues = \
        Frame(varsRoot), Frame(varsRoot), Frame(varsRoot), Frame(varsRoot), Frame(varsRoot)
        Label(names, text=' Variable Name ').pack(side = 'top')
        Label(lbs, text=' Lower Bound ').pack(side = 'top')
        Label(ubs, text=' Upper Bound ').pack(side = 'top')
        Label(tols, text=' Tolerance ').pack(side = 'top')
        
        ValsColumnName = StringVar()
        ValsColumnName.set(' Initial Point ')
        Label(currValues, textvariable=ValsColumnName).pack(side = 'top')
        self.ValsColumnName = ValsColumnName
        
        
        #                                                    Commands Frame
        CommandsRoot = Frame(RootFrame)
        CommandsRoot.pack(side = 'right', expand = False, fill='y')
        
       
        AddVar = Button(CommandsRoot, text = 'Add Variable', command = \
                        lambda: self.addVar(names, lbs, ubs, tols, currValues))
        AddVar.pack(side = 'top', fill='x')

        Next = Button(CommandsRoot, text = 'Next', command = lambda: ExperimentNumber.set(ExperimentNumber.get()+1))
        #Next.pack(side='bottom',  fill='x')

        names.pack(side = 'left', ipady=5)
        lbs.pack(side = 'left', ipady=5)
        ubs.pack(side = 'left', ipady=5)
        tols.pack(side = 'left', ipady=5)
        currValues.pack(side = 'left', ipady=5)
        #currValues.pack_forget()
        
        varsRoot.pack()
        
        Start = Button(CommandsRoot, text = 'Start', \
                       command = lambda: (Start.destroy(), \
                                          Next.pack(side='bottom',  fill='x'), 
                                          #C.pack(side = 'bottom', expand=True, fill='both'), 
                                          r1.config(state=DISABLED), 
                                          r2.config(state=DISABLED), 
                                          ObjTolEntry.config(state=DISABLED), 
                                          ObjEntry.pack(side='right', ipady=4),
                                          NN_Label.pack(side='right'), \
                                          self.startOptimization(root, varsRoot, AddVar, currValues, ValsColumnName, ObjEntry, ExperimentNumber, Next, NN, 
                                                            goal.get(), float(ObjTolEntry.get()), C)))
        Start.pack(side = 'bottom', fill='x')
        self.Start = Start
        
        if len(S) != 0:
            for i in range(len(S['names'])):
                tmp = S['values'][i] if self.x0 is None else self.x0.split(' ')[i]
                self.addVar(names, lbs, ubs, tols, currValues, S['names'][i], S['lbs'][i], S['ubs'][i], S['tols'][i], tmp)
        else:
            self.addVar(names, lbs, ubs, tols, currValues)
#        for i in range(nVars):
#            self.addVar(names, lbs, ubs, tols, currValues)
        
        

    def addVar(self, names, lbs, ubs, tols, currValues, _name='', _lb='', _ub='', _tol='', _val=''):
        nameEntry, lb, ub, tol, valEntry = Entry(names), Entry(lbs), Entry(ubs), Entry(tols), Entry(currValues)
        nameEntry.insert(0, _name)
        lb.insert(0, _lb)
        ub.insert(0, _ub)
        tol.insert(0, _tol)
        valEntry.insert(0, _val)
        self.NameEntriesList.append(nameEntry)
        self.LB_EntriesList.append(lb)
        self.UB_EntriesList.append(ub)
        self.TolEntriesList.append(tol)
        self.ValueEntriesList.append(valEntry)
        nameEntry.pack(side = 'top')
        lb.pack(side = 'top')
        ub.pack(side = 'top')
        tol.pack(side = 'top')
        valEntry.pack(side = 'top')

    def startOptimization(self, root, varsRoot, AddVar, currValues, \
                          ValsColumnName, ObjEntry, ExperimentNumber, Next, NN, goal, objtol, C):
        AddVar.destroy()
        ValsColumnName.set('Experiment parameters')
        n = len(self.NameEntriesList)
        Names, Lb, Ub, Tol, x0 = [], [], [], [], []
        for i in range(n):
            N, L, U, T, valEntry = \
            self.NameEntriesList[i], self.LB_EntriesList[i], self.UB_EntriesList[i], self.TolEntriesList[i], self.ValueEntriesList[i]
            N.config(state=DISABLED)
            L.config(state=DISABLED)
            U.config(state=DISABLED)
            T.config(state=DISABLED)
            #valEntry.config(state=DISABLED)
            name, lb, ub, tol, val = N.get(), L.get(), U.get(), T.get(), valEntry.get()
            Names.append(name)
            x0.append(float(val))
            Lb.append(float(lb) if lb != '' else -inf)
            Ub.append(float(ub) if ub != '' else inf)
            
            # TODO: fix zero
            Tol.append(float(tol) if tol != '' else 0) 
            
        x0, Tol, Lb, Ub = asfarray(x0), asfarray(Tol), asfarray(Lb), asfarray(Ub)
        x0 *= xtolScaleFactor / Tol
        #self.x0 = copy(x0)
        from openopt import NLP, oosolver
        p = NLP(objective, x0, lb = Lb * xtolScaleFactor / Tol, ub=Ub * xtolScaleFactor / Tol)
        self.prob = p
        #calculated_points = [(copy(x0), copy(float(ObjEntry.get())))
        p.args = (Tol, self, ObjEntry, p, root, ExperimentNumber, Next, NN, objtol, C)
        #p.graphics.rate = -inf
        #p.f_iter = 2
        solver = oosolver('bobyqa', useStopByException = False)
        p.solve(solver, iprint = 1, goal = goal)#, plot=1, xlabel='nf')
        self.solved = True
        if p.stopcase >= 0:
            self.ValsColumnName.set('Best parameters')
            NN.set('Best obtained objective value:')
        #Next.config(state=DISABLED)
        Next.destroy()
        #reverse = True if goal == 'min' else False
        
        calculated_items = self.calculated_points.items() if isinstance(self.calculated_points, dict) else self.calculated_points
        vals = [calculated_items[i][1] for i in range(len(calculated_items))]
        ind = argsort(vals)
        j = ind[0] if goal == 'min' else ind[-1]
        key, val = calculated_items[j]
        text_coords = key.split(' ')
        for i in range(len(self.ValueEntriesList)):
            self.ValueEntriesList[i].delete(0, END)
            self.ValueEntriesList[i].insert(0, text_coords[i])
        ObjEntry.delete(0, END)
        obj_tol = self.ObjTolEntry.get()
        val = float(val) * 1e4 * objtol
        ObjEntry.insert(0, str(val))
        ObjEntry.config(state=DISABLED)
        #print('Finished')

    def Plot(C, p):
        pass
        #C.create_polygon()
    #    import os
    #    if os.fork():
    #        import pylab
    #        pylab.plot(p.iterValues.f)
    #        pylab.show()
    
    def load(self, SessionSelectFrame):
        file = askopenfile(defaultextension='.pck', initialdir = self.hd, filetypes = [('Python pickle files', '.pck')])
        if file in (None, ''):
            return
        SessionSelectFrame.destroy()
        import pickle 
        S = pickle.load(file)
        if type(S['calculated_points']) == dict: # for backward compatibility
            S['calculated_points'] = S['calculated_points'].items()
        #S['goal']='max'
        self.x0 = S.get('x0', None) # this line should be BEFORE self.create(S)
        
        self.create(S)
        
        self.ObjTolEntry.insert(0, S['ObjTol'])
        self.ProjectNameEntry.insert(0, S.get('ProjectName', ''))
        self.goal.set(S['goal'])
        self.ExperimentNumber.set(len(self.calculated_points))
        if len(S['calculated_points']) != 0: 
            self.Start.invoke()
        
    
    def save_as(self, filename=None):
        if filename is None:
            filename = asksaveasfilename(defaultextension='.pck', initialdir = self.hd, filetypes = [('Python pickle files', '.pck')])
        if filename in (None, ''):
            return
        if not self.solved and self.ObjEntry.get() != '':
            s = 'For the sake of more safety and some other circumstances saving with non-empty objective entry is forbidden'
            print(s)
            showerror('OpenOpt', s)
            return
        self.filename = filename
        names = [s.get() for s in self.NameEntriesList]
        lbs = [s.get() for s in self.LB_EntriesList]
        ubs = [s.get() for s in self.UB_EntriesList]
        tols = [s.get() for s in self.TolEntriesList]
        values = [s.get() for s in self.ValueEntriesList]
        goal = self.goal.get()
        ObjTol = self.ObjTolEntry.get()
        calculated_points = self.calculated_points
        ProjectName = self.ProjectNameEntry.get()
        S = {'names':names, 'lbs':lbs, 'ubs':ubs, 'tols':tols, 'values':values, 'goal':goal, \
        'ObjTol':ObjTol, 'calculated_points':calculated_points, 'ProjectName':ProjectName, 'x0':self.x0}
        
        # TODO: handle exceptions
        file = open(filename, "w")
        import pickle
        pickle.dump(S, file)
        file.close()
        
    save = lambda self: self.save_as(self.filename)    
    
    def write_xls_report(self):
        try:
            import xlwt
        except ImportError:
            s = '''To create xls reports 
            you should have xlwt installed, 
            see http://www.python-excel.org/
            you could use easy_install xlwt (with admin rights)
            also, in Linux you could use 
            [sudo] aptitude install python-xlwt
            '''
            print(s)
            showerror('OpenOpt', s)
            return
            
        xls_file = asksaveasfilename(defaultextension='.xls', initialdir = self.hd, filetypes = [('xls files', '.xls')])
        if xls_file in (None, ''):
            return
            
        wb = xlwt.Workbook()
        ws = wb.add_sheet('OpenOpt factor analysis report')
        ws.write(0, 0, 'Name')
        ws.write(0, 1, self.ProjectNameEntry.get())
        ws.write(1, 0, 'Goal')
        ws.write(1, 1, self.goal.get() + 'imum')
        ws.write(2, 0, 'Objective Tolerance')
        ws.write(2, 1, self.ObjTolEntry.get())
        
        names = [s.get() for s in self.NameEntriesList]
        lbs = [s.get() for s in self.LB_EntriesList]
        ubs = [s.get() for s in self.UB_EntriesList]
        tols = [s.get() for s in self.TolEntriesList]
        
        ws.write(4, 0, 'Variable')
        ws.write(5, 0, 'Lower Bound')
        ws.write(6, 0, 'Upper Bound')
        ws.write(7, 0, 'Tolerance')
        for i in range(len(names)):
            ws.write(4, i+1, names[i])
            ws.write(5, i+1, float(lbs[i]))
            ws.write(6, i+1, float(ubs[i]))
            ws.write(7, i+1, float(tols[i]))
        
        
        ws.write(9, 0, 'Exp Number')
        ws.write(9, len(names)+1, 'Objective')
        # TODO: minor code cleanup
        CP = self.calculated_points
        if isinstance(CP, dict):
            CP = CP.items()
        for i in range(len(CP)):
            key,  val = CP[i]
            ws.write(10+i, 0, i+1)
            coords = key.split()
            for j, coordVal in enumerate(coords):
                ws.write(10+i, j+1, float(coordVal))
            ws.write(10+i, len(coords)+1, float(val))
        wb.save(xls_file)
    
    def exit(self):
        try:
            self.root.quit()
        except:
            pass
        try:
            self.root.destroy()
        except:
            pass

def objective(x, Tol, mfa, ObjEntry, p, root, ExperimentNumber, Next, NN, objtol, C):
    #print 'in objective', x
    Key = ''
    Values = []
    ValueEntriesList = mfa.ValueEntriesList
    calculated_points = mfa.calculated_points
    for i in range(x.size):
        Format = '%0.9f' if Tol[i] == 0 else ('%0.' + ('%d' % (-floor(log10(Tol[i])))) + 'f') if Tol[i]<1 else '%d'
        tmp = x[i] * Tol[i] / xtolScaleFactor
        key = Format % tmp
        Key += key + ' '
        Values.append(key)
    #print calculated_points
    #print Key
    if mfa.x0 is None: mfa.x0 = Key
    if Key in dict(calculated_points): # TODO: rework it
        return dict(calculated_points)[Key]
   
    for i in range(x.size):
        ValueEntriesList[i].delete(0, END)
        ValueEntriesList[i].insert(0, Values[i])
    NN.set('Enter experiment %i result:' % int(len(calculated_points)+1))
    
    ObjEntry.delete(0, END)
    root.wait_variable(ExperimentNumber)
    r = float(ObjEntry.get()) 
    
#    from scipy import rand
#    r = abs(x[0]* Tol[0] / xtolScaleFactor-0.13) + abs(x[1]* Tol[1] /xtolScaleFactor-0.15) #+ 0.0001 * rand(1)

    r *= 1e-4 / objtol
    if isinstance(calculated_points, list):
        calculated_points.append((Key, asscalar(copy(r)))) 
    else: # dict
        calculated_points[Key] = asscalar(copy(r))
    
#    rr = []
#    for i, val in enumerate(p.iterValues.f):
#        rr.append(i)
#        rr.append(val)
#    rr.append(i+1)
#    rr.append(r)
#    if len(p.iterValues.f) > 1:
#        C.create_line(*rr, fill = 'blue')
    return r
    
MFA = lambda: mfa().startSession()
if __name__ == '__main__':
    MFA()
    
