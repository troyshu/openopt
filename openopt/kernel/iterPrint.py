from numpy import atleast_1d, asfarray, log10
def signOfFeasible(p):
    r = '-'
    if p.isFeas(p.xk): r = '+'
    return r

textOutputDict = {\
'objFunVal': lambda p: p.iterObjFunTextFormat % (-p.fk if p.invertObjFunc else p.fk), 
'log10(maxResidual)': lambda p: '%0.2f' % log10(p.rk+1e-100), 
'log10(MaxResidual/ConTol)':lambda p: '%0.2f' % log10(max((p.rk/p.contol, 1e-100))), 
'residual':lambda p: '%0.1e' % p._Residual, 
'isFeasible': signOfFeasible, 
'nSolutions': lambda p: '%d' % p._nObtainedSolutions, 
'front length':lambda p: '%d' % p._frontLength, 
'outcome': lambda p: ('%+d' % -p._nOutcome if p._nOutcome != 0 else ''), 
'income': lambda p: ('%+d' % p._nIncome if p._nIncome != 0 else ''), 
}
delimiter = '   '

class ooTextOutput:
    def __init__(self):
        pass

    def iterPrint(self):

        if self.lastPrintedIter == self.iter: return

        if self.iter == 0 and self.iprint >= 0: # 0th iter (start)
            s = ' iter' + delimiter
            for fn in self.data4TextOutput:
                s += fn + delimiter
            print(s)
        elif self.iprint<0 or \
        (((self.iprint>0 and self.iter % self.iprint != 0) or self.iprint==0)  and not(self.isFinished or self.iter == 0)):
            return

        s = str(self.iter).rjust(5) + '  '
        for columnName in self.data4TextOutput:
            val = textOutputDict[columnName](self)
            #nWhole = length(columnName)
            s += val.rjust(len(columnName)) + ' '
        print(s)
        self.lastPrintedIter = self.iter
