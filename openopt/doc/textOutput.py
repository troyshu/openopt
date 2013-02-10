"""
You can manage text output in OpenOpt via the following prob parameters:
Prob field name           default value 
iprint                                10
iterObjFunTextFormat         '%0.3e'
finalObjFunTextFormat        '%0.8g'
 
iprint: do text output each iprint-th iteration
You can use iprint = 0 for final output only or iprint < 0 to omit whole output
In future warnings are intended to be shown if iprint >= -1. 
However, some solvers like ALGENCAN have their own text output system, that's hard to suppress, it requires using different approach like, for example, http://permalink.gmane.org/gmane.comp.python.scientific.user/15465

iterObjFunTextFormat: how iter output objFun values are represented
for example, '%0.3e' yields lines like 
 iter    objFunVal
    0  1.947e+03               
   10  1.320e+03            
   ...

finalObjFunTextFormat: how final output objFun value is represented
for example finalObjFunTextFormat='%0.1f' yields
...
objFunValue: 7.9

See Python language documentation for text format specification.
"""
