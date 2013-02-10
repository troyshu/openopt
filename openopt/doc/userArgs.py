"""
Example of using additional parameters for user f, c, h functions
Note! For oofun handling user parameters is performed 
in the same way: 
my_oofun.args = (...)
they will be passed to derivative function as well (if you have supplied it)
"""

from openopt import NLP
from numpy import asfarray

f = lambda x, a: (x**2).sum() + a * x[0]**4
x0 = [8, 15, 80]
p = NLP(f, x0)


#using c(x)<=0 constraints
p.c = lambda x, b, c: (x[0]-4)**2 - 1 + b*x[1]**4 + c*x[2]**4

#using h(x)=0 constraints
p.h = lambda x, d: (x[2]-4)**2 + d*x[2]**4 - 15
    
p.args.f = 4 # i.e. here we use a=4
# so it's the same to "a = 4; p.args.f = a" or just "p.args.f = a = 4" 

p.args.c = (1,2)

p.args.h = 15 

# Note 1: using tuple p.args.h = (15,) is valid as well

# Note 2: if all your funcs use same args, you can just use 
# p.args = (your args)

# Note 3: you could use f = lambda x, a: (...); c = lambda x, a, b: (...); h = lambda x, a: (...)

# Note 4: if you use df or d2f, they should handle same additional arguments;
# same to c - dc - d2c, h - dh - d2h

# Note 5: instead of myfun = lambda x, a, b: ...
# you can use def myfun(x, a, b): ...

r = p.solve('ralg')
"""
If you will encounter any problems with additional args implementation, 
you can use the simple python trick
p.f = lambda x: other_f(x, <your_args>)
same to c, h, df, etc
"""
