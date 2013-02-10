''' LCPSolve(M,q): procedure to solve the linear complementarity problem:

       w = M z + q
       w and z >= 0
       w'z = 0

   The procedure takes the matrix M and vector q as arguments.  The
   procedure has three returns.  The first and second returns are
   the final values of the vectors w and z found by complementary
   pivoting.  The third return is a 2 by 1 vector.  Its first
   component is a 1 if the algorithm was successful, and a 2 if a
   ray termination resulted.  The second component is the value of
   the artificial variable upon termination of the algorithm.
   The third component is the number of iterations performed in the
   outer loop.
 
   Derived from: http://www1.american.edu/academic.depts/cas/econ/gaussres/optimize/quadprog.src
   (original GAUSS code by Rob Dittmar <dittmar@stls.frb.org> )

   Lemke's Complementary Pivot algorithm is used here. For a description, see:
   http://ioe.engin.umich.edu/people/fac/books/murty/linear_complementarity_webbook/kat2.pdf

Copyright (c) 2010 Rob Dittmar, Enzo Michelangeli and IT Vision Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

'''
   
from numpy import *

def LCPSolve(M,q, pivtol=1e-8): # pivtol = smallest allowable pivot element
    rayTerm = False
    loopcount = 0
    if (q >= 0.).all(): # Test missing in Rob Dittmar's code
        # As w - Mz = q, if q >= 0 then w = q and z = 0
        w = q
        z = zeros_like(q)
        retcode = 0.
    else:
        dimen = M.shape[0] # number of rows
        # Create initial tableau
        tableau = hstack([eye(dimen), -M, -ones((dimen, 1)), asarray(asmatrix(q).T)])
        # Let artificial variable enter the basis
        basis = range(dimen) # basis contains a set of COLUMN indices in the tableau 
        locat = argmin(tableau[:,2*dimen+1]) # row of minimum element in column 2*dimen+1 (last of tableau)
        basis[locat] = 2*dimen # replace that choice with the row 
        cand = locat + dimen
        pivot = tableau[locat,:]/tableau[locat,2*dimen]
        tableau -= tableau[:,2*dimen:2*dimen+1]*pivot # from each column subtract the column 2*dimen, multiplied by pivot 
        tableau[locat,:] = pivot # set all elements of row locat to pivot
        # Perform complementary pivoting
        oldDivideErr = seterr(divide='ignore')['divide'] # suppress warnings or exceptions on zerodivide inside numpy
        while amax(basis) == 2*dimen:
            loopcount += 1
            eMs = tableau[:,cand]    # Note: eMs is a view, not a copy! Do not assign to it...
            missmask = eMs <= 0.
            quots = tableau[:,2*dimen+1] / eMs # sometimes eMs elements are zero, but we suppressed warnings...
            quots[missmask] = Inf # in any event, we set to +Inf elements of quots corresp. to eMs <= 0. 
            locat = argmin(quots)
            if abs(eMs[locat]) > pivtol and not missmask.all(): # and if at least one element is not missing 
                # reduce tableau
                pivot = tableau[locat,:]/tableau[locat,cand]
                tableau -= tableau[:,cand:cand+1]*pivot 
                tableau[locat,:] = pivot
                oldVar = basis[locat]
                # New variable enters the basis
                basis[locat] = cand
                # Select next candidate for entering the basis
                if oldVar >= dimen:
                    cand = oldVar - dimen
                else:
                    cand = oldVar + dimen
            else:
                rayTerm = True
                break
        seterr(divide=oldDivideErr) # restore original handling of zerodivide in Numpy
        # Return solution to LCP
        vars = zeros(2*dimen+1)
        vars[basis] = tableau[:,2*dimen+1]
        w = vars[:dimen]
        z = vars[dimen:2*dimen]    
        retcode = vars[2*dimen]
    # end if (q >= 0.).all() 
    
    if rayTerm:
        retcode = (2, retcode, loopcount)  # ray termination
    else:
        retcode = (1, retcode, loopcount)  # success
    return (w, z, retcode)

