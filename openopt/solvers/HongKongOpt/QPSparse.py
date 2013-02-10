"""
pname, e, Q, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None, c0 = MPSparse(filename)

Reads the description of a QP problem from a file in the extended
MPS format (QPS) described by Istvan Maros and Csaba Meszaros at:
http://www.sztaki.hu/~meszaros/public_ftp/qpdata/qpdata.ps

Returns a tuple (pname, e, Q, Aeq, beq, lb, ub, c0)
name: string
all others: numpy arrays

QPS Format:

The QP problems are assumed to be in the following form:

min f(x) = e'x + 1/2 x' Q x,  Q symmetric and positive semidefinite
                                       
subject to      Aeq x = beq,
                l <= x <= u.


After the BOUNDS section of MPS there is an new section introduced
by a QUADOBJ record followed by columnwise elements of Q; row and
columns are column names of A. Being the matrix symmetrical,
only lower triangular elements are listed.

---------------------------------------------------------------------
Field:    1           2          3         4         5         6
Columns:  2-3        5-12      15-22     25-36     40-47     50-61
0        1         2         3         4         5         6
1234567890123456789012345678901234567890123456789012345678901  << column
 11 22222222  33333333  444444444444   55555555  666666666666  << field
---------------------------------------------------------------

          NAME   problem name

          ROWS

           type     name

          COLUMNS
                   column       row       value     row      value
                    name        name                name
          RHS
                    rhs         row       value     row      value
                    name        name                name
          RANGES
                    range       row       value     row      value
                    name        name                name
          BOUNDS

           type     bound       column     value
                    name        name
          ENDATA
---------------------------------------------------------------------


---------------------------------------------------------------
NAME          QP example
ROWS
 N  obj
 G  r1
 L  r2
COLUMNS
 11 22222222  33333333  444444444444   55555555  666666666666  << field
    c1        r1                 2.0   r2                -1.0
    c1        obj                1.5
    c2        r1                 1.0   r2                 2.0
    c2        obj               -2.0
RHS
    rhs1      r1                 2.0   r2                 6.0
BOUNDS
 UP bnd1      c1                20.0
QUADOBJ
    c1        c1                 8.0
    c1        c2                 2.0
    c2        c2                10.0
ENDATA
---------------------------------------------------------------



"""
from numpy import *

def QPSparse(filename):
    pname = e = Q = A = b = Aeq = beq = lb = ub = None  # defaults for ret. params
    c0 = 0.
    f = open(filename, 'r')
    section = None
    rowtype = {}    # dict of row type corresponding to a given row name 
    colnum = {}     # dict of col number corresponding to a given col name
    colcnt = 0      # counter of columns
    acnt = 0        # counter of inequalities (up to size of A matrix)
    aeqcnt = 0      # counter of equalities (up to size of Aeq matrix)
    aindx = {}      # dict of row number in A matrix corresponding to a given row name
    aeqindx = {}    # dict of row number in Aeq matrix corresponding to a given row name
    objcoef = {}    # dict of coefficient in obj function corresponding to a given column name
    RHSdic = {}     # dict containing {row_name, RHSvalue}
    RANGESdic = {}   # dict containing {row_name, RANGEvalue}
    ucnt = 0
    qcnt = 0
    lineno = 0
    for line in f:
        lineno += 1
        line = line.upper().strip("\n\r")
        f1 = line[1:3].strip()
        f2 = line[4:12].strip()
        f3 = line[14:22].strip()
        f4 = line[24:36].strip().replace('D','E')
        f4 = 0. if f4 == "" else float(f4)
        f5 = line[39:47].strip()
        f6 = line[49:61].strip().replace('D','E')
        f6 = 0. if f6 == "" else float(f6)
        
        if line[0] != ' ': # indicator record, switches current section
        
            # see which section is being closed, and take appropriate action
            if section == 'ROWS':  # being closed
                # we know how many rows we have. Allocate lists of lists for A, Aeq
                # Alist[n][colname] contains coeff for row n and col name colname in Aeq
                Alist = [{} for i in range(acnt)] 
                # Aeqlist[n][colname] contains coeff for row n and col name colname in Aeq
                Aeqlist = [{} for i in range(aeqcnt)]
            elif section == 'COLUMNS':  # being closed
                # we know how any columns we have (colcnt). So we can now build Q, ub and lb
                Q = zeros((colcnt,colcnt))
                ub = array([Inf]*colcnt)
                lb = zeros(colcnt)
            elif section == 'RHS':      # being closed
                pass    # b, beq and c0 have been already set up
            elif section == 'RANGES':   # being closed
                pass    # TODO: add ranges
            elif section == 'BOUNDS':   # being closed
                if ucnt == 0:
                    ub = None
            elif section == 'QUADOBJ':  # being closed
                if qcnt == 0:
                    Q = None
                    
            # set the section indicator according to the new section    
            if f1 == 'AM':
                section = 'NAME'    # being opened
            elif f1 == 'OW':
                section = 'ROWS'    # being opened
            elif f1 == 'OL':
                section = 'COLUMNS' # being opened
            elif f1 == 'HS': 
                section = 'RHS'     # being opened
            elif f1 == 'AN': 
                section = 'RANGES'  # being opened
            elif f1 == 'OU': 
                section = 'BOUNDS'  # being opened
            elif f1 == 'UA':
                section = 'QUADOBJ' # being opened
            elif f1 == 'ND':
                section = None
                break;
            else:
                f.close()
                raise(ValueError('invalid indicator record in line '+str(lineno)+': "'+line+'"'))
        elif section == 'NAME':
            pname = f3
        elif section == 'ROWS':
            rname = f2
            if f1 == 'N':
                rowtype[rname] = 'N'
                obj = rname
            elif f1 == 'G':
                rowtype[rname] = 'G'
                aindx[rname] = acnt
                acnt += 1
            elif f1 == 'L':
                rowtype[rname] = 'L'
                aindx[rname] = acnt
                acnt += 1
            elif f1 == 'E':
                rowtype[rname] = 'E'
                aeqindx[rname] = aeqcnt
                aeqcnt += 1
            else:
                f.close()
                raise(ValueError('invalid row type "'+f1+'" in line '+str(lineno)+': "'+line+'"'))
        elif section == 'COLUMNS':
            cname = f2
            rnames = [0,0]
            vals = [0,0]
            if cname not in colnum:
                colnum[cname] = colcnt  # alocate a new column number
                colcnt += 1
            rnames[0] = f3
            vals[0] = f4
            rnames[1] = f5
            vals[1] = f6
            for i in (0,1): # both are possible
                rn = rnames[i]
                value = vals[i]
                if rn == '':
                    break
                if rn == obj: # then value is the coefficient of col cname in the obj function
                    objcoef[cname] = value
                elif rowtype[rn] == 'L': #
                    Alist[aindx[rn]][cname] = value
                elif rowtype[rn] == 'G': #
                    Alist[aindx[rn]][cname] = -value
                elif rowtype[rn] == 'E': #
                    Aeqlist[aeqindx[rn]][cname] = value
        elif section == 'RHS':
            # What is the RHS name for????
            rnames = [0,0]
            vals = [0,0]
            rnames[0] = f3
            vals[0] = f4
            rnames[1] = f5
            vals[1] = f6
            for i in (0,1): # both are possible
                rname = rnames[i]
                value = vals[i]
                if rname == '':
                    break
                RHSdic[rname] = value
        elif section == 'RANGES':
            # What is the RANGE name for????
            rnames = [0,0]
            vals = [0,0]
            rnames[0] = f3
            vals[0] = f4
            rnames[1] = f5
            vals[1] = f6
            for i in (0,1): # both are possible
                rname = rnames[i]
                value = vals[i]
                if rname == '':
                    break
                RANGESdic[rname] = value
        elif section == 'BOUNDS':
            # by default, x >= 0
            # what is the Bound name in f2 for??? 
            # UP : x <= b, x >= 0
            # LO :         x >= b
            # FX : x == b
            # FR : (no bound: remove default >= 0)
            # MI : x > -Inf
            # BV : x in (0, 1) # NOT SUPPORTED
            ic = colnum[f3]
            val = f4
            if f1 == 'UP':
                ub[ic] = f4
                ucnt += 1
            elif f1 == 'LO':
                lb[ic] = f4
            elif f1 == 'FX':
                # TODO add an equality constraint
                raise(ValueError('fixed variable (FX) bound not supported in line '+str(lineno)+': "'+line+'"'))
            elif f1 == 'FR':
                lb[ic] = -Inf
                ub[ic] = Inf
            elif f1 == 'MI':
                lb[ic] = -Inf
            elif f1 == 'BV':
                raise(ValueError('binary value (BV) bound not supported in line '+str(lineno)+': "'+line+'"'))
                
        elif section == 'QUADOBJ':
            ic1 = colnum[f2]
            ic2 = colnum[f3]
            val = f4
            Q[ic1,ic2] = val
            if ic1 != ic2:    # if not on diagonal
                Q[ic2,ic1] = val
            qcnt += 1
    f.close()
    if section != None:
        raise(EOFError('unexpected EOF while in section '+section))
        
   # Now we have all necessary info and we can build A,b,Aeq and Beq      
    if acnt > 0:
        A = zeros((acnt, colcnt))
        b = zeros(acnt)
        for rn in range(acnt):
            for c in Alist[rn]:
                A[rn, colnum[c]] = Alist[rn][c]
    if aeqcnt > 0:
        Aeq = zeros((aeqcnt, colcnt))
        beq = zeros(aeqcnt)
        for rn in range(aeqcnt):
            for c in Aeqlist[rn]:
                Aeq[rn, colnum[c]] = Aeqlist[rn][c]
        
    # ########## process RHS
    for rname in RHSdic:
        value = RHSdic[rname]
        rt = rowtype[rname]
        if rt == 'L': #
            b[aindx[rname]] = value  # constant term in obj function
        elif rt == 'G': #
            b[aindx[rname]] = -value
        elif rt == 'E': #
            beq[aeqindx[rname]] = value
        elif rt == 'N':
            c0 = -value # constant term in obj function
        
    # Handle RANGE lines. Each range causes a duplicate of a 
    # row in A and b, and a modification of original and new b.
    """
    D. The RANGES section is for constraints of the form:  h <= constraint <= u .
       The range of the constraint is  r = u - h .  The value of r is specified
       in the RANGES section, and the value of u or h is specified in the RHS
       section.  If b is the value entered in the RHS section, and r is the
       value entered in the RANGES section, then u and h are thus defined:

            row type       sign of r       h          u
            ----------------------------------------------
               L            + or -       b - |r|      b
               G            + or -         b        b + |r|
               E              +            b        b + |r|
               E              -          b - |r|      b
    """
    if A != None:
        addA = zeros((len(RANGESdic),A.shape[1]))
        addb = zeros(len(RANGESdic))

        for rname in RANGESdic:
            index = aindx[rname]    # row index in A and index in b 
            value = RANGESdic[rname]
            rt = rowtype[rname]
            if rt == 'L': #
                b1 = b[index] + abs(value) # sign???
            elif rt == 'G': #
                b1 = b[index] - abs(value) # sign???
            elif rt == 'E': #
                raise(ValueError('RANGES for rows of type E not yet supported in line '+str(lineno)+': "'+line+'"'))
                #b1 = b[index] - value

            addA[index,:] = -A[index,:]    # append to A duplicate of index row, with sign changed
            addb[index]   = -b1            # append to b other extreme, with sign changed

        A = vstack([A, addA])
        b = concatenate([b, addb])     
    
    e = zeros(colcnt)
    for c in objcoef:
        e[colnum[c]] = objcoef[c]

    # suppress redundant lb/ub
    nvars = e.shape[0]
    if lb != None and (lb == array([-Inf]*nvars)).all():
        lb = None
    if ub != None and (ub == array([Inf]*nvars)).all():
        ub = None
        
    return (pname, e, Q, A, b, Aeq, beq, lb, ub, c0)


