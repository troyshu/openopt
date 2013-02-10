from openopt import EIG

# create a 5 x 5 matrix
import numpy.random as nr
nr.seed(0)
N = 5
A = nr.rand(N, N) 

#define prob
p = EIG(A, goal = {'lm':3}) # search for 3 eigenvalues of largest magnitude
# or goal={'largest magnitude':3}, with or without space inside, case-insensitive
# for whole list of available goals see http://openopt.org/EIG

#solve
r = p.solve('arpack') # arpack is name of the involved solver

print(r.eigenvalues) # [ 0.14607289-0.19602952j -0.65372843+0.j          2.89776724+0.j        ]
# for i-th eigenvalue r.eigenvectors[:,i]  is corresponding vector, 
# as well as it is done for numpy/scipy functions
print(r.eigenvectors) 
'''
[[-0.10391145-0.56334829j  0.19592536+0.j          0.43733688+0.j        ]
 [-0.20999235+0.1812288j  -0.03219327+0.j          0.49662623+0.j        ]
 [-0.21334642+0.21648181j -0.55544796+0.j          0.42977207+0.j        ]
 [ 0.34828527+0.36295959j  0.62338178+0.j          0.38727512+0.j        ]
 [ 0.04820760-0.49714496j -0.51327338+0.j          0.47687818+0.j        ]]
 '''
