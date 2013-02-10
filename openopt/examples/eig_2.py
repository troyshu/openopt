# An example of OpenOpt EIG, see http://openopt.org/EIG for more examples and details
from openopt import EIG

# create a 5 x 5 matrix
import numpy.random as nr
nr.seed(0)
N = 5
A = nr.rand(N, N) 

#define prob
p = EIG(A) 
#solve
r = p.solve('numpy_eig') # solver numpy.linalg.eig will be used 

print(r.eigenvalues) # [ 2.89776724+0.j  -0.65372843+0.j 0.14607289+0.19602952j  0.14607289-0.19602952j -0.08530815+0.j]
# for i-th eigenvalue r.eigenvectors[:,i]  is corresponding vector, 
# as well as it is done for numpy/scipy functions
print(r.eigenvectors) 
'''
[[ 0.43733688+0.j         -0.19592536+0.j          0.57285154+0.j
   0.57285154+0.j          0.63764724+0.j        ]
 [ 0.49662623+0.j          0.03219327+0.j         -0.14013112+0.23938241j
  -0.14013112-0.23938241j -0.53642409+0.j        ]
 [ 0.42977207+0.j          0.55544796+0.j         -0.17419089+0.24907549j
  -0.17419089-0.24907549j  0.29171743+0.j        ]
 [ 0.38727512+0.j         -0.62338178+0.j         -0.42011495-0.27666898j
  -0.42011495+0.27666898j -0.45403266+0.j        ]
 [ 0.47687818+0.j          0.51327338+0.j          0.48015310-0.13758665j
   0.48015310+0.13758665j  0.12004364+0.j        ]]
 '''
