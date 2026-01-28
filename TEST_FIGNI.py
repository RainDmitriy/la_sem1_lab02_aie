from COO import COOMatrix
from type import *

C = COOMatrix([3,4,5], [0,1,2], [2,0,1], shape=(3,3))
Cd = C.to_dense()
print(Cd)
