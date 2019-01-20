import math

from numba import cuda, boolean
import numpy as np


TPB = 32

@cuda.jit
def boolean_mult(A, B, C):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=boolean)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=boolean)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x

    if x >= C.shape[0] or y >= C.shape[1]:
        return

    tmp = False
    bound = min(TPB, A.shape[0])
    for i in range(bpg):
        sA[tx, ty] = A[x, y]
        sB[tx, ty] = B[x, y]

        cuda.syncthreads()

        for j in range(bound):
            tmp = tmp or (sA[tx, j] and sB[j, ty])

        cuda.syncthreads()
    C[x, y] = tmp


A = np.random.randint(0, 2, (10, 10)).astype(np.bool)
B = np.random.randint(0, 2, (10, 10)).astype(np.bool)
C = np.zeros_like(A).astype(np.bool)

A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)
C_global_mem = cuda.to_device(C)

threadsperblock = (TPB, TPB)
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

boolean_mult[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
res = C_global_mem.copy_to_host()

print(np.sum(res ^ A.dot(B)))
