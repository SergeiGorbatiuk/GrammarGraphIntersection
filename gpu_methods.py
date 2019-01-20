from numba import cuda
import numpy as np
import math

threadsperblock = (16, 16)
blockspergrid = None
is_changed = cuda.device_array((1,), dtype=bool)

@cuda.jit
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = False
        for k in range(A.shape[1]):
            tmp = tmp or (A[row, k] and B[k, col])
        if tmp and not C[row, col]:
            is_changed[0] = True
            C[row, col] = tmp


def matrices_to_gpu(matrices):
    for nonterminal, matrix in matrices.items():
        assert str(matrix.dtype) == 'bool', 'Matrix dtype is {}'.format(matrix.dtype)
        matrices[nonterminal] = cuda.to_device(matrix)
    return matrices


def matrices_from_gpu(matrices):
    for nonterminal, matrix in matrices.items():
        assert str(matrix.dtype) == 'bool', 'Matrix dtype is {}'.format(matrix.dtype)
        matrices[nonterminal] = matrix.copy_to_host()
    return matrices


def update_matrix_gpu(head_mat, body_first_mat, body_second_mat):
    # new_matrix = cuda.device_array(head_mat.shape, dtype=bool)
    is_changed[0] = False
    blockspergrid_x = int(math.ceil(body_first_mat.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(body_second_mat.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    matmul[blockspergrid, threadsperblock](body_first_mat, body_second_mat, head_mat)
    return head_mat if is_changed.copy_to_host() else None
