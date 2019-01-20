from numba import cuda
import numpy as np
import math

threadsperblock = (16, 16)
blockspergrid = None
is_changed = cuda.device_array((1,), dtype=bool)


@cuda.jit
def matmul(A, B, C, is_changed):
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


@cuda.jit
def matmul_packed_both(A, B, C, C_i, is_changed):
    """Perform matrix multiplication of C = A * B
    with packed bits on both axis
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        value = 0
        for j in range(8):
            tmp = 0
            for k in range(A.shape[1]):
                if col + j >= B.shape[1]:
                    return
                tmp = tmp | (A[row, k] & B[k, col + j])
            value |= tmp << j
        if (value | C[row, col]) != C[row, col]:
            is_changed[0] = True
            C[row, col] |= value


def matrices_to_gpu(matrices):
    for nonterminal, matrix in matrices.items():
        # assert str(matrix.dtype) == 'bool', 'Matrix dtype is {}'.format(matrix.dtype)
        matrices[nonterminal] = cuda.to_device(matrix)
    return matrices


def matrices_from_gpu(matrices):
    for nonterminal, matrix in matrices.items():
        # assert str(matrix.dtype) == 'bool', 'Matrix dtype is {}'.format(matrix.dtype)
        matrices[nonterminal] = matrix.copy_to_host()
    return matrices


def update_matrix_gpu(head_mat, body_first_mat, body_second_mat, packed_y=False):
    blockspergrid_x = int(math.ceil(body_first_mat.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(body_second_mat.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    is_changed = cuda.device_array((1,), dtype=bool)
    if str(head_mat.dtype) == 'bool':
        matmul[blockspergrid, threadsperblock](body_first_mat, body_second_mat, head_mat, is_changed)
    elif str(head_mat.dtype) == 'uint8' and packed_y:
        matmul_packed_both[blockspergrid, threadsperblock](body_first_mat, body_second_mat, head_mat)
    else:
        raise ValueError('Type {} is not supported'.format(head_mat.dtype))
    return head_mat if is_changed[0] else None
