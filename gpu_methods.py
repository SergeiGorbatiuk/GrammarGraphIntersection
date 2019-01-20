from numba import cuda
import math

threadsperblock = (16, 16)


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
    size = 8
    if row < C.shape[0] and col < C.shape[1]:
        value = C[row, col]*0
        for j in range(size):
            tmp = 0
            for k in range(A.shape[1]):
                tmp = tmp | (A[row, k] & B[k, col + j])
            tmp = (tmp != 0) * 1
            value |= (tmp << (size - 1 - j))
        value = value | C[row, col]
        if (value | C[row, col]) != C[row, col]:
            is_changed[0] = True
            C[row, col] |= value
            row_ind = row // size
            row_mask = 1<<(size - 1 - row_ind % size)
            for i in range(size):
                col_mask = 1<<(size - 1 - col)
                col_value = ((value & col_mask) > 0) * 255
                C_i[row_ind, col+i] = (row_mask & col_value)


@cuda.jit
def matmul_packed(A, B, C, is_changed):
    row, col = cuda.grid(2)
    size = 8
    if row >= C.shape[0] or col >= C.shape[1]:
        return
    value = 0
    for k in range(A.shape[1]):
        cur_value_A = A[row, k]
        for j in range(size):
            if cur_value_A & 1:
                value |= (B[k * size + j, col])
            cur_value_A >>= 1
    old_value = C[row, col]
    new_value = old_value | value
    if new_value != old_value:
        C[row, col] = new_value
        is_changed[0] = True


def matrices_to_gpu(matrices):
    for nonterminal, matrix in matrices.items():
        matrices[nonterminal] = cuda.to_device(matrix)
    return matrices


def matrices_from_gpu(matrices):
    for nonterminal, matrix in matrices.items():
        matrices[nonterminal] = matrix.copy_to_host()
    return matrices


def update_matrix_gpu(matrices, head, body, matrices_i=None):
    head_mat, body_first_mat = matrices[head], matrices[body[0]]
    body_second_mat = (matrices if matrices_i is None else matrices_i)[body[1]]
    is_changed = cuda.device_array((1,), dtype=bool)

    blockspergrid_x = int(math.ceil(body_first_mat.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(body_second_mat.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    if str(head_mat.dtype) == 'bool':
        matmul[blockspergrid, threadsperblock](body_first_mat, body_second_mat, head_mat, is_changed)
        if not is_changed[0]:
            return False
        matrices[head] = head_mat
        return True
    elif str(head_mat.dtype) == 'uint8' and matrices_i is not None:
        head_mat_packed_y = matrices_i[head]
        matmul_packed_both[blockspergrid, threadsperblock](body_first_mat, body_second_mat, head_mat, head_mat_packed_y, is_changed)
        if not is_changed[0]:
            return False
        matrices[head] = head_mat
        matrices_i[head] = head_mat_packed_y
        return True
    elif str(head_mat.dtype) == 'uint8':
        matmul_packed[blockspergrid, threadsperblock](body_first_mat, body_second_mat, head_mat, is_changed)
        print(is_changed[0])
        if not is_changed[0]:
            return False
        matrices[head] = head_mat
        return True
    else:
        raise ValueError('Type {} is not supported'.format(head_mat.dtype))

