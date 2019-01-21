import numpy as np


def matrices_to_type(matrices, type, threadsperblock, axis=-1, shared_memory=False):
    if type in [np.uint8, 'uint8', 'byte']:
        for key, matrix in matrices.items():
            if shared_memory:
                size = 8 * threadsperblock[0]
                matrix = np.pad(matrix, [(0, (size-matrix.shape[0])%size), (0,(size-matrix.shape[1])%size)], 'constant')
            matrix = np.packbits(matrix, axis=axis)
            matrices[key] = matrix
    elif type in [np.uint32, 'uint32', 'int']:
        for key, matrix in matrices.items():
            if shared_memory:
                size = 32 * threadsperblock[0]
                matrix = np.pad(matrix, [(0, (size-matrix.shape[0])%size), (0,(size-matrix.shape[1])%size)], 'constant')
            matrix = np.pad(matrix, [(0, 0), (0, (32 - matrix.shape[1] % 32) % 32)], 'constant').astype(np.uint32)
            packed_matrix = sum(matrix[:, i::32] << (31 - i) for i in range(32))
            matrices[key] = packed_matrix
    else:
        raise ValueError('Casting to type {} is not supported yet'.format(type))
    return matrices


def matrices_from_type(matrices, type, nodes_amount, axis=-1):
    if type in [np.uint8, 'uint8', 'byte']:
        for key, matrix in matrices.items():
            matrix = np.unpackbits(matrix, axis=axis)[:nodes_amount, :nodes_amount]
            matrices[key] = matrix
    elif type in [np.uint32, 'uint32', 'int']:
        for key, matrix in matrices.items():
            full_matrix = np.zeros((matrix.shape[0], matrix.shape[1] * 32), dtype=bool)
            for i in range(32):
                full_matrix[:, i::32] = (matrix >> (31 - i)) & 1
            matrices[key] = full_matrix[:nodes_amount, :nodes_amount]
    else:
        raise ValueError('Casting to type {} is not supported yet'.format(type))
    return matrices


def get_boolean_adjacency_matrices(grammar, inv_grammar, graph, graph_size):
    size = graph_size
    matrices = {i: np.zeros((size, size), dtype=np.bool) for i in grammar}
    for row, verts in graph.items():
        for col, value in verts.items():
            if value in inv_grammar:
                for nonterminal in inv_grammar[value]:
                    matrices[nonterminal][row, col] = True
    return matrices
