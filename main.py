import argparse
import numpy as np
from gpu_methods import matrices_to_gpu, matrices_from_gpu, update_matrix_gpu
from collections import defaultdict
from parsing_utils import parse_grammar, parse_graph, products_set
from copy import deepcopy
import time

silent = False
on_gpu = True


def log(message):
    if not silent:
        print(message)


def update_matrix(matrices, head, body, matrices_i=None):
    if on_gpu:
        return update_matrix_gpu(matrices, head, body, matrices_i=matrices_i)
    else:
        return update_matrix_cpu(matrices, head, body, matrices_i=matrices_i)


def main(grammar_file, graph_file, type='bool', pack_both_axis=False):
    t_start = t_parse_start = time.time()
    grammar, inverse_grammar = parse_grammar(grammar_file)
    graph, graph_size = parse_graph(graph_file)

    t_parse_end = t_bool_adj_start = time.time()
    matrices = get_boolean_adjacency_matrices(grammar, inverse_grammar, graph, graph_size)
    matrices_packed_y = deepcopy(matrices) if pack_both_axis else None
    if type != 'bool':
        matrices_to_type(matrices, type)
        if matrices_packed_y is not None:
            matrices_to_type(matrices_packed_y, type)
    if on_gpu:
        matrices = matrices_to_gpu(matrices)

    t_bool_adj_end = t_solution_start = time.time()
    remove_terminals(grammar, inverse_grammar)
    iterate_on_grammar(grammar, inverse_grammar, matrices, matrices_packed_y=matrices_packed_y)
    if on_gpu:
        matrices = matrices_from_gpu(matrices)
    if type != 'bool':
        matrices = matrices_from_type(matrices, type, graph_size)

    t_solution_end = time.time()
    print(solution_string(matrices))
    t_end = time.time()
    log(f'Parsing files took {t_parse_end - t_parse_start} s')
    log(f'Getting adjacent matrices took {t_bool_adj_end - t_bool_adj_start} s')
    log(f'Solving took {t_solution_end - t_solution_start} s')
    log(f'Total execution time (with print) is {t_end - t_start} s')


def matrices_to_type(matrices, type, axis=-1):
    if type in [np.uint8, 'uint8', 'byte']:
        for key, matrix in matrices.items():
            matrices[key] = np.packbits(matrix, axis=axis)
    elif type in [np.uint32, 'uint32', 'int']:
        for key, matrix in matrices.items():
            matrix = np.pad(matrix, [(0, 0), (0, (32 - matrix.shape[1] % 32) % 32)], 'constant', constant_values=0).astype(np.uint32)
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


def remove_terminals(grammar, inverse_grammar):
    terminals = [body for body in inverse_grammar.keys() if type(body) is str]
    for terminal in terminals:
        heads = inverse_grammar.pop(terminal)
        for head in heads:
            grammar[head].remove(terminal)
    log('Successfully removed terminals from grammar. Amount was {}'.format(len(terminals)))


def get_boolean_adjacency_matrices(grammar, inv_grammar, graph, graph_size):
    size = graph_size
    matrices = {i: np.zeros((size, size), dtype=np.bool) for i in grammar}
    for row, verts in graph.items():
        for col, value in verts.items():
            if value in inv_grammar:
                for nonterminal in inv_grammar[value]:
                    matrices[nonterminal][row, col] = True
    log('Calculated {} adjacency matrices of shape {}'.format(len(matrices), (size, size)))
    return matrices


def iterate_on_grammar(grammar, inverse_grammar, matrices, matrices_packed_y=None):
    inverse_by_nonterm = defaultdict(set)
    for body, heads in inverse_grammar.items():
        assert type(body) is tuple, 'Left terminals in grammar: {}'.format(body)
        for head in heads:
            if body[0] != head:
                inverse_by_nonterm[body[0]].add((head, body))
            if body[1] != head:
                inverse_by_nonterm[body[1]].add((head, body))

    log('Built inverse_by_nonterm dictionary')

    to_recalculate = set(products_set(grammar))
    while to_recalculate:
        head, body = to_recalculate.pop()
        assert type(body) is tuple, 'Body is either str or tuple, not {}'.format(type(body))
        is_changed = update_matrix(matrices, head, body, matrices_i=matrices_packed_y)
        if not is_changed:
            continue
        for product in inverse_by_nonterm[head]:
            if product != (head, body):
                to_recalculate.add(product)
    log('Finished iterating on grammar. Final matrices are calculated')


def update_matrix_cpu(matrices, head, body, matrices_i=False):
    head_mat = matrices[head]
    body_first_mat, body_second_mat = matrices[body[0]], matrices[body[1]]
    if str(head_mat.dtype) == 'bool':
        new_matrix = head_mat + body_first_mat.dot(body_second_mat)
        matrices[head] = new_matrix
        return np.any(new_matrix != head_mat)
    else:
        raise ValueError('Multiplication of matrices type {} is not supported'.format(head_mat.dtype))


def solution_string(matrices):
    lines = []
    for nonterminal, matrix in matrices.items():
        xs, ys = np.where(matrix)
        # restoring true vertices numbers
        xs += 1
        ys += 1
        pairs = np.vstack((xs, ys)).T
        pairs_vals = ' '.join(map(lambda pair: ' '.join(pair), pairs.astype('str').tolist()))
        lines.append('{} {}'.format(nonterminal, pairs_vals))
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('grammar', type=str, help='File with grammar in CNF')
    parser.add_argument('graph', type=str, help='Path to a directional graph')
    parser.add_argument('-s', '--silent', action='store_true', help='Print logs into console')
    parser.add_argument('-c', '--cpu', action='store_true', help='Run on CPU')
    parser.add_argument('-t', '--type', type=str, default='bool', help='Compress bools to type')
    parser.add_argument('-b', '--pack_both_axis', action='store_true', help='Store matrices packed by both axis')
    args = parser.parse_args()
    silent = args.silent
    on_gpu = not args.cpu

    main(args.grammar, args.graph, type=args.type, pack_both_axis=args.pack_both_axis)
