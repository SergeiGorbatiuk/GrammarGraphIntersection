import argparse
import numpy as np
from gpu_methods import matrices_to_gpu, matrices_from_gpu, update_matrix_gpu, threadsperblock
from collections import defaultdict
from parsing_utils import parse_grammar, parse_graph, products_set
import time
from math_utils import *

silent = False
on_gpu = True
shared_memory = False


def log(message):
    if not silent:
        print(message)


def update_matrix(matrices, head, body, shared_memory=False):
    if on_gpu:
        return update_matrix_gpu(matrices, head, body, shared_memory=shared_memory)
    else:
        return update_matrix_cpu(matrices, head, body)


def main(grammar_file, graph_file, type='bool'):
    t_start = t_parse_start = time.time()
    grammar, inverse_grammar = parse_grammar(grammar_file)
    graph, graph_size = parse_graph(graph_file)

    t_parse_end = t_bool_adj_start = time.time()
    matrices = get_boolean_adjacency_matrices(grammar, inverse_grammar, graph, graph_size)
    remove_terminals(grammar, inverse_grammar)
    if type != 'bool':
        matrices_to_type(matrices, type, threadsperblock, shared_memory=shared_memory)


    ####################################################################
    t_bool_adj_end = t_solution_start = time.time()
    if on_gpu:
        matrices = matrices_to_gpu(matrices)
    solve(grammar, inverse_grammar, matrices)
    if on_gpu:
        matrices = matrices_from_gpu(matrices)
    t_solution_end = time.time()
    ####################################################################

    if type != 'bool':
        matrices = matrices_from_type(matrices, type, graph_size)


    print(solution_string(matrices))
    t_end = time.time()
    log(f'Parsing files took {t_parse_end - t_parse_start} s')
    log(f'Getting adjacent matrices took {t_bool_adj_end - t_bool_adj_start} s')
    log(f'Solving took {t_solution_end - t_solution_start} s')
    log(f'Total execution time (with print) is {t_end - t_start} s')


def remove_terminals(grammar, inverse_grammar):
    terminals = [body for body in inverse_grammar.keys() if type(body) is str]
    for terminal in terminals:
        heads = inverse_grammar.pop(terminal)
        for head in heads:
            grammar[head].remove(terminal)
    log('Successfully removed terminals from grammar. Amount was {}'.format(len(terminals)))


def solve(grammar, inverse_grammar, matrices):
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
        is_changed = update_matrix(matrices, head, body, shared_memory=shared_memory)
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
    parser.add_argument('-m', '--memory_shared', action='store_true', help='Use multiplication with shared memory')
    args = parser.parse_args()
    silent = args.silent
    on_gpu = not args.cpu
    shared_memory = args.memory_shared

    main(args.grammar, args.graph, type=args.type)
