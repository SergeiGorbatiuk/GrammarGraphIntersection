import argparse
import numpy as np
from collections import defaultdict
from parsing_utils import parse_grammar, parse_graph, products_set
import time

silent = False


def log(message):
    if not silent:
        print(message)


def main(grammar_file, graph_file):
    t_start = t_parse_start = time.time()
    grammar, inverse_grammar = parse_grammar(grammar_file)
    graph, graph_size = parse_graph(graph_file)

    t_parse_end = t_bool_adj_start = time.time()
    matrices = get_boolean_adjacency_matrices(grammar, inverse_grammar, graph, graph_size)

    t_bool_adj_end = t_solution_start = time.time()
    remove_terminals(grammar, inverse_grammar)
    iterate_on_grammar(grammar, inverse_grammar, matrices)

    t_solution_end = time.time()
    print(solution_string(matrices))
    t_end = time.time()
    log(f'Parsing files took {t_parse_end - t_parse_start} ms')
    log(f'Getting adjacent matrices took {t_bool_adj_end - t_bool_adj_start} ms')
    log(f'Solving took {t_solution_end - t_solution_start} ms')
    log(f'Total execution time (with print) is {t_end - t_start} ms')


def remove_terminals(grammar, inverse_grammar):
    terminals = [body for body in inverse_grammar.keys() if type(body) is str]
    for terminal in terminals:
        heads = inverse_grammar.pop(terminal)
        for head in heads:
            grammar[head].remove(terminal)
    log('Successfully removed terminals from grammar. Amount was {}'.format(len(terminals)))


def get_boolean_adjacency_matrices(grammar, inv_grammar, graph, graph_size):
    size = graph_size
    # FIXME: replace with np.uint8
    matrices = {i: np.zeros((size, size), dtype=np.bool) for i in grammar}
    for row, verts in graph.items():
        for col, value in verts.items():
            if value in inv_grammar:
                for nonterminal in inv_grammar[value]:
                    matrices[nonterminal][row, col] = True
    log('Calculated {} adjacency matrices of shape {}'.format(len(matrices), (size, size)))
    return matrices


def iterate_on_grammar(grammar, inverse_grammar, matrices):
    # Needed for smarter iterating
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
        matrix = matrices[head]
        new_matrix = matrix + matrices[body[0]].dot(matrices[body[1]])
        if np.any(new_matrix != matrix):
            to_recalculate |= inverse_by_nonterm[head]
            matrices[head] = new_matrix
    log('Finished iterating on grammar. Final matrices are calculated')


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
    args = parser.parse_args()
    silent = args.silent

    main(args.grammar, args.graph)
