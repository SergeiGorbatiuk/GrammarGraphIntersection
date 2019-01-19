import argparse
import numpy as np
from collections import defaultdict
from parsing_utils import parse_grammar, parse_graph, products_set

silent = False

def log(message):
    if not silent:
        print(message)


def main(grammar_file, graph_file):
    grammar, inverse_grammar = parse_grammar(grammar_file)
    graph = parse_graph(graph_file)
    matrices = get_boolean_adjacency_matrices(grammar, inverse_grammar, graph)
    remove_terminals(grammar, inverse_grammar)
    iterate_on_grammar(grammar, inverse_grammar, matrices)
    for k, matr in matrices.items():
        print(f'{k} - {np.sum(matr)}')


def remove_terminals(grammar, inverse_grammar):
    terminals = [body for body in inverse_grammar.keys() if type(body) is str]
    for terminal in terminals:
        heads = inverse_grammar.pop(terminal)
        for head in heads:
            grammar[head].remove(terminal)
    log('Successfully removed terminals from grammar. Amount was {}'.format(len(terminals)))


def get_boolean_adjacency_matrices(grammar, inv_grammar, graph):
    size = len(graph)
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
            inverse_by_nonterm[body[0]].add((head, body))
            inverse_by_nonterm[body[1]].add((head, body))
            assert head not in body, 'Same non-terminal in head and body'

    log('Built inverse_by_nonterm dictionary')

    to_recalculate = set(products_set(grammar))
    while to_recalculate:
        head, body = to_recalculate.pop()
        assert type(body) is tuple, 'Body is either str or tuple, not {}'.format(type(body))
        matrix = matrices[head]
        new_matrix = matrix + matrices[body[0]].dot(matrices[body[1]])
        if np.any(new_matrix != matrix):
            to_recalculate |= inverse_by_nonterm[head]
    log('Finished iterating on grammar. Final matrices are calculated')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('grammar', type=str, help='File with grammar in CNF')
    parser.add_argument('graph', type=str, help='Path to a directional graph')
    args = parser.parse_args()

    main(args.grammar, args.graph)
