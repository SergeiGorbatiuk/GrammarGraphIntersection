import argparse
from collections import defaultdict
import numpy as np
from parsing_utils import *


def main(grammar_file, graph_file):
    grammar, inverse_grammar = parse_grammar(grammar_file)
    graph = parse_graph(graph_file)
    matrs = get_boolean_adjacensy_matrices(grammar, inverse_grammar, graph)
    for k, matr in matrs.items():
        print(f'{k} - {np.sum(matr)}')


def get_boolean_adjacensy_matrices(grammar, inv_grammar, graph):
    sz = len(graph)
    matrices = {i: np.zeros((sz, sz), dtype=np.uint8) for i in grammar}
    for row, verts in graph.items():
        for col, value in verts.items():
            if value in inv_grammar:
                matrices[inv_grammar[value]][row][col] = 1
    return matrices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('grammar', type=str, help='File with grammar in CNF')
    parser.add_argument('graph', type=str, help='Path to a directional graph')
    args = parser.parse_args()

    main(args.grammar, args.graph)
