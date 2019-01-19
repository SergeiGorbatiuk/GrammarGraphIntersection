from collections import defaultdict


def parse_grammar(_file):
    grammar, inverse_grammar = defaultdict(set), defaultdict(set)
    with open(_file, 'rt') as gramm:
        lines = gramm.readlines()
    for line in lines:
        terms = line.split()
        if len(terms) == 2:
            grammar[terms[0]].add(terms[1])
            inverse_grammar[terms[1]].add(terms[0])
        elif len(terms) == 3:
            grammar[terms[0]].add((terms[1], terms[2]))
            inverse_grammar[terms[1], terms[2]].add(terms[0])
        else:
            assert False, f'File malformed. Error near {line}, got {terms}'

    return grammar, inverse_grammar


def parse_graph(_file):
    result_graph = defaultdict(dict)
    with open(_file, 'rt') as graph:
        lines = graph.readlines()
    for line in lines:
        terms = line.split()
        assert len(terms) == 3
        result_graph[int(terms[0])][int(terms[2].rstrip(','))] = terms[1]
    return result_graph


def products_set(grammar):
    products = set()
    for head in grammar:
        for body in grammar[head]:
            products.add((head, body))
    return products