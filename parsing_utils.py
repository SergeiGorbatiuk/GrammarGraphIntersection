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
    all_verts = set()
    with open(_file, 'rt') as graph:
        lines = graph.readlines()
    for line in lines:
        terms = line.split()
        assert len(terms) == 3
        from_vert, to_vert = int(terms[0]), int(terms[2].rstrip(','))
        result_graph[from_vert][to_vert] = terms[1]

        all_verts.add(from_vert)
        all_verts.add(to_vert)
    # FIXME: another way to fix nodes numbers starting with 1 or more (don't wanna return min)
    minimum = min(all_verts)
    for i in range(minimum):
        result_graph[i] = {}
    return result_graph, minimum + len(all_verts)


def products_set(grammar):
    products = set()
    for head in grammar:
        for body in grammar[head]:
            products.add((head, body))
    return products