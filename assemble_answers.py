import os
import argparse
from collections import defaultdict


def main(results_dir, output_path):
    assert os.path.exists(results_dir), 'Provided path does not exist'
    result_files = os.listdir(results_dir)
    assert len(result_files) > 0, 'Directory is empty'

    testcases = []

    final_results = defaultdict(dict)

    with open(os.path.join(results_dir, result_files[0]), 'rt') as f:
        header = f.readline()
        graph_name_header, gram_name_header, time_header = header.split(',')[:3]
        for line in f.readlines():
            graph, grammar = line.strip().split(',')[:2]
            testcases.append(graph + '_-_' + grammar)

    for file in result_files:
        person = file.split('.')[0]
        with open(os.path.join(results_dir, file), 'rt') as f:
            f.readline()  # omitting header
            for line in f.readlines():
                graph, grammar, time, correct = line.strip().split(',')
                tag = graph + '_-_' + grammar
                result = time if correct else -1
                final_results[person][tag] = result

    with open(output_path, 'wt') as f:
        f.write('Name' + ','+','.join(tag)+'\n')  # header
        for person, person_res in final_results.items():
            res_string = person + ',' + ','.join([final_results[person][tag] for tag in testcases])
            f.write(res_string + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str, help='Directory containing results in .csv format')
    parser.add_argument('output_path', type=str, help='Path to output file')
    args = parser.parse_args()

    main(args.results_dir, args.output_path)
