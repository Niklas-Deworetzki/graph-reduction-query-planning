import json
from datetime import datetime
from pathlib import Path
from typing import AbstractSet, Any, Callable, Generator

from grqe.corpus import Corpus
from grqe.cqp import parse
from grqe.evaluation import FullEvaluator
from grqe.profiling import extract_profiling_trace, profile
from grqe.query import Node
from grqe.transformations import optimize, sanitize
from grqe.type_definitions import ResultSet

TRACEFILE_FORMATTING = '%Y-%m-%d %H:%M:%S.jsonl'


def execute(query_str: str, evaluator: FullEvaluator) -> tuple[Node, ResultSet]:
    query = parse(query_str)
    query = optimize(query)
    query = sanitize(query)
    query = optimize(query)

    if 0 in query.possible_widths():
        raise ValueError('Query matches empty result.')

    return query, evaluator.eval_fully(query)


def query_server(
        corpus_dir: Path,
        corpus_name: str,
        traces_dir: Path,
        query_generator: Generator[str],
        result_callback: Callable[[Corpus, str, ResultSet], Any]
):
    traces_dir.mkdir(parents=True, exist_ok=True)
    tracefile_path = traces_dir / datetime.now().strftime(TRACEFILE_FORMATTING)

    with Corpus(corpus_name, corpus_dir).lock() as corpus, tracefile_path.open('w+') as trace_file:
        evaluator = FullEvaluator(corpus)

        for query in query_generator:
            with profile('execution'):
                reduced, result = execute(query, evaluator)

            with profile('results.process'):
                result_callback(corpus, query, result)

            trace = extract_profiling_trace(reduced, query)
            json.dump(trace, trace_file, indent=None)
            print(file=trace_file, flush=True)


def main():
    def ask_for_query():
        while True:
            yield input('Query: ')

    def show_results(corpus: Corpus, query: str, result: ResultSet):
        print(len(result))
        show_n_results(corpus, result, EXAMPLE_RESULTS)

    query_server(
        Path('/home/niklas/encoded/'),
        'wikipedia',
        Path('traces'),
        ask_for_query(),
        show_results
    )


INTERESTING_FEATURES = ['form', 'upos']
CONTEXT_SIZE = 0
EXAMPLE_RESULTS = 0


def show_n_results(corpus: Corpus, results: AbstractSet[tuple[int, int]], limit: int):
    for (b, e), _ in zip(results, range(limit)):
        decoded_features = corpus.decode(range(max(0, b - CONTEXT_SIZE), min(e + CONTEXT_SIZE, len(corpus) - 1)),
                                         INTERESTING_FEATURES)
        tracks = [[f.decode() for f in decoded_features[feature]] for feature in INTERESTING_FEATURES]
        widths = [max(len(v) for v in values_at_position) for values_at_position in zip(*tracks)]

        for track in tracks:
            line = '\t'.join(f'{value: <{width}}' for value, width in zip(track, widths))
            print(line)
        print()


if __name__ == '__main__':
    main()
