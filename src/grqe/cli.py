import argparse
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable, Optional

from grqe.corpus.build_index import build_binary_index, build_unary_index
from grqe.corpus.corpus import Corpus
from grqe.corpus.encode import encode_corpus
from grqe.type_definitions import BinarySignature, UnarySignature
from grqe.util import progress_bar

DEFAULT_BINARY_DISTANCE = 2
DEFAULT_BINARY_FREQUENCY = .05


def warn(msg: str):
    print(msg, file=sys.stderr)


def run_encode(corpus_dir: Path, corpus_name: str, args: argparse.Namespace) -> bool:
    columns = list(args.columns)
    spans = {}
    for span_definition in args.span:
        span, *attributes = span_definition
        spans[span] = set(attributes)

    files = [Path(file) for file in args.files]

    if spans and not files:
        warn('No files were provided but spans are present.')
        warn('Try using  --  to separate input files from spans.')
        return False

    corpus = Corpus(corpus_name, corpus_dir)
    encode_corpus(
        corpus.base,
        columns,
        spans,
        files
    )
    return True


@dataclass
class IndexesToBuild:
    configuration_is_invalid: bool = False
    unary: set[UnarySignature] = field(default_factory=set)
    binary: set[BinarySignature] = field(default_factory=set)
    spans: dict[str, set[UnarySignature]] = field(default_factory=dict)
    min_frequency: Optional[int] = None

    def __len__(self):
        return len(self.unary) + len(self.binary) + sum(len(values) for values in self.spans.values())

    def _report_config_error(self, msg: str):
        warn(msg)
        self.configuration_is_invalid = True

    def _collect_unary_indexes(self, args: argparse.Namespace, corpus: Corpus):
        unary_indexes = set(args.unary)
        for unsupported_feature in sorted(unary_indexes - corpus.features()):
            self._report_config_error(
                f'Cannot build unary index "{unsupported_feature}": {unsupported_feature} is not a known feature.'
            )

        self.unary.update(map(UnarySignature, unary_indexes))
        if args.all_unary:
            self.unary.update(map(UnarySignature, corpus.features()))

    def _collect_binary_indexes(self, args: argparse.Namespace, corpus: Corpus):
        for (feature1, distance, feature2) in args.binary:
            if not distance.isnumeric():
                self._report_config_error(
                    f'Cannot build binary index "{feature1} {distance} {feature2}": '
                    f'{distance} is not a valid integer.'
                )
                continue

            for feature in (feature1, feature2):
                if feature not in corpus.features():
                    self._report_config_error(
                        f'Cannot build binary index "{feature1} {distance} {feature2}": '
                        f'{feature} is not a known feature.'
                    )
                    continue

            self.binary.add(BinarySignature(feature1, int(distance), feature2))

        if args.all_binary is not None:
            if len(args.all_binary) == 0:
                distance = DEFAULT_BINARY_DISTANCE
                features = []

            else:
                distance, *features = args.all_binary

                if not distance.isnumeric():
                    self._report_config_error(f'Cannot build binary indexes: {distance} is not a valid integer.')
                    return
                distance = max(0, int(distance))

            if len(features) == 0:
                features = corpus.features()
            else:
                features = list(features)

            distances = range(distance + 1)
            for feature1, distance, feature2 in product(features, distances, features):
                self.binary.add(BinarySignature(feature1, distance, feature2))

    def _collect_span_indexes(self, args: argparse.Namespace, corpus: Corpus):
        for span, *features in args.span:
            if span not in corpus.spans():
                self._report_config_error(
                    f'Cannot build span indexes for {span}: {span} is not known.'
                )
                continue

            defined_span_features = corpus.spans()[span].keys()
            span_features = set(features)
            for unsupported_feature in sorted(span_features - defined_span_features):
                self._report_config_error(
                    f'Cannot build span index for {span}: {unsupported_feature} is not a known feature.'
                )

            self.spans[span] = set(map(UnarySignature, span_features))

        if args.all_span:
            for span, annotations in corpus.spans().items():
                self.spans[span] = set(map(UnarySignature, annotations.keys()))

    def _check_implied_indexes(self, args: argparse.Namespace, corpus: Corpus):
        unary_features = {signature.feature for signature in self.unary}
        unary_features |= {signature.feature for signature in corpus.unary_indexes().keys()}
        binary_features = {feature for signature in self.binary for feature in (signature.feature1, signature.feature2)}

        missing_features = binary_features - unary_features
        if missing_features and not args.add_implied:
            self._report_config_error(
                'The following unary indexes are required but not requested: ' +
                ' '.join(sorted(missing_features))
            )
        else:
            self.unary.update(map(UnarySignature, missing_features))

    def _collect_min_frequency(self, args: argparse.Namespace, corpus: Corpus):
        if args.frequency is not None:
            if args.frequency.is_integer():
                self.min_frequency = args.frequency
            elif 0 < args.frequency < 1:
                self.min_frequency = args.frequency * len(corpus)
            else:
                self._report_config_error(
                    f'Invalid minimum frequency. {args.frequency} must be a decimal between 0 and 1, or an integer.'
                )

    @staticmethod
    def from_args(corpus: Corpus, args: argparse.Namespace) -> IndexesToBuild:
        configuration = IndexesToBuild()
        configuration._collect_unary_indexes(args, corpus)
        configuration._collect_binary_indexes(args, corpus)
        configuration._collect_span_indexes(args, corpus)
        configuration._check_implied_indexes(args, corpus)
        configuration._collect_min_frequency(args, corpus)
        return configuration


def run_index(corpus_dir: Path, corpus_name: str, args: argparse.Namespace) -> bool:
    if not corpus_dir.exists() or not corpus_dir.is_dir():
        warn(f'Corpus dir does not exist. Encode corpus first: {corpus_dir}')
        return False

    with Corpus(corpus_name, corpus_dir).lock() as corpus:
        indexes_to_build = IndexesToBuild.from_args(corpus, args)
        if indexes_to_build.configuration_is_invalid:
            return False

        with progress_bar(len(indexes_to_build), 'Building indexes') as pbar:
            for unary_index in indexes_to_build.unary:
                build_unary_index(corpus.tokens()[unary_index.feature])
                pbar()

            for span, span_indexes in indexes_to_build.spans.items():
                for span_index in span_indexes:
                    build_unary_index(corpus.spans()[span][span_index.feature])
                    pbar()

            for binary_index in indexes_to_build.binary:
                build_binary_index(corpus, binary_index, indexes_to_build.min_frequency)
                pbar()
        return True


def main():
    parser = make_parser()
    args = parser.parse_args()

    commands: dict[str, Callable[[Path, str, argparse.Namespace], bool]] = {
        'encode': run_encode,
        'index': run_index,
    }
    corpus_dir = Path(args.corpus_dir)
    corpus_name = args.corpus
    success = commands[args.command](corpus_dir, corpus_name, args)
    sys.exit(0 if success else 1)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='corpus',
        description='Manage corpus and indexes.',
        add_help=True,
    )
    subparsers = parser.add_subparsers(
        dest='command',
        metavar='COMMAND',
        required=True,
        title='commands',
        description='Available commands',
    )
    encoder = subparsers.add_parser(
        'encode',
        help='Encode a corpus.',
        description='Encodes a corpus from a given set of source files.',
    )
    indexer = subparsers.add_parser(
        'index',
        help='Create indexes for a corpus.',
        description='Creates indexes for an encoded corpus. Requires the corpus to be encoded first.',
    )
    for common_argument_parser in (encoder, indexer):
        common_argument_parser.add_argument(
            'corpus',
            type=str,
            help='Name of the corpus',
        )
        common_argument_parser.add_argument(
            '--corpus-dir', '-D',
            metavar='DIR',
            required=True,
            type=str,
            help='Directory for the encoded corpora.',
        )

    # Encoder arguments.
    encoder.add_argument(
        '--columns', '-c',
        metavar='COLUMN',
        action='extend',
        nargs='*',
        default=[],
        type=str,
        help='Column names for annotations extracted from the .VRT file.',
    )
    encoder.add_argument(
        '--span', '-s',
        nargs='*',
        action='append',
        metavar=('SPAN', 'ATTRIBUTES'),
        default=[],
        type=str,
        help='Spans to extract from the .VRT file and their extracted attributes. '
             'Each invocation of this attribute specifies one span name followed by all attributes to extract.'
    )
    encoder.add_argument(
        'files',
        nargs='*',
        default=[],
        type=str,
        help='The corpus files to encode.',
    )

    # Indexer arguments.
    indexer.add_argument(
        '--all-unary', '-U',
        action='store_true',
        help='Build unary indexes for all token features.'
    )
    indexer.add_argument(
        '--unary', '-u',
        type=str,
        nargs='*',
        default=[],
        metavar='ATTRIBUTE',
        action='append',
        help='Build unary indexes for the given token features.',
    )
    indexer.add_argument(
        '--all-span', '-S',
        action='store_true',
        help='Build indexes for all span features.'
    )
    indexer.add_argument(
        '--span', '-s',
        nargs='*',
        action='append',
        metavar=('SPAN', 'FEATURES'),
        default=[],
        type=str,
        help='Build indexes for the given span features.'
    )
    indexer.add_argument(
        '--all-binary', '-B',
        type=str,
        nargs='*',
        metavar=('DISTANCE', 'FEATURES'),
        help='Build binary indexes for all token features up to DISTANCE apart. '
             'Optionally, a set of features to encode up to DISTANCE apart can be given. '
             f'If no distance is specified, {DEFAULT_BINARY_DISTANCE} will be used.'
    )
    indexer.add_argument(
        '--binary', '-b',
        type=str,
        nargs=3,
        default=[],
        action='append',
        metavar=('ATTRIBUTE1', 'DISTANCE', 'ATTRIBUTE2'),
        help='Build binary indexes for the given attribute combinations.'
    )
    indexer.add_argument(
        '--add-implied', '-i',
        action='store_true',
        help='When a binary index is requested, build the underlying unary indexes as well.'
    )
    indexer.add_argument(
        '--frequency', '--min-frequency', '-f',
        type=float,
        nargs='?',
        metavar='FREQUENCY',
        const=DEFAULT_BINARY_FREQUENCY,
        help='Only features that appear more than FREQUENCY times will be encoded. '
             'If FREQUENCY is a number between 0 and 1, it will be interpreted as a percentage of the corpus size. '
             f'If no value is provided, {DEFAULT_BINARY_FREQUENCY} will be used.',
    )

    return parser


if __name__ == '__main__':
    main()
