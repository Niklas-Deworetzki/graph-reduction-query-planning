import argparse
import sys
from itertools import product
from pathlib import Path
from typing import Callable, Optional

from grqe.corpus.build_index import build_index
from grqe.corpus.corpus import Corpus
from grqe.corpus.encode import encode_corpus
from grqe.type_definitions import BinarySignature, UnarySignature
from grqe.util import progress

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


def run_index(corpus_dir: Path, corpus_name: str, args: argparse.Namespace) -> bool:
    if not corpus_dir.exists() or not corpus_dir.is_dir():
        warn(f'Corpus dir does not exist. Encode corpus first: {corpus_dir}')
        return False

    configuration_is_invalid: bool = False

    def collect_unary_indexes(known_features: set[str]) -> list[UnarySignature]:
        nonlocal configuration_is_invalid
        unary: list[UnarySignature] = []
        for feature in args.unary:
            if feature not in known_features:
                warn(f'Cannot build unary index "{feature}": {feature} is not a known feature.')
                configuration_is_invalid = True
                continue

            unary.append(UnarySignature(feature))

        if args.all_unary:
            unary.extend(map(UnarySignature, args.all_unary))
        return unary

    def collect_binary_indexes(known_features: set[str]) -> list[BinarySignature]:
        nonlocal configuration_is_invalid
        binary: list[BinarySignature] = []
        for (feature1, distance, feature2) in args.binary:
            if not distance.isnumeric():
                warn(f'Cannot build binary index "{feature1} {distance} {feature2}": '
                     f'{distance} is not a valid integer.')
                configuration_is_invalid = True
                continue

            for feature in (feature1, feature2):
                if feature not in known_features:
                    warn(f'Cannot build binary index "{feature1} {distance} {feature2}": '
                         f'{feature} is not a known feature.')
                    configuration_is_invalid = True
                    continue

            binary.append(BinarySignature(feature1, int(distance), feature2))

        if args.all_binary is not None:
            distances = range(args.all_binary + 1)
            for feature1, distance, feature2 in product(features, distances, features):
                binary.append(BinarySignature(feature1, distance, feature2))

        return binary

    def check_implied_indexes(binary: list[BinarySignature], unary: list[UnarySignature]):
        nonlocal configuration_is_invalid
        unary_features = {signature.feature for signature in unary}
        unary_features |= {signature.feature for signature in corpus.unary_indexes().keys()}
        binary_features = {feature for signature in binary for feature in (signature.feature1, signature.feature2)}

        missing_features = binary_features - unary_features
        if missing_features and not args.add_implied:
            warn('The following unary indexes are required but not requested: ' +
                 ' '.join(sorted(missing_features)))
            configuration_is_invalid = True
        else:
            for missing_feature in missing_features:
                unary.append(UnarySignature(missing_feature))

    def get_min_frequency(corpus_size: int) -> Optional[int]:
        nonlocal configuration_is_invalid
        min_frequency = None
        if args.frequency is not None:
            if args.frequency.is_integer():
                min_frequency = args.frequency
            elif 0 < args.frequency < 1:
                min_frequency = args.frequency * corpus_size
            else:
                warn(f'Invalid minimum frequency. {args.frequency} must be a decimal between 0 and 1, or an integer.')
                configuration_is_invalid = True

        return min_frequency

    with Corpus(corpus_name, corpus_dir).lock() as corpus:
        features = corpus.features()

        unary_signatures = collect_unary_indexes(features)
        binary_signatures = collect_binary_indexes(features)
        check_implied_indexes(binary_signatures, unary_signatures)
        min_frequency = get_min_frequency(len(corpus))

        if configuration_is_invalid:
            return False

        all_signatures = unary_signatures + binary_signatures
        for signature in progress(all_signatures, 'Building indexes'):
            build_index(corpus, signature, min_frequency)
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
        help='Build unary indexes for all features.'
    )
    indexer.add_argument(
        '--unary', '-u',
        type=str,
        nargs='*',
        default=[],
        metavar='ATTRIBUTE',
        action='append',
        help='Build unary indexes for the given features.',
    )
    indexer.add_argument(
        '--all-binary', '-B',
        type=int,
        nargs='?',
        metavar='DISTANCE',
        const=DEFAULT_BINARY_DISTANCE,
        help='Build binary indexes for all features up to DISTANCE apart. '
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
