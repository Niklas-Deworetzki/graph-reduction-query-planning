import argparse
import sys
from pathlib import Path

from grqe.corpus.corpus import CorpusDir
from grqe.corpus.encode import encode_corpus


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='encode',
        description='Encode a corpus from .VRT files.',
        add_help=True,
    )
    parser.add_argument(
        'files',
        nargs='*',
        default=[],
        type=str,
        help='The .VRT files to encode.',
    )
    parser.add_argument(
        '--corpus',
        required=True,
        type=str,
        help='Directory for the encoded corpus.',
    )
    parser.add_argument(
        '--columns',
        nargs='*',
        metavar='COLUMN',
        default=[],
        type=str,
        help='The column names extracted from the .VRT file.',
    )
    parser.add_argument(
        '--span',
        nargs='*',
        action='append',
        metavar=('SPAN', 'ATTRIBUTES'),
        default=[],
        type=str,
        help='The spans extracted from the .VRT file. '
             'ATTRIBUTES can be repeated arbitrarily many times, specifying the extracted attributes.'
    )
    parser.add_argument(
        '--index',
        nargs='*',
        action='append',
        metavar='INDEX',
        default=[],
        type=str,
        help='Shape of the index to build. Either COLUMN or COLUM DISTANCE COLUMN.'
    )

    parser.add_argument(
        '--unary-indexes',
        action='store_true',
        dest='unary_indexes',
        help='Build unary indexes for all features.'
    )
    parser.add_argument(
        '--no-unary-indexes',
        action='store_false',
        dest='unary_indexes',
        help='Do not build unary indexes for all features.'
    )

    parser.add_argument(
        '--binary-indexes',
        type=int,
        metavar='DISTANCE',
        help='Build binary indexes for all features up to DISTANCE apart.'
    )

    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    columns = list(args.columns)
    spans = {}
    for span_definition in args.span:
        span, *attributes = span_definition
        spans[span] = set(attributes)

    files = [Path(file) for file in args.files]
    corpus = CorpusDir(Path(args.corpus))

    if spans and not files:
        print('No files were provided but spans are present.')
        print('Try using  --  to separate input files from spans.')
        sys.exit(1)

    encode_corpus(
        corpus,
        columns,
        spans,
        files,
    )


if __name__ == '__main__':
    main()
