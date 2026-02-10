import argparse
import html
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Iterable

from grqe.index.corpus import AnnotationsDir, CorpusDir

XML_NAME_START_CHAR = r'[_:a-zA-Z\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD]'
XML_NAME_CHAR = fr'(?:{XML_NAME_START_CHAR}|[\-.0-9\u00B7\u0300-\u036F\u203F-\u2040])'
XML_NAME = fr'(?:{XML_NAME_START_CHAR}{XML_NAME_CHAR}*)'

XML_ATTRIBUTE_REGEX = re.compile(fr'({XML_NAME}) *= *"([^"]*)"')


def extract_opening_tag(tag: str) -> tuple[str, dict[str, bytes]]:
    end_of_name = tag.find(' ')
    if end_of_name == -1:
        end_of_tag = tag.rfind('>')
        return tag[1:end_of_tag], {}
    else:
        attributes = {
            match.group(1): match.group(2).encode()
            for match in XML_ATTRIBUTE_REGEX.finditer(tag, end_of_name)
        }
        return tag[1:end_of_name], attributes


@dataclass(frozen=True)
class CorpusHandler:
    on_token: Callable[[int, dict[str, bytes]], None]
    on_span: Callable[[str, int, int, dict[str, bytes]], None]


@dataclass(frozen=True)
class VrtParser:
    NO_VALUE: ClassVar[bytes] = b''

    columns: list[str]
    spans: dict[str, set[str]]

    def parse(
            self,
            callback: CorpusHandler,
            *paths: Path,
            included_filetypes: set[str] = frozenset({'.vrt'})
    ) -> int:
        def collect_corpus_files():
            for path in paths:
                if path.is_file():
                    yield path

                else:
                    for (dirpath, _, filenames) in os.walk(path):
                        for filename in filenames:
                            if any(filename.endswith(suffix) for suffix in included_filetypes):
                                yield Path(os.path.join(dirpath, filename))

        token_count = 0
        for file in collect_corpus_files():
            token_count += self._parse_vrt(file, token_count, callback)
        return token_count

    def _parse_vrt(self, file: Path, token_count: int, callback: CorpusHandler) -> int:
        open_spans: list[str] = []
        open_span_data: list[tuple[int, dict[str, bytes]]] = []

        with file.open('r') as f:
            for lineno, content in enumerate(f, start=1):
                if content.isspace():
                    continue

                if content.startswith('<'):
                    closing_bracket = content.rfind('>')
                    match content[1]:
                        case '!':  # <!-- xml comment -->
                            continue
                        case '?':  # <?xml processing instruction>
                            continue

                        case '/':  # </SPAN>
                            span_name = content[2:closing_bracket].strip()
                            if open_spans[-1] != span_name:
                                continue  # End of a region not indexed.

                            open_spans.pop()
                            start, attributes = open_span_data.pop()

                            extracted_attributes = {
                                key: attributes.get(key, VrtParser.NO_VALUE)
                                for key in self.spans[span_name]
                            }
                            callback.on_span(span_name, start, token_count, extracted_attributes)

                        case _:
                            span_name, attributes = extract_opening_tag(content)

                            if span_name in self.spans:
                                open_spans.append(span_name)
                                open_span_data.append((token_count, attributes))

                else:
                    column_values = html.unescape(content.rstrip('\n\r')).split('\t')
                    attributes = {
                        column: value.encode()
                        for column, value in zip(self.columns, column_values)
                    }
                    callback.on_token(token_count, attributes)
                    token_count += 1
        return token_count


class AnnotationCollector:
    column_data: dict[str, set[bytes]]
    span_ranges: dict[str, list[tuple[int, int]]]
    span_data: dict[str, dict[str, set[bytes]]]

    def __init__(self, columns: list[str], spans: dict[str, set[str]]):
        self.column_data = {column: set() for column in columns}

        self.span_ranges = {}
        self.span_data = {}
        for span, keys in spans.items():
            self.span_ranges[span] = []
            self.span_data[span] = {key: set() for key in keys}

    def on_token(self, position: int, attributes: dict[str, bytes]):
        for key, collector in self.column_data.items():
            collector.add(attributes[key])

    def on_span(self, span: str, begin: int, end: int, attributes: dict[str, bytes]):
        self.span_ranges[span].append((begin, end))
        for key, collector in self.span_data[span].items():
            collector.add(attributes[key])


class AnnotationEncoder:
    column_data: dict[str, AnnotationsDir]
    span_data: dict[str, dict[str, AnnotationsDir]]
    span_count: dict[str, int]

    def __init__(self, corpus: CorpusDir, columns: list[str], spans: dict[str, set[str]]):
        self.column_data = {column: corpus.tokens.annotation(column) for column in columns}

        self.span_data = {}
        self.span_count = {}
        for span, keys in spans.items():
            span_dir = corpus.spans.span(span)
            self.span_data[span] = {key: span_dir.annotation(key) for key in keys}
            self.span_count[span] = 0

        for annotation_dir in self.column_data.values():
            annotation_dir.prepare_write_values()

        for annotations in self.span_data.values():
            for annotation_dir in annotations.values():
                annotation_dir.prepare_write_values()

    def write_value(self, offset: int, dir: AnnotationsDir, value: bytes):
        encoded_value = dir.symbols.to_symbol(value)
        dir.values[offset] = encoded_value

    def on_token(self, position: int, attributes: dict[str, bytes]):
        for key, annotations_dir in self.column_data.items():
            self.write_value(position, annotations_dir, attributes[key])

    def on_span(self, span: str, begin: int, end: int, attributes: dict[str, bytes]):
        count = self.span_count[span]

        for key, annotations_dir in self.span_data[span].items():
            self.write_value(count, annotations_dir, attributes[key])

        self.span_count[span] = count + 1


def encode_corpus(
        corpus: CorpusDir,
        columns: list[str],
        spans: dict[str, set[str]],
        sources: Iterable[Path]
):
    parser = VrtParser(columns, spans)

    collector = AnnotationCollector(columns, spans)
    token_count = parser.parse(CorpusHandler(collector.on_token, collector.on_span), *sources)

    corpus.tokens.persist_token_count(token_count)
    for column in columns:
        corpus.tokens.annotation(column).write_symbols(
            collector.column_data[column]
        )

    for span, span_annotations in spans.items():
        span_dir = corpus.spans.span(span)

        span_ranges = collector.span_ranges[span]
        is_tiling = span_is_tiling(span_ranges, token_count)
        span_dir.persist_ranges(span_ranges, is_tiling)
        for span_annotation in span_annotations:
            span_dir.annotation(span_annotation).write_symbols(
                collector.span_data[span][span_annotation]
            )

    encoder = AnnotationEncoder(corpus, columns, spans)
    parser.parse(CorpusHandler(encoder.on_token, encoder.on_span), *sources)


def span_is_tiling(ranges: list[tuple[int, int]], token_count: int) -> bool:
    if ranges[0][0] != 0 or ranges[-1][1] != token_count:
        return False

    for i in range(len(ranges) - 1):
        (_, lend), (rbegin, _) = ranges[i], ranges[i + 1]
        if lend != rbegin:
            return False
    return True


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
