import sys
from pathlib import Path
from typing import Iterable

from grqe.corpus.corpus import AnnotationsDir, CorpusDir
from grqe.corpus.parser import CorpusHandler, VrtParser


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
    print('Preparing files...', file=sys.stderr)
    parser = VrtParser(columns, spans, sources)

    print('Collecting symbols...', file=sys.stderr)
    collector = AnnotationCollector(columns, spans)
    token_count = parser.process(CorpusHandler(collector.on_token, collector.on_span))

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

    print('Encoding data...', file=sys.stderr)
    encoder = AnnotationEncoder(corpus, columns, spans)
    parser.process(CorpusHandler(encoder.on_token, encoder.on_span))


def span_is_tiling(ranges: list[tuple[int, int]], token_count: int) -> bool:
    if ranges[0][0] != 0 or ranges[-1][1] != token_count:
        return False

    for i in range(len(ranges) - 1):
        (_, lend), (rbegin, _) = ranges[i], ranges[i + 1]
        if lend != rbegin:
            return False
    return True
