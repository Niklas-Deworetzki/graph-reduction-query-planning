import json
import os
import re
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Iterable, Optional, override

from grqe.corpus.disk import IntArray, RangeArray, SparseRangeArray, SymbolCollection, TilingRangeArray
from grqe.corpus.index import BinaryIndex, UnaryIndex
from grqe.types import BinarySignature, Symbol, UnarySignature

CORPUS_DIR = Path(os.path.abspath(os.getcwd()))


class Corpus:
    name: str
    base: CorpusDir

    def __init__(self, name: str, corpora_dir: Path = None):
        self.name = name

        root = (corpora_dir if corpora_dir is not None else CORPUS_DIR) / name
        self.base = CorpusDir(self, root)

    def tokens(self) -> dict[str, AnnotationsDir]:
        return self.base.tokens.annotations()

    def spans(self) -> dict[str, dict[str, AnnotationsDir]]:
        return {
            span: dir.annotations()
            for span, dir in self.base.spans.spans().items()
        }

    def unary_indexes(self) -> dict[UnarySignature, UnaryIndex]:
        return self.base.indexes.unary_indexes

    def unary_index(self, feature: str) -> Optional[UnaryIndex]:
        signature = feature
        return self.unary_indexes().get(signature)

    def binary_indexes(self) -> dict[BinarySignature, BinaryIndex]:
        return self.base.indexes.binary_indexes

    def binary_index(self, feature1: str, distance: int, feature2: str) -> Optional[BinaryIndex]:
        signature = (feature1, distance, feature2)
        return self.binary_indexes().get(signature)

    def __len__(self):
        ...


class DirNode(ABC):
    """Abstract base class for entities backed by the file system."""

    path: Path

    def __init__(self, path: Path):
        """Creates the directory with the given name."""
        self.path = path
        self.ensure()

    def ensure(self):
        self.path.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.close()

    @abstractmethod
    def acquire(self):
        ...

    @abstractmethod
    def close(self):
        ...

    def load_metadata(self, path: Path = None) -> dict:
        """Attempts to load 'metadata.json' in this directory, if present."""
        try:
            with open((path or self.path) / 'metadata.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError, json.JSONDecodeError:
            return {}

    def save_metadata(self, **kwargs):
        """Saves the given parameters to 'metadata.json' in this directory."""
        with open(self.path / 'metadata.json', 'w') as f:
            json.dump(kwargs, f)


class CorpusDir(DirNode):
    """Directory structure for an entire corpus, holding tokens, spans and indexes."""
    tokens: TokensDir
    spans: SpansDir
    indexes: IndexDir

    def __init__(self, corpus: Corpus, base_dir: Path):
        super().__init__(base_dir)
        self.tokens = TokensDir(self.path / 'tokens')
        self.spans = SpansDir(self.path / 'spans')
        self.indexes = IndexDir(corpus, self.path / 'indexes')

    @override
    def acquire(self):
        self.tokens.acquire()
        self.spans.acquire()
        self.indexes.acquire()

    @override
    def close(self):
        self.tokens.close()
        self.spans.close()
        self.indexes.close()


class AnnotationsDir(DirNode):
    """Directory structure for one track of annotations. Contains the encoded SymbolCollection
    and the symbols at each position.

    Both are available after calling load."""
    symbols_path: Path
    values_path: Path

    symbols: SymbolCollection
    values: IntArray

    count: int

    def __init__(self, path: Path, count: int):
        super().__init__(path)
        self.symbols_path = self.path / 'symbols'
        self.values_path = self.path / 'values'

        self.count = count
        self.symbols = None
        self.values = None

    @override
    def acquire(self):
        """Load the symbols and values for this annotation."""
        self.symbols = SymbolCollection(self.symbols_path)
        self.values = IntArray(self.values_path)

    @override
    def close(self):
        """Release the symbols and values for this annotation."""
        self.symbols.close()
        self.values.close()
        self.symbols = None
        self.values = None

    def to_symbol(self, value: bytes) -> Symbol:
        return self.symbols.to_symbol(value)

    def write_symbols(self, values: set[bytes]):
        SymbolCollection.build(self.symbols_path, values)

    def prepare_write_values(self):
        self.symbols = SymbolCollection(self.symbols_path)
        self.values = IntArray.create(
            self.count, self.values_path, max_value=len(self.symbols)
        )


class TokensDir(DirNode):
    """Manages all token-level annotations."""
    count: int

    managed_annotations: dict[str, AnnotationsDir]

    def __init__(self, path: Path):
        super().__init__(path)
        self.count = self.load_metadata().get('count', 0)

        self.managed_annotations = {
            p.name: AnnotationsDir(p, self.count)
            for p in self.path.iterdir()
            if p.is_dir()
        }

    def persist_token_count(self, count: int):
        self.count = count
        self.save_metadata(count=count)

    @override
    def acquire(self):
        for annotation in self.managed_annotations.values(): annotation.acquire()

    @override
    def close(self):
        for annotation in self.managed_annotations.values(): annotation.close()

    def annotations(self) -> dict[str, AnnotationsDir]:
        return self.managed_annotations

    def annotation(self, name: str) -> AnnotationsDir:
        if name not in self.managed_annotations:
            result = AnnotationsDir(self.path / name, self.count)
            self.managed_annotations[name] = result
            return result
        return self.managed_annotations[name]


class SpanType(StrEnum):
    TILING = 'tiling'
    SPARSE = 'sparse'


class SpanDir(DirNode):
    """One encoded span. Has metadata attached and annotations as children."""
    ranges_path: Path

    count: int
    ranges: RangeArray

    managed_annotations: dict[str, AnnotationsDir]

    RANGES_CONSTRUCTORS: ClassVar[dict[SpanType, type[RangeArray]]] = {
        SpanType.TILING: TilingRangeArray,
        SpanType.SPARSE: SparseRangeArray,
    }

    def __init__(self, path: Path):
        super().__init__(path)
        self.ranges_path = self.path / 'ranges'

        self.count = self.load_metadata().get('count')
        self.ranges = None

        self.managed_annotations = {
            p.name: AnnotationsDir(p, self.count)
            for p in self.path.iterdir()
            if p.is_dir()
        }

    @override
    def acquire(self):
        for annotation in self.managed_annotations.values(): annotation.acquire()

        span_type = self.load_metadata()['type']
        self.ranges = self.RANGES_CONSTRUCTORS[SpanType(span_type)](self.ranges_path)

    @override
    def close(self):
        for annotation in self.managed_annotations.values(): annotation.close()

        self.ranges.close()
        self.ranges = None

    def persist_ranges(self, ranges: Iterable[tuple[int, int]], is_tiling: bool):
        if is_tiling:
            span_type = SpanType.TILING
        else:
            span_type = SpanType.SPARSE

        self.count = self.RANGES_CONSTRUCTORS[span_type].build(self.ranges_path, ranges)
        self.save_metadata(count=self.count, type=span_type.value)

    def annotations(self) -> dict[str, AnnotationsDir]:
        return self.managed_annotations

    def annotation(self, name: str) -> AnnotationsDir:
        if name not in self.managed_annotations:
            result = AnnotationsDir(self.path / name, self.count)
            self.managed_annotations[name] = result
            return result
        return self.managed_annotations[name]


class SpansDir(DirNode):
    """Manages all encoded spans."""

    managed_spans: dict[str, SpanDir]

    def __init__(self, path):
        super().__init__(path)
        self.managed_spans = {
            p.name: SpanDir(p)
            for p in self.path.iterdir()
            if p.is_dir()
        }

    @override
    def acquire(self):
        for span in self.managed_spans.values(): span.acquire()

    @override
    def close(self):
        for span in self.managed_spans.values(): span.close()

    def spans(self) -> dict[str, SpanDir]:
        return self.managed_spans

    def span(self, name: str) -> SpanDir:
        if name not in self.managed_spans:
            result = SpanDir(self.path / name)
            self.managed_spans[name] = result
            return result
        return self.managed_spans[name]


class IndexDir(DirNode):
    BINARY_PATTERN = re.compile(r'(\w+)@(\d+)@(\w+)')

    corpus: Corpus
    unary_indexes: dict[UnarySignature, UnaryIndex]
    binary_indexes: dict[BinarySignature, BinaryIndex]

    def __init__(self, corpus: Corpus, path: Path):
        super().__init__(path)
        self.corpus = corpus

        for p in self.path.iterdir():
            if not p.is_dir():
                continue
            elif match := IndexDir.BINARY_PATTERN.fullmatch(p.name):
                feature1, distance_str, feature2 = match.groups()
                signature = (feature1, int(distance_str), feature2)
                self.binary_indexes[signature] = BinaryIndex(self.corpus, p, signature)
            else:
                signature = p.name
                self.unary_indexes[signature] = UnaryIndex(self.corpus, p, signature)

    def filename(self, signature: UnarySignature | BinarySignature) -> str:
        match signature:
            case UnarySignature():
                return str(signature)
            case BinarySignature():
                return signature[0] + '@' + str(signature[1]) + '@' + signature[2]
        raise ValueError(f'Unsupported index signature: {signature}')

    @override
    def acquire(self):
        pass

    @override
    def close(self):
        for index in self.unary_indexes.values(): index.close()
        for index in self.binary_indexes.values(): index.close()

    def unary(self, feature: str) -> UnaryIndex:
        signature = feature
        if signature not in self.unary_indexes:
            path = self.path / feature
            path.mkdir(parents=True, exist_ok=True)
            self.unary_indexes[signature] = UnaryIndex(self.corpus, path, signature)
        return self.unary_indexes[signature]

    def binary(self, feature1: str, distance: int, feature2: str) -> BinaryIndex:
        signature = (feature1, distance, feature2)
        if signature not in self.unary_indexes:
            path = self.path / f'{feature1}@{distance}@{feature2}'
            path.mkdir(parents=True, exist_ok=True)
            self.binary_indexes[signature] = BinaryIndex(self.corpus, path, signature)
        return self.binary_indexes[signature]
