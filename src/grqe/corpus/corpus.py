import json
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager, suppress
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Generator, Iterable, Optional, Self, override

from grqe.corpus import build_binary_index, build_unary_index
from grqe.corpus.disk import IntArray, RangeArray, SparseRangeArray, SymbolCollection, TilingRangeArray
from grqe.corpus.frequencies import Frequencies, build_frequency_file
from grqe.corpus.index import BinaryIndex, Index, UnaryIndex
from grqe.type_definitions import BinarySignature, Feature, IndexSignature, Symbol, UnarySignature
from grqe.util import transaction

CORPUS_DIR = Path(os.path.abspath(os.getcwd()))


class Corpus:
    name: str
    base: CorpusDir

    def __init__(self, name: str, corpora_dir: Path = None):
        self.name = name

        root = (corpora_dir if corpora_dir is not None else CORPUS_DIR) / name
        self.base = CorpusDir(self, root)

    @contextmanager
    def lock(self) -> Generator[Self]:
        # TODO: Actually implement some locking logic
        try:
            self.base.acquire()
            yield self
        finally:
            self.base.release()

    def get_symbol(self, feature: Feature, value: bytes) -> Symbol:
        return self.base.tokens.annotation(feature).to_symbol(value)

    def features(self) -> set[str]:
        return set(self.tokens().keys())

    def feature(self, name: str) -> AnnotationsDir:
        return self.base.tokens.annotation(name)

    def tokens(self) -> dict[str, AnnotationsDir]:
        return self.base.tokens.annotations()

    def spans(self) -> dict[str, SpanDir]:
        return self.base.spans.spans()

    def span(self, name: str) -> SpanDir:
        return self.spans()[name]

    def unary_indexes(self) -> dict[UnarySignature, UnaryIndex]:
        return {
            UnarySignature(feature): annotation.index
            for feature, annotation in self.base.tokens.annotations().items()
            if annotation.index
        }

    def unary_index(self, feature: str) -> Optional[UnaryIndex]:
        signature = UnarySignature(feature)
        return self.unary_indexes().get(signature)

    def binary_indexes(self) -> dict[BinarySignature, BinaryIndex]:
        return self.base.indexes.binary_indexes

    def binary_index(self, feature1: str, distance: int, feature2: str) -> Optional[BinaryIndex]:
        signature = BinarySignature(feature1, distance, feature2)
        return self.binary_indexes().get(signature)

    def __len__(self):
        return self.base.tokens.count


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
        self.release()

    @abstractmethod
    def acquire(self):
        ...

    @abstractmethod
    def release(self):
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
    def release(self):
        self.tokens.release()
        self.spans.release()
        self.indexes.release()


class AnnotationsDir(DirNode):
    """Directory structure for one track of annotations. Contains the encoded SymbolCollection
    and the symbols at each position.

    Both are available after calling load."""
    symbols_path: Path
    values_path: Path
    index_path: Path
    frequencies_path: Path

    symbols: SymbolCollection
    values: IntArray
    index: Optional[UnaryIndex]
    frequencies: Optional[Frequencies]

    count: int

    def __init__(self, path: Path, count: int):
        super().__init__(path)
        self.symbols_path = self.path / 'symbols'
        self.values_path = self.path / 'values'
        self.index_path = self.path / 'index'
        self.frequencies_path = self.path / 'frequencies'

        self.count = count
        self.symbols = None
        self.values = None
        self.index = None
        self.frequencies = None

    @override
    def acquire(self):
        """Load the symbols and values for this annotation."""
        self.symbols = SymbolCollection(self.symbols_path)
        self.values = IntArray(self.values_path)
        with suppress(FileNotFoundError):
            self.index = UnaryIndex(self.index_path, self.values)
        with suppress(FileNotFoundError):
            self.frequencies = Frequencies(self.frequencies_path)

    @override
    def release(self):
        """Release the symbols and values for this annotation."""
        self.symbols.close()
        self.values.close()
        self.symbols = None
        self.values = None

        if self.index:
            self.index.close()
            self.index = None

    def to_symbol(self, value: bytes) -> Symbol:
        return self.symbols.to_symbol(value)

    def write_symbols(self, values: set[bytes]):
        SymbolCollection.build(self.symbols_path, values)

    def prepare_write_values(self):
        self.symbols = SymbolCollection(self.symbols_path)
        self.values = IntArray.create(
            self.count, self.values_path, max_value=len(self.symbols)
        )

    def create_index(self, force: bool = False, include_frequencies: bool = True):
        if self.index_path.exists() and not force:
            return

        if self.path.parent.name != 'tokens':
            description = f'{self.path.parent.name}.{self.path.name}'
        else:
            description = self.path.name

        frequencies_path = self.frequencies_path if include_frequencies else None
        with transaction(self.index_path):
            build_unary_index(description, self.index_path, self.values, frequencies_path)

    def create_frequencies(self, force: bool = False):
        if self.frequencies_path.exists() and not force:
            return

        with transaction(self.frequencies_path):
            build_frequency_file(self.frequencies_path, self.values)


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
    def release(self):
        for annotation in self.managed_annotations.values(): annotation.release()

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
    def release(self):
        for annotation in self.managed_annotations.values(): annotation.release()

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
    def release(self):
        for span in self.managed_spans.values(): span.release()

    def spans(self) -> dict[str, SpanDir]:
        return self.managed_spans

    def span(self, name: str) -> SpanDir:
        if name not in self.managed_spans:
            result = SpanDir(self.path / name)
            self.managed_spans[name] = result
            return result
        return self.managed_spans[name]


class IndexDir(DirNode):
    corpus: Corpus
    binary_indexes: dict[BinarySignature, BinaryIndex]

    def __init__(self, corpus: Corpus, path: Path):
        super().__init__(path)
        self.corpus = corpus
        self.unary_indexes = {}
        self.binary_indexes = {}

    @staticmethod
    def filename(signature: IndexSignature) -> str:
        return str(signature)

    @override
    def acquire(self):
        for p in self.path.iterdir():
            if not p.is_dir():
                continue
            else:
                signature = IndexSignature.parse(p.name)
                if isinstance(signature, BinarySignature):
                    feature1 = self.corpus.tokens()[signature.feature1].values
                    feature2 = self.corpus.tokens()[signature.feature2].values
                    self.binary_indexes[signature] = BinaryIndex(p, feature1, signature.distance, feature2)

    @override
    def release(self):
        for index in self.binary_indexes.values(): index.close()

    def unary(self, feature: str) -> UnaryIndex:
        return self.corpus.base.tokens.annotation(feature).index

    def _lookup[S: IndexSignature, I: Index](self, signature: S, ctor: type[I], cache: dict[S, I]) -> I:
        if signature not in cache:
            path = self.path / self.filename(signature)
            path.mkdir(parents=True, exist_ok=True)
            cache[signature] = ctor(self.corpus, path, signature)
        return cache[signature]

    def binary(self, feature1: str, distance: int, feature2: str) -> BinaryIndex:
        signature = BinarySignature(feature1, distance, feature2)
        return self._lookup(signature, BinaryIndex, self.binary_indexes)

    def create_index(self, signature: IndexSignature, force: bool = False, min_frequency: int = None):
        if isinstance(signature, UnarySignature):
            self.corpus.base.tokens.annotation(signature.feature).create_index(force=force)

        elif isinstance(signature, BinarySignature):
            index_path = self.path / self.filename(signature)

            if index_path.exists() and not force:
                return

            feature1 = self.corpus.tokens()[signature.feature1]
            feature2 = self.corpus.tokens()[signature.feature2]
            with transaction(index_path):
                build_binary_index(
                    str(signature), index_path,
                    feature1.index, feature1.frequencies,
                    signature.distance,
                    feature2.index, feature2.frequencies,
                    min_frequency
                )
