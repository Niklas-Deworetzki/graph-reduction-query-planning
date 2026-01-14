from typing import Any, ClassVar, Iterator, Optional
from collections.abc import Callable
from pathlib import Path
import json
import logging

from pyroaring import BitMap

from .disk import IntArray, IntBytesMap, SymbolRange, SymbolList
from .corpus import Corpus
from .literals import Template, Instance
from .util import add_suffix, binsearch_range
from ..debug import stopwatch


################################################################################
## Inverted sentence index
## Implemented as a sorted array of symbols (interned strings)
## This is a kind of modified suffix array - a "pruned" SA if you like

class Index:
    dir_suffix: ClassVar[str] = '.indexes'

    corpus: Corpus
    template: Template
    smallsets: Optional[IntArray]
    bigsets: Optional[IntBytesMap]
    path: Path

    config: dict[str, int]

    @staticmethod
    def init_io[X](action: Callable[[], X]) -> Optional[X]:
        try:
            return action()
        except FileNotFoundError:
            return None

    def __init__(self, corpus: Corpus, template: Template) -> None:
        self.corpus = corpus
        self.template = template
        self.path = self.indexpath(corpus, template)

        self.smallsets = Index.init_io(lambda: IntArray(self.path))
        self.bigsets = Index.init_io(lambda: IntBytesMap(self.path))
        if self.smallsets is None and self.bigsets is None:
            raise FileNotFoundError(f"Index does not exist: {self.path}")

        self.config = self._loadconfig()

    def _loadconfig(self) -> dict[str, int]:
        try:
            with open(self.getconfigpath(self.path)) as configfile:
                return json.load(configfile)  # type: ignore
        except FileNotFoundError:
            assert self.smallsets
            return self.smallsets.getconfig()

    def __str__(self) -> str:
        return f'{type(self).__name__}:{self.template}'

    def __len__(self) -> int:
        return self.config['size']

    def __enter__(self) -> 'Index':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.smallsets is not None:
            self.smallsets.close()
        if self.bigsets is not None:
            self.bigsets.close()

    def search(self, instance: Instance, offset: int = 0) -> BitMap:
        # FIXME: This method owns 20% of its own runtime, but does not do a lot.
        try:
            ranges = self.get_instance_range(instance)
        except ValueError:
            return BitMap()
        result = BitMap()
        for start, end in ranges:
            result |= self.lookup_smallset(start, end)
            result |= self.lookup_bigset(start, end)
        if offset:
            result = result.shift(-offset)
        return result

    def get_instance_range(self, instance: Instance) -> list[tuple[int, int]]:
        raise NotImplementedError("Must be overridden by a subclass")

    def get_search_key(self) -> Callable[[int], int]:
        raise NotImplementedError("Must be overridden by a subclass")

    def lookup_smallset(self, start_key: int, end_key: int) -> BitMap:
        if self.smallsets:
            try:
                with stopwatch('smallset_seek'):
                    search_key = self.get_search_key()
                    error = (start_key == end_key)
                    start, end = binsearch_range(0, len(self.smallsets) - 1, start_key, end_key, search_key, error=error)
                if 0 <= start <= end < len(self.smallsets):
                    with stopwatch('smallset_load'):
                        data = self.smallsets.slice(start, end + 1)
                    with stopwatch('smallset_deserialize'):
                        return BitMap(data)
            except (KeyError, IndexError):
                pass
        return BitMap()

    def lookup_bigset(self, start_key: int, end_key: int) -> BitMap:
        if self.bigsets:
            try:
                if start_key == end_key:
                    with stopwatch('bigset_load'):
                        bmap = self.bigsets[start_key]
                    with stopwatch('bigset_deserialize'):
                        return BitMap.deserialize(bmap)
                else:
                    with stopwatch('bigset_load'):
                        bmaps = self.bigsets.slice(start_key, end_key)
                    logging.debug(f"Found {len(bmaps)} bitmaps between {start_key}..{end_key}")
                    with stopwatch('bigset_deserialize'):
                        return BitMap.union(*(BitMap.deserialize(bm) for bm in bmaps))
            except (KeyError, IndexError):
                pass
        return BitMap()

    @staticmethod
    def getconfigpath(path: Path) -> Path:
        return add_suffix(path, '.cfg')

    @staticmethod
    def indexpath(corpus: Corpus, template: Template) -> Path:
        basepath = corpus.path.with_suffix(Index.dir_suffix)
        return basepath / str(template) / str(template)

    @staticmethod
    def get(corpus: Corpus, template: Template) -> 'Index':
        if len(template) == 1:
            return UnaryIndex(corpus, template)
        elif len(template) == 2:
            return BinaryIndex(corpus, template)
        else:
            raise ValueError(f"Cannot handle indexes of length {len(template)}: {template}")

    @staticmethod
    def indexes_for(corpus: Corpus) -> Iterator['Index']:
        basepath = corpus.path.with_suffix(Index.dir_suffix)
        for index_root in basepath.iterdir():
            index_name = index_root.name
            index_name.find('+s')

            try:
                yield Index.get(corpus, Template.parse(corpus, index_root.name))
            except (ValueError, FileNotFoundError):
                pass


class UnaryIndex(Index):
    def __init__(self, corpus: Corpus, template: Template) -> None:
        assert len(template) == 1, f"UnaryIndex templates must have length 1: {template}"
        super().__init__(corpus, template)

    def get_instance_range(self, instance: Instance) -> list[tuple[int, int]]:
        assert len(instance) == 1, f"UnaryIndex instance must have length 1: {instance}"
        (value,) = instance
        return [
            v if isinstance(v, SymbolRange) else (v, v)
            for v in (value.symbols if isinstance(value, SymbolList) else [value])
        ]

    def get_search_key(self) -> Callable[[int], int]:
        assert self.smallsets
        tmpl = self.template.template[0]
        features = self.corpus.tokens[tmpl.feature]
        offset = tmpl.offset
        index = self.smallsets.array

        def search_key(k: int) -> int:
            return features[index[k] + offset]

        return search_key


class BinaryIndex(Index):
    def __init__(self, corpus: Corpus, template: Template) -> None:
        assert len(template) == 2, f"BinaryIndex templates must have length 2: {template}"
        super().__init__(corpus, template)

    def get_instance_range(self, instance: Instance) -> list[tuple[int, int]]:
        assert len(instance) == 2, f"BinaryIndex instance must have length 2: {instance}"
        (left, right) = instance
        if isinstance(left, tuple):
            raise ValueError("BinaryIndex cannot have a range as left value")
        if isinstance(left, list) and isinstance(right, list):
            raise ValueError("BinaryIndex cannot have lists for both left and right values")
        return [
            (leftshifted + rightsym1, leftshifted + rightsym2)
            for leftsym in (left.symbols if isinstance(left, SymbolList) else [left])
            for leftshifted in [leftsym << 32]
            for rightrange in (right.symbols if isinstance(right, SymbolList) else [right])
            for (rightsym1, rightsym2) in [(rightrange, rightrange) if isinstance(rightrange, int) else rightrange]
        ]

    def get_search_key(self) -> Callable[[int], int]:
        assert self.smallsets
        tmpl1, tmpl2 = self.template.template
        offset1, offset2 = tmpl1.offset, tmpl2.offset
        features1 = self.corpus.tokens[tmpl1.feature]
        features2 = self.corpus.tokens[tmpl2.feature]
        index = self.smallsets.array

        def search_key(k: int) -> int:
            key1 = features1[index[k] + offset1]
            key2 = features2[index[k] + offset2]
            return (key1 << 32) + key2

        return search_key
