from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Iterable, Optional, Self

from pyroaring import BitMap

from grqe.corpus.disk import IntArray, IntBytesMap
from grqe.type_definitions import Symbol
from grqe.util import binsearch_range


################################################################################
## Inverted sentence index
## Implemented as a sorted array of symbols (interned strings)
## This is a kind of modified suffix array - a "pruned" SA if you like

class Index:
    path: Path
    smallsets: Optional[IntArray]
    bigsets: Optional[IntBytesMap]

    SMALLSET_FILENAME: ClassVar[str] = 'smallsets'
    BIGSET_FILENAME: ClassVar[str] = 'bigsets'

    @staticmethod
    def init_io[X](action: Callable[[], X]) -> Optional[X]:
        try:
            return action()
        except FileNotFoundError:
            return None

    def __init__(self, path: Path) -> None:
        self.path = path

        self.smallsets = Index.init_io(lambda: IntArray(self.path / Index.SMALLSET_FILENAME))
        self.bigsets = Index.init_io(lambda: IntBytesMap(self.path / Index.BIGSET_FILENAME))
        if self.smallsets is None and self.bigsets is None:
            raise FileNotFoundError(f"Index does not exist: {self.path}")

    def __str__(self) -> str:
        return f'{type(self).__name__}:{self.path}'

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.smallsets is not None:
            self.smallsets.close()
        if self.bigsets is not None:
            self.bigsets.close()

    def do_search(self, ranges: Iterable[tuple[int, int]], offset: int = 0) -> BitMap:
        result = BitMap()
        for start, end in ranges:
            if res := self.lookup_smallset(start, end):
                result |= res

            if res := self.lookup_bigset(start, end):
                result |= res
        if offset:
            result = result.shift(-offset)
        return result

    def get_smallset_searchkey(self) -> Callable[[int], int]:
        raise NotImplementedError("Must be overridden by a subclass")

    def lookup_smallset(self, start_key: int, end_key: int) -> Optional[BitMap]:
        if self.smallsets:
            try:
                search_key = self.get_smallset_searchkey()
                error = (start_key == end_key)
                start, end = binsearch_range(0, len(self.smallsets) - 1, start_key, end_key, search_key, error=error)
                if 0 <= start <= end < len(self.smallsets):
                    data = self.smallsets[start:end + 1]
                    return BitMap(data)
            except (KeyError, IndexError):
                pass
        return None

    def lookup_bigset(self, start_key: int, end_key: int) -> Optional[BitMap]:
        if self.bigsets:
            try:
                if start_key == end_key:
                    bmap = self.bigsets[start_key]
                    return BitMap.deserialize(bmap)
                else:
                    bmaps = self.bigsets.slice(start_key, end_key)
                    return BitMap.union(*(BitMap.deserialize(bm) for bm in bmaps))
            except (KeyError, IndexError):
                pass
        return None


class UnaryIndex(Index):
    feature: IntArray

    def __init__(self, path: Path, feature: IntArray):
        super().__init__(path)
        self.feature = feature

    def search(self, sym: Symbol, offset: int = 0) -> BitMap:
        return self.do_search([(sym, sym)], offset)

    def get_smallset_searchkey(self) -> Callable[[int], int]:
        assert self.smallsets
        features = self.feature
        index = self.smallsets

        def search_key(k: int) -> int:
            return features[index[k]]

        return search_key


class BinaryIndex(Index):
    feature1: IntArray
    distance: int
    feature2: IntArray

    bitshift: int

    def __init__(self, path: Path, feature1: IntArray, distance: int, feature2: IntArray):
        super().__init__(path)
        self.feature1, self.distance, self.feature2 = feature1, distance, feature2
        self.bitshift = self.feature2.itemsize * 8

    def search(self, sym1: Symbol, sym2: Symbol, offset: int = 0) -> BitMap:
        key = (sym1 << self.bitshift) + sym2
        return self.do_search([(key, key)], offset)

    def get_smallset_searchkey(self) -> Callable[[int], int]:
        assert self.smallsets

        features1, features2 = self.feature1, self.feature2
        index = self.smallsets

        def search_key(k: int) -> int:
            key1 = features1[index[k]]
            key2 = features2[index[k] + self.distance]
            return (key1 << self.bitshift) + key2

        return search_key
