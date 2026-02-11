import json
import sys
from abc import ABC, abstractmethod
from mmap import mmap
from typing import Any, ClassVar, Iterator, Self, override

from .util import *

type Symbol = int


def do_mmap(file: Path, itemsize: int) -> memoryview:
    with file.open('r+b') as file:
        try:
            map = mmap(file.fileno(), 0)
        except ValueError:  # When file is empty.
            map = bytearray(0)
    return memoryview(map).cast(get_typecode(itemsize))


class OnDisk(ABC):

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @abstractmethod
    def close(self) -> None:
        ...


class Array[T]:

    def __iter__(self) -> Iterator[T]:
        raise NotImplemented()

    def __getitem__(self, index: int) -> T:
        raise NotImplemented()

    def __len__(self) -> int:
        raise NotImplemented()


class IntArray(OnDisk, Array[int]):
    """Array of integers."""

    ARRAY_SUFFIX: ClassVar[str] = '.ia'
    DEFAULT_ITEMSIZE: ClassVar[int] = 4

    path: Path
    itemsize: int
    _array: memoryview

    def __init__(self, path: Path):
        with open(IntArray.getconfigpath(path)) as configfile:
            config = json.load(configfile)
        assert config['byteorder'] == sys.byteorder, f'Byte order "{config['byteorder']}" is not supported.'

        self.itemsize = config['itemsize'] or IntArray.DEFAULT_ITEMSIZE
        self._array = do_mmap(IntArray.getpath(path), self.itemsize)
        self.path = path

    def __len__(self) -> int:
        return self._array.__len__()

    def __getitem__(self, item):
        return self.__getitem__(item)

    def slice(self, j: int, k: int) -> memoryview:
        return self._array[j:k]

    def __setitem__(self, key, value):
        self._array.__setitem__(key, value)

    def __iter__(self):
        return self._array.__iter__()

    @override
    def close(self) -> None:
        obj = self._array.obj
        self._array.release()
        if isinstance(obj, mmap):
            obj.close()

    @staticmethod
    def _write_config(path: Path, itemsize: int, size: int):
        with open(IntArray.getconfigpath(path), 'w') as configfile:
            full_config = {
                'itemsize': itemsize,
                'byteorder': sys.byteorder,
                'size': size,
            }
            json.dump(full_config, configfile)

    @staticmethod
    def create(size: int, path: Path, max_value: int = None, itemsize: int = None) -> 'IntArray':
        assert not (max_value and itemsize), 'Only one of "max_value" and "itemsize" should be provided.'

        if max_value is not None:
            itemsize = get_integer_size(max_value)
        elif itemsize is None:
            itemsize = IntArray.DEFAULT_ITEMSIZE

        IntArray._write_config(path, itemsize, size)
        with open(IntArray.getpath(path), 'wb') as file:
            file.truncate(size * itemsize)
        return IntArray(path)

    @staticmethod
    def build(path: Path, values: Iterable[int], size: int = None,
              max_value: int = None, itemsize: int = None) -> int:
        if size is None:
            if hasattr(values, '__len__'):
                size = len(values)
            else:
                values = list(values)
                size = len(values)

        array = IntArray.create(size, path, max_value, itemsize)

        realsize = 0
        for val in values:
            array[realsize] = val
            realsize += 1
        array.close()

        if realsize < size:
            array.truncate(realsize)
        return realsize

    def truncate(self, newsize: int) -> None:
        itemsize = self._array.itemsize

        if isinstance(self._array.obj, mmap):
            self.close()
            with open(self.path, 'r+b') as file:
                file.truncate(newsize * itemsize)
            self._array = do_mmap(self.path, itemsize)

        IntArray._write_config(self.path, itemsize, newsize)

    @staticmethod
    def getpath(path: Path) -> Path:
        return add_suffix(path, IntArray.ARRAY_SUFFIX)

    @staticmethod
    def getconfigpath(path: Path) -> Path:
        return add_suffix(IntArray.getpath(path), '.cfg')


class BytesArray(OnDisk, Array[bytes]):
    """Array of bytes. Uses mmap to store all bytes objects and IntArray to store their start points."""

    VALUE_SEPERATOR: ClassVar[bytes] = b'\0'

    _starts: IntArray
    _rawdata: mmap

    def __init__(self, path: Path) -> None:
        startspath, rawpath = self.getpaths(path)
        self._starts = IntArray(startspath)
        with rawpath.open('r+b') as file:
            self._rawdata = mmap(file.fileno(), 0)

    def __len__(self) -> int:
        return len(self._starts) - 1

    def __getitem__(self, i: int) -> bytes:
        arr = self._starts._array
        start, end = arr[i], arr[i + 1]
        return self._rawdata[start:end - 1]

    def slice(self, j: int, k: int) -> list[bytes]:
        return [self[i] for i in range(j, k)]

    def __setitem__(self, i: int, value: bytes) -> None:
        raise TypeError(f'{self.__class__.__name__} does not support item assignment')

    def __iter__(self) -> Iterator[bytes]:
        raw = self._rawdata
        arr = self._starts._array

        for i in range(len(self)):
            start = arr[i]
            end = arr[i + 1]
            yield raw[start:end - 1]

    def close(self) -> None:
        self._rawdata.close()
        self._starts.close()

    @staticmethod
    def build(path: Path, values: Iterable[bytes]) -> int:
        startspath, rawpath = BytesArray.getpaths(path)

        pos = 0
        starts: list[int] = [pos]
        with rawpath.open('wb') as rawfile:
            for val in values:
                pos += rawfile.write(val)
                pos += rawfile.write(BytesArray.VALUE_SEPERATOR)
                starts.append(pos)

        IntArray.build(startspath, starts, len(starts), max_value=starts[-1])
        return len(starts) - 1

    @staticmethod
    def getpaths(path: Path) -> tuple[Path, Path]:
        return add_suffix(path, '.starts'), add_suffix(path, '.rawdata')


class SymbolCollection(OnDisk, Array[bytes]):
    """Sorted BytesArray of unique strings."""

    _bytesarray: BytesArray

    def __init__(self, path: Path):
        self._bytesarray = BytesArray(path)

    def __len__(self):
        return self._bytesarray.__len__()

    def __getitem__(self, item):
        return self._bytesarray.__getitem__(item)

    def __iter__(self):
        return self._bytesarray.__iter__()

    @override
    def close(self) -> None:
        self._bytesarray.close()

    def from_symbol(self, symbol: Symbol) -> bytes:
        return self[symbol]

    def to_symbol(self, name: bytes) -> Symbol:
        try:
            ba = self._bytesarray
            return binsearch(0, len(self) - 1, name, lambda i: ba[i])
        except (KeyError, IndexError, ValueError):
            raise ValueError(f'Symbol does not exist: {name.decode(errors='ignore')}')

    @staticmethod
    def build(path: Path, names: Iterable[bytes]) -> int:
        names = sorted({b''} | set(names))
        assert names[0] == b''
        return BytesArray.build(path, names)


class IntBytesMap(OnDisk):
    """Mapping sorted integers (the symbols) to bytes."""

    _keys: IntArray
    _values: BytesArray

    def __init__(self, path: Path):
        keyspath, valspath = self.getpaths(path)
        self._keys = IntArray(keyspath)
        self._values = BytesArray(valspath)

    def __len__(self):
        return self._keys.__len__()

    def __getitem__(self, key: int) -> bytes:
        pos = binsearch(0, len(self._keys) - 1, key, lambda k: self._keys[k])
        return self._values[pos]

    def slice(self, start_key: int, end_key: int) -> list[bytes]:
        start, end = binsearch_range(0, len(self._keys) - 1, start_key, end_key, lambda k: self._keys[k])
        return self._values.slice(start, end + 1)

    def __setitem__(self, key: int, value: bytes) -> None:
        raise TypeError('IntBytesMap does not support item assignment')

    def close(self) -> None:
        self._keys.close()
        self._values.close()

    @staticmethod
    def build(path: Path, keys: list[int], values: list[bytes],
              size: int = None, max_value: int = None) -> None:
        # Note: the 'keys' must be sorted!
        keyspath, valspath = IntBytesMap.getpaths(path)
        IntArray.build(keyspath, keys, size=size, max_value=max_value)
        BytesArray.build(valspath, values)

    @staticmethod
    def getpaths(path: Path) -> tuple[Path, Path]:
        return add_suffix(path, '.keys'), add_suffix(path, '.values')


class RangeArray(OnDisk, Array[tuple[int, int]], ABC):

    @classmethod
    @abstractmethod
    def build(cls, path: Path, ranges: Iterable[tuple[int, int]]) -> int:
        ...


class SparseRangeArray(RangeArray):
    _starts: IntArray
    _lengths: IntArray

    def __init__(self, path: Path):
        startspath, lengthspath = SparseRangeArray.getpaths(path)
        self._starts = IntArray(startspath)
        self._lengths = IntArray(lengthspath)

    def __len__(self):
        return self._starts.__len__()

    @override
    def __getitem__(self, key: int) -> tuple[int, int]:
        start = self._starts[key]
        length = self._lengths[key]
        return start, start + length

    @override
    def __iter__(self) -> Iterator[tuple[int, int]]:
        for i in range(len(self)):
            yield self[i]

    def close(self) -> None:
        self._starts.close()
        self._lengths.close()

    @classmethod
    @override
    def build(cls, path: Path, ranges: Iterable[tuple[int, int]]) -> int:
        startspath, lengthspath = SparseRangeArray.getpaths(path)

        starts, lengths = [], []
        for start, end in ranges:
            starts.append(start)
            lengths.append(end - start)

        IntArray.build(startspath, starts)
        return IntArray.build(lengthspath, lengths)

    @staticmethod
    def getpaths(path: Path) -> tuple[Path, Path]:
        return add_suffix(path, '.starts'), add_suffix(path, '.lengths')


class TilingRangeArray(RangeArray):
    _starts: IntArray

    def __init__(self, path: Path):
        self._starts = IntArray(path)

    def __len__(self) -> int:
        return len(self._starts) - 1

    def __getitem__(self, index: int) -> tuple[int, int]:
        start = self._starts[index]
        next_start = self._starts[index + 1]
        return start, next_start

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for i in range(len(self)):
            yield self[i]

    def close(self) -> None:
        self._starts.close()

    @classmethod
    @override
    def build(cls, path: Path, ranges: Iterable[tuple[int, int]]) -> int:
        starts = []
        for start, end in ranges:
            starts.append(start)
        starts.append(end if starts else 0)

        return IntArray.build(path, starts) - 1
