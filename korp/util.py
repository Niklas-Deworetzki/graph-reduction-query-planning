import math
import re
from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Literal, NewType, Protocol, TypeVar

###############################################################################
## Project-specific constants and functions

Feature = NewType('Feature', bytes)
FValue = NewType('FValue', bytes)

WORD = Feature(b'word')
SENTENCE = Feature(b's')

EMPTY = FValue(b'')
START = FValue(b's')


def check_feature(feature: Feature) -> None:
    assert isinstance(feature, bytes), f"Feature must be a bytestring: {feature!r}"
    assert re.match(br'^[a-z_][a-z_0-9]*$', feature), f"Ill-formed feature: {feature.decode()}"


###############################################################################
## Type definitions

ByteOrder = Literal['little', 'big']

T = TypeVar('T')


class ComparableProtocol(Protocol):
    """Protocol for annotating comparable types."""

    @abstractmethod
    def __lt__(self, other: 'CT', /) -> bool: ...

    @abstractmethod
    def __eq__(self, other: 'CT', /) -> bool: ...


CT = TypeVar('CT', bound=ComparableProtocol)


###############################################################################
## File/path utilities

def add_suffix(path: Path, suffix: str) -> Path:
    """Add the suffix to the path, unless it's already there."""
    path = Path(path)
    if path.suffix != suffix:
        path = Path(str(path) + suffix)
    return path


###############################################################################
## N:o bytes needed to store integer values

def get_integer_size(max_value: int) -> int:
    """The minimal n:o bytes needed to store values `0...max_value`"""
    itemsize = math.ceil(math.log(max_value + 1, 2) / 8)
    assert 1 <= itemsize <= 8
    if itemsize == 3:
        return 4
    elif itemsize > 4:
        return 8
    else:
        return itemsize


TypeFormat = Literal['B', 'H', 'I', 'Q']


def get_typecode(itemsize: int) -> TypeFormat:
    """Returns the memoryview typecode for the given bytesize of unsigned integers"""
    typecodes: dict[int, TypeFormat] = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}
    return typecodes[itemsize]


###############################################################################
## Reading (compressed and uncompressed) files

def binsearch_lookup(start: int, end: int, key: CT, lookup: Callable[[int], CT]) -> bool:
    try:
        binsearch(start, end, key, lookup)
        return True
    except KeyError:
        return False


def binsearch(start: int, end: int, key: CT, lookup: Callable[[int], CT], error: bool = True) -> int:
    while start <= end:
        mid = (start + end) // 2
        mykey = lookup(mid)
        if mykey == key:
            return mid
        elif mykey < key:
            start = mid + 1
        else:
            end = mid - 1
    if error:
        raise KeyError(f'Key "{key}" not found')
    return -1


def binsearch_first(start: int, end: int, key: CT, lookup: Callable[[int], CT], error: bool = True) -> int:
    while start <= end:
        mid = (start + end) // 2
        if lookup(mid) < key:
            start = mid + 1
        else:
            end = mid - 1
    if error and lookup(start) != key:
        keystr = key.decode(errors='ignore') if isinstance(key, bytes) else str(key)
        raise KeyError(f'Key "{keystr}" not found')
    return start


def binsearch_last(start: int, end: int, key: CT, lookup: Callable[[int], CT], error: bool = True) -> int:
    while start <= end:
        mid = (start + end) // 2
        if key < lookup(mid):
            end = mid - 1
        else:
            start = mid + 1
    if error and lookup(end) != key:
        raise KeyError(f'Key "{key}" not found')
    return end


def binsearch_range(start: int, end: int, start_key: CT, end_key: CT, lookup: Callable[[int], CT],
                    error: bool = True) -> tuple[int, int]:
    start = binsearch_first(start, end, start_key, lookup, error)
    end = binsearch_last(start, end, end_key, lookup, error)
    return start, end
