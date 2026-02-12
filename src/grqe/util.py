import math
from pathlib import Path
from typing import Callable, Iterable, Literal

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

type ByteOrder = Literal['little', 'big']
type TypeFormat = Literal['B', 'H', 'I', 'Q']
type IntegerSize = Literal[1, 2, 4, 8]

TYPE_CODES: dict[IntegerSize, TypeFormat] = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}


def add_suffix(path: Path, suffix: str) -> Path:
    """Add the suffix to the path, unless it's already there."""
    if path.suffix != suffix:
        path = Path(str(path) + suffix)
    return path


###############################################################################
## N:o bytes needed to store integer values

def get_integer_size(max_value: int) -> IntegerSize:
    """The minimal n:o bytes needed to store values `0...max_value`"""
    itemsize = math.ceil(math.log(max_value + 1, 2) / 8)
    for i in sorted(TYPE_CODES.keys()):
        if itemsize <= i:
            return i
    else:
        raise ValueError(f'Value {max_value} exceeds 64 bit limit.')


def get_typecode(itemsize: IntegerSize) -> TypeFormat:
    """Returns the memoryview typecode for the given bytesize of unsigned integers"""
    return TYPE_CODES[itemsize]


def binsearch_lookup[C: SupportsRichComparison](start: int, end: int, key: C, lookup: Callable[[int], C]) -> bool:
    return -1 != binsearch(start, end, key, lookup, error=False)


def binsearch[C: SupportsRichComparison](start: int, end: int, key: C, lookup: Callable[[int], C],
                                         error: bool = True) -> int:
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
        raise KeyError()
    return -1


def binsearch_first[C: SupportsRichComparison](start: int, end: int, key: C, lookup: Callable[[int], C],
                                               error: bool = True) -> int:
    while start <= end:
        mid = (start + end) // 2
        if lookup(mid) < key:
            start = mid + 1
        else:
            end = mid - 1

    if lookup(start) == key:
        return start

    if error:
        raise KeyError()
    return -1


def binsearch_last[C: SupportsRichComparison](start: int, end: int, key: C, lookup: Callable[[int], C],
                                              error: bool = True) -> int:
    while start <= end:
        mid = (start + end) // 2
        if key < lookup(mid):
            end = mid - 1
        else:
            start = mid + 1

    if lookup(end) == key:
        return end

    if error:
        raise KeyError()
    return end


def binsearch_range[C: SupportsRichComparison](start: int, end: int, start_key: C, end_key: C,
                                               lookup: Callable[[int], C],
                                               error: bool = True) -> tuple[int, int]:
    start = binsearch_first(start, end, start_key, lookup, error)
    end = binsearch_last(start, end, end_key, lookup, error)
    return start, end


try:
    import tqdm


    def progress[T](it: Iterable[T], description: str, nested_level: int = 0) -> Iterable[T]:
        should_remain = nested_level > 0
        return tqdm.tqdm(it, desc=description, position=nested_level, leave=should_remain)

except ImportError:

    def progress[T](it: Iterable[T], description: str, nested_level: int = 0) -> Iterable[T]:
        return it
