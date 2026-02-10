import math
from typing import Callable, Literal

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

type ByteOrder = Literal['little', 'big']
type TypeFormat = Literal['B', 'H', 'I', 'Q']

TYPE_CODES: dict[int, TypeFormat] = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}


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


def get_typecode(itemsize: int) -> TypeFormat:
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
