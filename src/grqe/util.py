import math
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Generator, Iterable, Literal

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


@contextmanager
def transaction(*files: Path):
    try:
        yield ()
    except Exception as e:
        for file in files:
            shutil.rmtree(file)
        raise e


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


def tree_files(roots: Iterable[Path], included_file_types: Iterable[str]):
    for path in roots:
        if path.is_file():
            yield path
        else:
            for (dirpath, _, filenames) in os.walk(path):
                for filename in filenames:
                    if any(filename.endswith(suffix) for suffix in included_file_types):
                        yield Path(os.path.join(dirpath, filename))


def get_linecount(file: Path) -> int:
    try:
        with file.open('r') as f:
            lineno = 0
            for _ in f:
                lineno += 1
            return lineno
    except IOError:
        return 0


try:
    import tqdm


    def progress[T](it: Iterable[T], description: str, **kwargs) -> Iterable[T]:
        return tqdm.tqdm(it, desc=description, **kwargs)


    @contextmanager
    def progress_bar(maximum: int, description: str, **kwargs) -> Generator[Callable[[], None]]:
        pbar = tqdm.tqdm(total=maximum, desc=description, **kwargs)
        yield lambda: pbar.update()
        pbar.close()


except ImportError:

    def progress[T](it: Iterable[T], description: str, **kwargs) -> Iterable[T]:
        return it


    @contextmanager
    def progress_bar(maximum: int, description: str, **kwargs) -> Generator[Callable[[], None]]:
        yield lambda: None
