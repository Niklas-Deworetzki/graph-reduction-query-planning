from collections import defaultdict
from pathlib import Path
from typing import Callable, Generator, Iterator, Optional, Sized

from pyroaring import BitMap

from grqe.corpus.disk import IntArray, IntBytesMap
from grqe.corpus.frequencies import Frequencies, compute_occurrences
from grqe.corpus.index import Index, UnaryIndex
from grqe.type_definitions import Symbol
from grqe.util import progress


def collect_from_data(features: IntArray, description: str) -> Generator[tuple[int, int]]:
    yield from enumerate(progress(features, description, unit_scale=True))


def collect_for_binary(
        description: str,
        feature1: IntArray, feature1_frequency: Callable[[Symbol], int],
        distance: int,
        feature2: IntArray, feature2_frequency: Callable[[Symbol], int],
        min_frequency: int = None,
):
    size = len(feature1) - distance
    bitshift = feature2.itemsize * 8

    if min_frequency is None:
        for pos in progress(range(size), description, unit_scale=True):
            val1 = feature1[pos]
            val2 = feature2[pos + distance]
            yield pos, (val1 << bitshift) + val2

    else:
        for pos in progress(range(size), description, unit_scale=True):
            val1 = feature1[pos]
            val2 = feature2[pos + distance]
            if feature1_frequency(val1) >= min_frequency and feature2_frequency(val2) >= min_frequency:
                yield pos, (val1 << bitshift) + val2


BITMAP_NATIVE_INTEGER_SIZE = BitMap().to_array().itemsize


def build_index_via_bitmaps(
        path: Path,
        collected: Iterator[tuple[int, int]],
        value_track_size: int,
        write_frequencies_to_file: Path = None,
) -> int:
    bitmaps: dict[int, BitMap] = defaultdict(BitMap)

    size = 0
    for pos, value in collected:
        bitmaps[value].add(pos)
        size += 1

    sorted_values = sorted(bitmaps.keys())
    max_value = sorted_values[-1]
    if write_frequencies_to_file is not None:
        write_frequency_file(write_frequencies_to_file, sorted_values, bitmaps)

    small_indexsize = 0
    big_keys: list[int] = []
    big_values: list[bytes] = []
    with IntArray.create(size, path / Index.SMALLSET_FILENAME, max_value=value_track_size) as small_index:

        for value in sorted_values:
            bitmap = bitmaps[value]
            serialized_bitmap = bitmap.serialize()

            # Is it cheaper to store set as bitmap or just the raw bytes?
            if len(serialized_bitmap) <= (small_index.itemsize * len(bitmap)):
                big_keys.append(value)
                big_values.append(serialized_bitmap)
            else:
                if small_index.itemsize == BITMAP_NATIVE_INTEGER_SIZE:
                    small_index[small_indexsize:small_indexsize + len(bitmap)] = bitmap.to_array()
                    small_indexsize += len(bitmap)
                else:
                    for element in bitmap:
                        small_index[small_indexsize] = element
                        small_indexsize += 1

    del bitmaps
    del sorted_values

    if small_indexsize == 0:
        IntArray.getpath(small_index.path).unlink()
        IntArray.getconfigpath(small_index.path).unlink()
    else:
        small_index.truncate(small_indexsize)

    if len(big_keys) > 0:
        IntBytesMap.build(path / Index.BIGSET_FILENAME, big_keys, big_values, size=size, max_value=max_value)

    del big_keys
    del big_values
    return size


def write_frequency_file(file: Path, sorted_keys: list[int], collected_occurrences: dict[int, Sized]):
    max_key = sorted_keys[-1]
    max_frequency = max(len(occurrences) for occurrences in collected_occurrences.values())

    frequencies = (len(collected_occurrences[key]) if key in collected_occurrences else 0
                   for key in range(max_key + 1))
    IntArray.build(file, frequencies, size=max_key + 1, max_value=max_frequency)


def build_binary_index(
        description: str, index_dir: Path,
        feature1: UnaryIndex, feature1_frequency: Optional[Frequencies],
        distance: int,
        feature2: UnaryIndex, feature2_frequency: Optional[Frequencies],
        min_frequency: int = None,
):
    index_dir.mkdir(parents=True, exist_ok=True)

    if feature1_frequency is not None:
        freq1 = feature1_frequency.for_symbol
    else:
        occ1 = compute_occurrences(feature1.feature)
        freq1 = lambda symbol: occ1[symbol]

    if feature2_frequency is not None:
        freq2 = feature2_frequency.for_symbol
    else:
        occ2 = compute_occurrences(feature2.feature)
        freq2 = lambda symbol: occ2[symbol]

    collect = collect_for_binary(description, feature1.feature, freq1, distance, feature2.feature, freq2, min_frequency)
    build_index_via_bitmaps(index_dir, collect, len(feature1.feature))


def build_unary_index(
        description: str,
        index_dir: Path,
        values: IntArray,
        frequencies_file: Path = None,
):
    index_dir.mkdir(parents=True, exist_ok=True)
    collect = collect_from_data(values, description)
    build_index_via_bitmaps(index_dir, collect, len(values), frequencies_file)
