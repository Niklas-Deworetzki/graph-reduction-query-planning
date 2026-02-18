from collections import defaultdict
from pathlib import Path
from typing import Generator, Iterator

from pyroaring import BitMap

from grqe.corpus.disk import IntArray, IntBytesMap
from grqe.corpus.corpus import Corpus, IndexDir
from grqe.corpus.index import Index
from grqe.type_definitions import BinarySignature, IndexSignature, UnarySignature


def collect_for_unary(corpus: Corpus, signature: UnarySignature) -> Generator[tuple[int, int]]:
    yield from enumerate(corpus.tokens()[signature.feature].values)


def collect_for_binary(corpus: Corpus, signature: BinarySignature,
                       min_frequency: int = None) -> Generator[tuple[int, int]]:
    features1 = corpus.tokens()[signature.feature1].values
    features2 = corpus.tokens()[signature.feature2].values

    size = len(corpus) - signature.distance
    bitshift = features2.itemsize * 8
    if min_frequency is None:
        for pos in range(size):
            val1 = features1[pos]
            val2 = features2[pos + signature.distance]
            yield pos, (val1 << bitshift) + val2

    else:
        unary1 = corpus.base.indexes.unary(signature.feature1)
        unary2 = corpus.base.indexes.unary(signature.feature2)

        for pos in range(size):
            val1 = features1[pos]
            val2 = features2[pos + signature.distance]
            if len(unary1.search(val1)) >= min_frequency and len(unary2.search(val2)) >= min_frequency:
                yield pos, (val1 << bitshift) + val2

BITMAP_NATIVE_INTEGER_SIZE = BitMap().to_array().itemsize

def build_index_via_bitmaps(
        path: Path,
        collected: Iterator[tuple[int, int]],
) -> int:
    bitmaps: dict[int, BitMap] = defaultdict(BitMap)

    size = 0
    for pos, value in collected:
        bitmaps[value].add(pos)
        size += 1

    sorted_values = sorted(bitmaps.keys())
    max_value = sorted_values[-1]

    small_indexsize = 0
    big_keys = []
    big_values = []
    with IntArray.create(size, path / Index.SMALLSET_FILENAME, max_value=size) as small_index:

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

    if small_indexsize == 0:
        IntArray.getpath(small_index.path).unlink()
        IntArray.getconfigpath(small_index.path).unlink()
    else:
        small_index.truncate(small_indexsize)

    if len(big_keys) > 0:
        IntBytesMap.build(path / Index.BIGSET_FILENAME, big_keys, big_values, size=size, max_value=max_value)
    return size


def build_index(corpus: Corpus, signature: IndexSignature, min_frequency: int = None):
    index_dir = corpus.base.indexes.path / IndexDir.filename(signature)
    match signature:
        case UnarySignature():
            collect = collect_for_unary(corpus, signature)
        case BinarySignature():
            collect = collect_for_binary(corpus, signature, min_frequency)
    index_dir.mkdir(parents=True, exist_ok=True)
    build_index_via_bitmaps(index_dir, collect)
