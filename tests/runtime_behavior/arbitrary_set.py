from random import Random
from typing import Iterable

from pyroaring import BitMap

from grqe.sets import BucketRangeSet

RANDOM = Random(4547902010)


def random_set(number_buckets: int, elements_per_bucket: int, corpus_size: int) -> BucketRangeSet:
    if elements_per_bucket > corpus_size:
        raise ValueError(f'Cannot sample {elements_per_bucket} positions in corpus of size {corpus_size}')

    positions = list(range(corpus_size))

    buckets = {}
    current_bucket = 1
    while len(buckets) < number_buckets:
        if RANDOM.randint(0, 1):
            buckets[current_bucket] = BitMap(RANDOM.sample(positions, elements_per_bucket))
            current_bucket += 1
    return BucketRangeSet(buckets)


def random_sets(number_buckets: int, elements_per_bucket: int, corpus_size: int) -> Iterable[BucketRangeSet]:
    while True:
        yield random_set(number_buckets, elements_per_bucket, corpus_size)
