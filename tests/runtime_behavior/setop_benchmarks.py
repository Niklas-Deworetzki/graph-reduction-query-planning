import time
from typing import Callable, Iterable

from tqdm import tqdm

from grqe.sets import BucketRangeSet
from random import Random

RANDOM = Random(4547902010)


def create_set(number_buckets: int, elements_per_bucket: int, corpus_size: int) -> BucketRangeSet:
    if elements_per_bucket > corpus_size:
        raise ValueError(f'Cannot sample {elements_per_bucket} positions in corpus of size {corpus_size}')

    positions = list(range(corpus_size))

    buckets = {}
    current_bucket = 1
    while len(buckets) < number_buckets:
        if RANDOM.randint(0, 1):
            RANDOM.shuffle(positions)
            buckets[current_bucket] = positions[0:elements_per_bucket]
            current_bucket += 1
    return BucketRangeSet(buckets)


def random_sets(number_buckets: int, elements_per_bucket: int, corpus_size: int) -> Iterable[BucketRangeSet]:
    while True:
        yield create_set(number_buckets, elements_per_bucket, corpus_size)


def benchmark(op: Callable[[BucketRangeSet], None], max_buckets: int, max_elements: int, iterations: int):
    inputs = [create_set(max_buckets, max_elements, max_elements * 10) for _ in
              tqdm(range(iterations), 'Generating inputs')]

    start_time = time.perf_counter_ns()
    for input_set in inputs: op(input_set)
    end_time = time.perf_counter_ns()

    elapsed = end_time - start_time
    print(f'[{op.__name__}] {(elapsed // (iterations * 1000)) / 1000} ms/it')


def iterate_set(s: BucketRangeSet):
    # naive implementation:     1125.935 ms/it
    # heapq implementation:      458.553 ms/it
    sum(a + b for a, b in s)


benchmark(iterate_set, 100, 10000, 50)
