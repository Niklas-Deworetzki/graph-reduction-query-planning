import time
from typing import Callable

from tqdm import tqdm

from arbitrary_set import random_set
from grqe.sets import BucketRangeSet


def benchmark(op: Callable[[BucketRangeSet], None], max_buckets: int, max_elements: int, iterations: int):
    inputs = [random_set(max_buckets, max_elements, max_elements * 10) for _ in
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
