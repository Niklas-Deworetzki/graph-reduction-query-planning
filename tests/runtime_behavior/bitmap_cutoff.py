import random
from statistics import mean
from typing import Iterable

import numpy as np
from pyroaring import BitMap
import matplotlib.pyplot as plt

random.seed(42)


def size_to_store_n_numbers(n: int, corpus_size: int) -> int:
    bitmap = BitMap()
    while len(bitmap) < n:
        bitmap.add(random.randint(0, corpus_size))

    return len(bitmap.serialize())


def sample(
        sizes: Iterable[int],
        corpus_size: int,
        samples_per_size: int
) -> dict[int, list[int]]:
    return {
        size: [size_to_store_n_numbers(size, corpus_size) for _ in range(samples_per_size)]
        for size in sizes
        if corpus_size > size
    }


def draw():
    colors = {
        100_000: 'yellow',
        1_000_000: 'orange',
        10_000_000: 'red',
        100_000_000: 'violet',
        1_000_000_000: 'blue',
    }
    sizes = sorted(x * (10 ** y) for y in range(5) for x in [1, 5])

    overhead = {}

    for corpus_size in colors.keys():
        samples = sample(
            sizes,
            corpus_size,
            10
        )

        x = list(samples.keys())
        y = [mean(s) for s in samples.values()]

        a, b = np.polyfit(x, y, 1)
        overhead[corpus_size] = b


    X = list(overhead.keys())
    Y = list(overhead.values())
    plt.plot(X, Y, color='black')
    plt.xscale('log')

    plt.show()


draw()
