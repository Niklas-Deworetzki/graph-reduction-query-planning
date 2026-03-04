from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, Self

from grqe.corpus.index import IntArray
from grqe.type_definitions import Symbol


def build_frequency_file(file: Path, values: IntArray):
    occurrences = compute_occurrences(values)

    highest_key = max(occurrences.keys())
    highest_occurrence = max(occurrences.values())

    frequencies = (occurrences[key] for key in range(highest_key + 1))
    IntArray.build(file, frequencies, size=highest_key + 1, max_value=highest_occurrence)


def compute_occurrences(values: IntArray) -> dict[Symbol, int]:
    occurrences = defaultdict(int)
    for value in values:
        occurrences[value] += 1
    return occurrences


class Frequencies:
    path: Path
    frequencies: IntArray

    FREQUENCIES_FILENAME: ClassVar[str] = 'frequencies'

    def __init__(self, path: Path) -> None:
        self.path = path

        try:
            self.frequencies = IntArray(self.path / Frequencies.FREQUENCIES_FILENAME)
        except FileNotFoundError:
            raise FileNotFoundError(f'Frequencies file does not exist: {self.path}')

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.frequencies.close()

    def __getitem__(self, key: int) -> int:
        return self.frequencies[key]

    def for_symbol(self, symbol: Symbol) -> int:
        if 0 <= symbol < len(self.frequencies):
            return self.frequencies[symbol]
        return 0
