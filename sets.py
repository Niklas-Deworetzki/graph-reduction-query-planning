import collections
from typing import Dict, Iterator, Tuple

from pyroaring import BitMap

type Range = Tuple[int, int]


class BucketRangeSet(collections.abc.Set[Range]):
    buckets: Dict[int, BitMap]

    def __init__(self, buckets: Dict[int, BitMap]):
        self.buckets = buckets

    def copy(self) -> 'BucketRangeSet':
        buckets = {key: BitMap(value) for key, value in self.buckets}
        return BucketRangeSet(buckets)

    def join(self, other: 'BucketRangeSet', distance: int = 0) -> 'BucketRangeSet':
        buckets: Dict[int, BitMap] = {}
        for self_length, self_value in self.buckets.items():
            offset = self_length + distance
            expected_endpoints = self_value.shift(offset)

            for other_length, other_value in other.buckets.items():
                joined_length = self_length + other_length

                joined = expected_endpoints & other_value
                joined = joined.shift(-offset)

                if joined_length not in buckets:
                    buckets[joined_length] = joined
                else:
                    buckets[joined_length] |= joined
        return BucketRangeSet(buckets)

    def extend(self, l_off: int, r_off: int) -> 'BucketRangeSet':
        length_increase = r_off - l_off
        buckets = {
            key + length_increase: value.shift(l_off)
            for key, value in self.buckets.items()
        }
        return BucketRangeSet(buckets)

    def __and__(self, other: 'BucketRangeSet') -> 'BucketRangeSet':
        common_keys = self.buckets.keys() & other.buckets.keys()
        buckets = {
            key: self.buckets[key] & other.buckets[key]
            for key in common_keys
        }
        return BucketRangeSet(buckets)

    def __or__(self, other: 'BucketRangeSet') -> 'BucketRangeSet':
        common_keys = self.buckets.keys() & other.buckets.keys()
        buckets = {
            key: self.buckets[key] | other.buckets[key]
            for key in common_keys
        }
        for key in (self.buckets.keys() - common_keys):
            buckets[key] = self.buckets[key]
        for key in (other.buckets.keys() - common_keys):
            buckets[key] = other.buckets[key]
        return BucketRangeSet(buckets)

    def __sub__(self, other: 'BucketRangeSet') -> 'BucketRangeSet':
        common_keys = self.buckets.keys() & other.buckets.keys()
        for key in common_keys:
            self.buckets[key] -= other.buckets[key]
        return self

    def __len__(self):
        return sum(len(bm) for bm in self.buckets.values())

    def __contains__(self, x: Range) -> bool:
        (l, r) = x
        length = r - l
        return length in self.buckets and l in self.buckets[length]

    def __iter__(self) -> Iterator[Range]:
        xs = [(p, p + length) for length, bm in self.buckets.items() for p in bm]
        xs.sort()
        yield from xs
