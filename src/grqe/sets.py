import collections
from collections import defaultdict
from typing import Dict, Iterator, Tuple

from pyroaring import BitMap

type Range = Tuple[int, int]

type RangeSet = collections.abc.Set[Range]


class BucketRangeSet(collections.abc.Set[Range]):
    buckets: Dict[int, BitMap]

    @staticmethod
    def of(s: set[Range]) -> 'BucketRangeSet':
        data = defaultdict(BitMap)
        for (x, y) in s:
            data[y - x].add(x)
        return BucketRangeSet(data)

    def __init__(self, buckets: Dict[int, BitMap]):
        self.buckets = buckets

    def copy(self) -> 'BucketRangeSet':
        buckets = {key: BitMap(value) for key, value in self.buckets.items()}
        return BucketRangeSet(buckets)

    def conjunction(self, other: 'BucketRangeSet') -> 'BucketRangeSet':
        common_keys = self.buckets.keys() & other.buckets.keys()
        buckets = {
            key: self.buckets[key] & other.buckets[key]
            for key in common_keys
        }
        return BucketRangeSet(buckets)

    def disjunction(self, other: 'BucketRangeSet') -> 'BucketRangeSet':
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

    def difference(self, other: 'BucketRangeSet') -> 'BucketRangeSet':
        common_keys = self.buckets.keys() & other.buckets.keys()
        buckets = {
            key: self.buckets[key] - other.buckets[key]
            for key in common_keys
        }
        for key in (self.buckets.keys() - common_keys):
            buckets[key] = self.buckets[key]
        return BucketRangeSet(buckets)

    def covered_by(self, container: 'BucketRangeSet') -> 'BucketRangeSet':
        buckets = {}
        for self_size, self_startpoints in self.buckets.items():
            mask = BitMap()

            for container_size, container_startpoints in container.buckets.items():
                for container_startpoint in container_startpoints:
                    endpoint = container_startpoint + container_size + 1 - self_size
                    if endpoint > container_startpoint:
                        mask.add_range(container_startpoint, endpoint)

            bucket = mask & self_startpoints
            if len(bucket) > 0:
                buckets[self_size] = bucket
        return BucketRangeSet(buckets)

    def join(self, other: 'BucketRangeSet', distance: int = 0) -> 'BucketRangeSet':
        buckets: Dict[int, BitMap] = {}
        for self_length, self_value in self.buckets.items():
            offset = self_length + distance
            expected_endpoints = self_value.shift(offset)

            for other_length, other_value in other.buckets.items():
                joined_length = self_length + other_length + distance

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
        return self.conjunction(other)

    def __or__(self, other: 'BucketRangeSet') -> 'BucketRangeSet':
        return self.disjunction(other)

    def __sub__(self, other: 'BucketRangeSet') -> 'BucketRangeSet':
        return self.difference(other)

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
