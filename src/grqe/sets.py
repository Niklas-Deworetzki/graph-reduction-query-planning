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

    @staticmethod
    def empty() -> 'BucketRangeSet':
        return BucketRangeSet({})

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

        # Just copy the reference for all non-shared keys.
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

        # Non shared keys are not modified.
        for key in (self.buckets.keys() - common_keys):
            buckets[key] = self.buckets[key]
        return BucketRangeSet(buckets)

    def _flatten(self, excluded_suffix: int = 0) -> BitMap:
        result = BitMap()
        for size, startpoints in self.buckets.items():
            for startpoint in startpoints:
                endpoint = startpoint + size - excluded_suffix
                if endpoint >= startpoint:
                    result.add_range(startpoint, endpoint + 1)  # +1 because end is exclusive.
        return result

    def covered_by(self, container: 'BucketRangeSet') -> 'BucketRangeSet':
        # This implementation assumes that regions in the container are non-overlapping.
        # We then build a mask of all the possible positions in which a length-n match can start
        #  and be contained in the container. (That is [c.start, c.end - n] for each c in container)
        # As this has at least one 0 at the end of each span in the container, we can then use
        #  shift and & to quickly compute a mask for n + 1 length matches; iterating all
        #  match lengths from smallest to biggest.

        if not self.buckets:
            return BucketRangeSet.empty()

        result = {}

        widths = sorted(self.buckets.keys())
        if widths[0] == 0:
            # Special case for length 0, because spans are not 0-terminated in the mask here.
            widths = widths[1:]
            result[0] = container._flatten() & self.buckets[0]

            if not widths:  # No other lengths available, just length-0 done.
                return BucketRangeSet(result)

        active_key = widths[0]
        mask = container._flatten(active_key)
        for self_size in widths:
            self_startpoints = self.buckets[self_size]
            for _ in range(self_size - active_key):
                mask &= mask.shift(-1)
            active_key = self_size

            result[self_size] = self_startpoints & mask
        return BucketRangeSet(result)

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
