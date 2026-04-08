import heapq
import struct
from collections import defaultdict
from functools import reduce
from typing import BinaryIO, Dict, Iterable, Iterator, override

from pyroaring import BitMap

from grqe.profiling import profile
from grqe.type_definitions import Range, ResultSet


class BucketRangeSet(ResultSet):
    buckets: Dict[int, BitMap]

    @override
    @classmethod
    def of(cls, s: Iterable[Range]) -> 'BucketRangeSet':
        data = defaultdict(BitMap)
        for (x, y) in s:
            data[y - x].add(x)
        return BucketRangeSet(data)

    @override
    @classmethod
    def empty(cls) -> 'BucketRangeSet':
        return BucketRangeSet({})

    def __init__(self, buckets: Dict[int, BitMap]):
        self.buckets = buckets

    @override
    @classmethod
    def conjunction(cls, *sets: 'BucketRangeSet') -> 'BucketRangeSet':
        common_keys = reduce(lambda a, b: a & b, (s.buckets.keys() for s in sets))
        buckets = {
            key: reduce(BitMap.__and__, (s.buckets[key] for s in sets))
            for key in common_keys
        }
        return BucketRangeSet(buckets)

    @override
    @classmethod
    def disjunction(cls, *sets: 'BucketRangeSet') -> 'BucketRangeSet':
        buckets = {}
        for s in sets:
            for width, bucket in s.buckets.items():
                if width not in buckets:
                    buckets[width] = bucket
                else:
                    buckets[width] = buckets[width] | bucket
        return BucketRangeSet(buckets)

    def _join(self, other: 'BucketRangeSet', distance: int = 0) -> 'BucketRangeSet':
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

    @override
    @classmethod
    def sequence(cls, *sets: 'ResultSet') -> 'ResultSet':
        return reduce(cls._join, sets)

    @override
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

    def _flatten(self) -> tuple[BitMap, BitMap]:
        covered = BitMap()
        endpoints = BitMap()

        for size, startpoints in self.buckets.items():
            for startpoint in startpoints:
                endpoint = startpoint + size
                covered.add_range(startpoint, endpoint)
                endpoints.add(endpoint)
        return covered, endpoints

    @override
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
        with profile('covered_by.mask'):
            mask, endpoints = container._flatten()

        while widths[0] <= 1:
            width, *widths = widths
            result[width] = mask & self.buckets[width]

            if not widths:  # No other lengths available.
                return BucketRangeSet(result)

        active_key = 1
        for self_size in widths:
            self_startpoints = self.buckets[self_size]
            for _ in range(self_size - active_key):
                mask -= endpoints
                endpoints = endpoints.shift(-1)
            active_key = self_size

            result[self_size] = self_startpoints & mask
        return BucketRangeSet(result)

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self.buckets.values())

    @override
    def __contains__(self, x: Range) -> bool:
        (l, r) = x
        length = r - l
        return length in self.buckets and l in self.buckets[length]

    @staticmethod
    def _iterate_bucket(length, bucket) -> Iterable[Range]:
        return ((start, start + length) for start in bucket)

    @override
    def __iter__(self) -> Iterator[Range]:
        iterators = (
            BucketRangeSet._iterate_bucket(length, bucket)
            for length, bucket in self.buckets.items()
        )
        yield from heapq.merge(*iterators)

    @override
    def serialize(self, f: BinaryIO) -> None:
        f.write(struct.pack('<I', len(self.buckets)))

        for size, bucket in self.buckets.items():
            data = bucket.serialize()
            f.write(struct.pack('<II', size, len(data)))
            f.write(data)

    @override
    @classmethod
    def deserialize(cls, f: BinaryIO) -> 'BucketRangeSet':
        bucket_count, = struct.unpack('<I', f.read(4))
        buckets = {}

        for _ in range(bucket_count):
            size, payload_size = struct.unpack('<II', f.read(4 + 4))
            payload = f.read(payload_size)
            buckets[size] = BitMap.deserialize(payload)

        return BucketRangeSet(buckets)
