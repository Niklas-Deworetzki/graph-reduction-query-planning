import re
from abc import ABC
from dataclasses import astuple, dataclass
from typing import ClassVar


type Feature = str
type FValue = bytes

type Symbol = int


class IndexSignature(ABC):
    BINARY_PATTERN: ClassVar[re.Pattern] = re.compile(r'(\w+)@(\d+)@(\w+)')

    @staticmethod
    def parse(s: str) -> IndexSignature:
        if match := IndexSignature.BINARY_PATTERN.fullmatch(s):
            feature1, distance_str, feature2 = match.groups()
            return BinarySignature(feature1, int(distance_str), feature2)
        else:
            return UnarySignature(s)

    def __iter__(self):
        return astuple(self)

@dataclass(frozen=True)
class UnarySignature(IndexSignature):
    feature: Feature

    def __str__(self):
        return self.feature


@dataclass(frozen=True)
class BinarySignature(IndexSignature):
    feature1: Feature
    distance: int
    feature2: Feature

    def __str__(self):
        return f'{self.feature1}@{self.distance}@{self.feature2}'


@dataclass(frozen=True, order=True, kw_only=True)
class OffsetFeature:
    offset: int
    feature: Feature
