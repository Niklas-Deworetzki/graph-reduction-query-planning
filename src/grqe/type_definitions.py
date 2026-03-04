import re
from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

type Feature = str
type FValue = bytes

type Symbol = int


class IndexSignature(ABC):

    @staticmethod
    def parse(s: str) -> IndexSignature:
        if match := BinarySignature.SIGNATURE_PATTERN.fullmatch(s):
            feature1, distance_str, feature2 = match.groups()
            return BinarySignature(feature1, int(distance_str), feature2)
        else:
            return UnarySignature(s)


@dataclass(frozen=True, order=True)
class UnarySignature(IndexSignature):
    feature: Feature

    def __str__(self):
        return self.feature


@dataclass(frozen=True, init=False, order=True)
class BinarySignature(IndexSignature):
    SEPARATOR_CHAR: ClassVar[str] = '@'
    SIGNATURE_PATTERN: ClassVar[re.Pattern] = re.compile(rf'(\w+){SEPARATOR_CHAR}(\d+){SEPARATOR_CHAR}(\w+)')

    distance: int
    feature1: Feature
    feature2: Feature

    def __init__(self, feature1: Feature, distance: int, feature2: Feature):
        # We want this order in the constructor signature, but distance as the first in the derived order.
        object.__setattr__(self, 'distance', distance)
        object.__setattr__(self, 'feature1', feature1)
        object.__setattr__(self, 'feature2', feature2)

    def __str__(self):
        return BinarySignature.SEPARATOR_CHAR.join([self.feature1, str(self.distance), self.feature2])


@dataclass(frozen=True, order=True, kw_only=True)
class OffsetFeature:
    offset: int
    feature: Feature
