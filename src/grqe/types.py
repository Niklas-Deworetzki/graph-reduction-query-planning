from dataclasses import dataclass

type Feature = str
type FValue = bytes

type Symbol = int

type UnarySignature = str
type BinarySignature = tuple[str, int, str]


@dataclass(frozen=True, order=True, kw_only=True)
class OffsetFeature:
    offset: int
    feature: Feature
