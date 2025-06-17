from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import NewType

from .disk import Symbol, SymbolList, SymbolRange
from .util import Feature, check_feature

################################################################################
## Literals, templates and instances

Instance = NewType('Instance', Sequence[Symbol | SymbolRange | SymbolList])


@dataclass(frozen=True, order=True)
class TemplateLiteral:
    offset: int
    feature: Feature

    def __post_init__(self) -> None:
        check_feature(self.feature)

    def __str__(self) -> str:
        return f"{self.feature.decode()}:{self.offset}"

    @staticmethod
    def parse(litstr: str) -> 'TemplateLiteral':
        try:
            featstr, offset = litstr.split(':')
            feature = Feature(featstr.lower().encode())
            return TemplateLiteral(int(offset), feature)
        except (ValueError, AssertionError):
            raise ValueError(f"Ill-formed template literal: {litstr}")


@dataclass(frozen=True, order=True, init=False)
class Template:
    size: int  # Having 'size' first means shorter templates are ordered before longer
    template: tuple[TemplateLiteral, ...]

    def __init__(self, template: Sequence[TemplateLiteral]) -> None:
        # We need to use __setattr__ because the class is frozen:
        object.__setattr__(self, 'template', tuple(template))
        object.__setattr__(self, 'size', len(self.template))
        try:
            assert self.template == tuple(sorted(set(self.template))), f"Unsorted template"
            assert len(self.template) > 0, f"Empty template"
        except AssertionError:
            raise ValueError(f"Invalid template: {self}")

    def __str__(self) -> str:
        return '+'.join(map(str, self.template))

    def __iter__(self) -> Iterator[TemplateLiteral]:
        return iter(self.template)

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def parse(template_str: str) -> 'Template':
        try:
            return Template([
                TemplateLiteral.parse(litstr)
                for litstr in template_str.split('+')
            ])
        except (ValueError, AssertionError):
            raise ValueError(
                "Ill-formed template - it should be on the form pos:0 or word:0+pos:2: " + template_str
            )
