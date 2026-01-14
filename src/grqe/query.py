import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generator, Optional, Sequence as Seq

from grqe.sets import BucketRangeSet

Cost = float

Value = BucketRangeSet


@dataclass(frozen=True)
class Atom:
    relative_position: int
    key: str
    value: str


class Node(ABC):
    cost: Cost = field(hash=False)
    value: Optional[Value] = field(hash=False)
    _refcount: int = field(hash=False)

    def __post_init__(self):
        self.cost = math.inf
        self.value = None
        self._refcount = 0
        for child in self.children():
            child._refcount += 1

    def is_evaluated(self) -> bool:
        return self.value is not None

    def deref_value(self, does_mutate: bool = True) -> Value:
        self._refcount -= 1
        if self._refcount and does_mutate:
            return self.value.copy()
        return self.value

    @abstractmethod
    def children(self) -> Generator['Node']:
        ...

    def flatten(self) -> Generator['Node']:
        yield self
        for child in self.children():
            yield from child.flatten()


@dataclass(unsafe_hash=True)
class Negation(Node):
    element: Node

    def children(self) -> Generator['Node']:
        yield self.element


@dataclass(unsafe_hash=True)
class Conjunction(Node):
    elements: Seq[Node]

    def children(self) -> Generator['Node']:
        yield from self.elements


@dataclass(unsafe_hash=True)
class Disjunction(Node):
    elements: Seq[Node]

    def children(self) -> Generator['Node']:
        yield from self.elements


@dataclass(unsafe_hash=True)
class Sequence(Node):
    elements: Seq[Node]

    def children(self) -> Generator['Node']:
        yield from self.elements


@dataclass(unsafe_hash=True)
class Subtraction(Node):
    lhs: Node
    rhs: Node

    def children(self) -> Generator['Node']:
        yield self.lhs
        yield self.rhs


@dataclass(unsafe_hash=True)
class Extend(Node):
    element: Node
    lhs: int
    rhs: int

    def children(self) -> Generator['Node']:
        yield self.element


@dataclass(unsafe_hash=True)
class Lookup(Node):
    atoms: Seq[Atom]

    def children(self) -> Generator['Node']:
        yield from ()


@dataclass(unsafe_hash=True)
class Alternative(Node):
    elements: Seq[Node]

    def children(self) -> Generator['Node']:
        yield from self.elements
