import math
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

from sets import BucketRangeSet

Cost = float

Value = BucketRangeSet


@dataclass(frozen=True)
class Atom:
    relative_position: int
    key: str
    value: str


class Node(ABC):
    cost: Cost
    value: Optional[Value]
    _refcount: int

    def __post_init__(self):
        self.cost = math.inf
        self.value = None
        self._refcount = 0

    def is_evaluated(self) -> bool:
        return self.value is not None

    def deref_value(self, does_mutate: bool = True) -> Value:
        self._refcount -= 1
        if self._refcount and does_mutate:
            return self.value.copy()
        return self.value


@dataclass
class Negation(Node):
    element: Node


@dataclass
class Conjunction(Node):
    elements: List[Node]


@dataclass
class Disjunction(Node):
    elements: List[Node]


@dataclass
class Sequence(Node):
    elements: List[Node]


@dataclass
class Subtraction(Node):
    lhs: Node
    rhs: Node


@dataclass
class Extend(Node):
    element: Node
    lhs: int
    rhs: int


@dataclass
class Lookup(Node):
    atoms: List[Atom]


@dataclass
class Alternative(Node):
    elements: List[Node]


def initialize_refcount(node: Node):
    node._refcount += 1
    match node:
        case Alternative() | Conjunction() | Disjunction() | Sequence():
            for child in node.elements: initialize_refcount(child)

        case Subtraction():
            initialize_refcount(node.lhs)
            initialize_refcount(node.rhs)

        case Extend():
            initialize_refcount(node.element)
