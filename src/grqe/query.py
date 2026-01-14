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

    def is_evaluated(self) -> bool:
        return self.value is not None

    def deref_value(self, does_mutate: bool = True) -> Value:
        self._refcount -= 1
        if self._refcount and does_mutate:
            return self.value.copy()
        return self.value



@dataclass(unsafe_hash=True)
class Negation(Node):
    element: Node



@dataclass(unsafe_hash=True)
class Conjunction(Node):
    elements: Seq[Node]



@dataclass(unsafe_hash=True)
class Disjunction(Node):
    elements: Seq[Node]



@dataclass(unsafe_hash=True)
class Sequence(Node):
    elements: Seq[Node]



@dataclass(unsafe_hash=True)
class Subtraction(Node):
    lhs: Node
    rhs: Node



@dataclass(unsafe_hash=True)
class Extend(Node):
    element: Node
    lhs: int
    rhs: int



@dataclass(unsafe_hash=True)
class Lookup(Node):
    atoms: Seq[Atom]



@dataclass(unsafe_hash=True)
class Alternative(Node):
    elements: Seq[Node]

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
