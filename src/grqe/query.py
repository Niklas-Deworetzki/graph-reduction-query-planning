import math
from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, ClassVar, Generator, Iterable, Optional, Sequence as Seq, override

from grqe.sets import BucketRangeSet

__all__ = [
    'Cost', 'Value', 'Atom', 'Node',
    'share',
]

Cost = float

Value = BucketRangeSet


@dataclass(frozen=True, order=True)
class Atom:
    relative_position: int
    key: str
    value: str


class Node(ABC):
    # Instance fields in this class will be provided for every node.
    # We mark them specially, so that they can be mutable and nodes are still hashable and
    #  compare equal on their fields.
    __SUPERCLASS_FIELD_MARKER__: ClassVar = field(
        init=False,  # Exclude from generated constructor.
        compare=False,  # Exclude from generated comparison.
        hash=False,  # Exclude from generated hash function.
        repr=False,  # Exclude from generated representation.
    )

    # Mark all the instance fields for proper behavior.
    cost: Cost = __SUPERCLASS_FIELD_MARKER__
    value: Optional[Value] = __SUPERCLASS_FIELD_MARKER__
    _refcount: int = __SUPERCLASS_FIELD_MARKER__

    # Set statically for each subclass to define specific subclass behavior.
    arity: ClassVar[Optional[int]] = None
    __children_iter__: ClassVar[Callable[['Node'], Iterable['Node']]]

    def __post_init__(self):
        self.cost = math.inf
        self.value = None
        self._refcount = 0
        for child in self.children():
            child._refcount += 1

    __known_subtypes__: ClassVar[list[type[Node]]] = []

    def __init_subclass__(cls):
        Node.__known_subtypes__.append(cls)

    def is_evaluated(self) -> bool:
        return self.value is not None

    def deref_value(self, does_mutate: bool = True) -> Value:
        self._refcount -= 1
        if self._refcount and does_mutate:
            return self.value.copy()
        return self.value

    def children(self) -> Generator['Node']:
        yield from self.__children_iter__()

    def flatten(self) -> Generator['Node']:
        yield self
        for child in self.children():
            yield from child.flatten()

    @classmethod
    def construct(cls, elements: Iterable[Node]) -> 'Node':
        if cls.arity is not None:
            return cls(*elements)
        return cls(tuple(elements))

    @classmethod
    def class_tag(cls) -> int:
        return Node.__known_subtypes__.index(cls)

    @cached_property
    def signature(self) -> tuple:
        tag = self.class_tag()
        child_tags = tuple(c.signature for c in self.children())
        return tag, len(child_tags), *child_tags


def node(*fields: str, var_arity: bool = False):
    """
    Decorator used to define AST nodes.
    """

    def decorate(cls):
        # Apply @dataclass(unsafe_hash=True) to the provided class.
        # This generates a fitting hash function, turning the instances hashable.
        cls = dataclass(unsafe_hash=True)(cls)

        # Set the class-level arity.
        cls.arity = None if var_arity else len(fields)

        if var_arity:
            assert len(fields) == 1, 'Variable arity operators are required to have 1 instance variable.'
            cls.__children_iter__ = lambda self: getattr(self, fields[0])
        else:
            names = tuple(fields)
            cls.__children_iter__ = lambda self: (getattr(self, n) for n in names)

        __all__.append(cls.__name__)
        return cls

    return decorate


@node()
class Lookup(Node):
    atoms: Iterable[Atom]

    @cached_property
    @override
    def signature(self) -> tuple:
        tag = self.class_tag()
        immutable_atoms = tuple(self.atoms)
        return tag, len(immutable_atoms), *immutable_atoms


@node('element')
class Negation(Node):
    element: Node


@node('elements', var_arity=True)
class Conjunction(Node):
    elements: Iterable[Node]


@node('elements', var_arity=True)
class Disjunction(Node):
    elements: Iterable[Node]


@node('elements', var_arity=True)
class Alternative(Node):
    elements: Iterable[Node]


@node('elements', var_arity=True)
class Sequence(Node):
    elements: Seq[Node]


@node('lhs', 'rhs')
class Subtraction(Node):
    lhs: Node
    rhs: Node


@node()
class Arbitrary(Node):
    pass


@node()
class Epsilon(Node):
    pass


def share(root: Node) -> Node:
    cached: list[Node] = []

    def rec(node: Node) -> Node:
        for cache in cached:
            if cache == node:
                return cache

        children = (rec(c) for c in node.children())
        if isinstance(node, Lookup):
            res = node
        else:
            res = node.construct(children)
        cached.append(res)
        return res

    return rec(root)
