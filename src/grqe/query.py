import math
from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, ClassVar, Generator, Iterable, Optional, Sequence as Seq, override

from grqe.sets import BucketRangeSet

__all__ = [
    'Cost', 'Value', 'Atom', 'Node',
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
    __inherited_to_subclasses: ClassVar = field(
        init=False,  # Exclude from generated constructor.
        compare=False,  # Exclude from generated comparison.
        hash=False,  # Exclude from generated hash function.
        repr=False,  # Exclude from generated representation.
    )

    # Mark all the instance fields for proper behavior.
    cost: Cost = __inherited_to_subclasses
    value: Optional[Value] = __inherited_to_subclasses
    _refcount: int = __inherited_to_subclasses

    # Set statically for each subclass to define specific subclass behavior.
    arity: ClassVar[Optional[int]] = None
    is_associative: ClassVar[bool] = False
    is_commutative: ClassVar[bool] = False
    is_idempotent: ClassVar[bool] = False
    _children_iter: ClassVar[Callable[['Node'], Iterable['Node']]]

    def __post_init__(self):
        self.cost = math.inf
        self.value = None

        # Initialize reference counting.
        self._refcount = 0
        for child in self.children():
            child._refcount += 1

    # Metadata of implemented classes. Used for signature computation.
    OPERATOR_TYPES: ClassVar[list[type[Node]]] = []

    @classmethod
    def class_tag(cls) -> int:
        """Numerical tag representing the concrete node type."""
        return Node.OPERATOR_TYPES.index(cls)

    def __init_subclass__(cls):
        Node.OPERATOR_TYPES.append(cls)

    def is_evaluated(self) -> bool:
        return self.value is not None

    def deref_value(self, does_mutate: bool = True) -> Value:
        self._refcount -= 1
        if self._refcount and does_mutate:
            return self.value.copy()
        return self.value

    def children(self) -> Generator['Node']:
        """Iterate over all direct descendants."""
        yield from self._children_iter()

    def flatten(self) -> Generator['Node']:
        """Iterate over all nodes in the sub-graph with this node as its root."""
        yield self
        for child in self.children():
            yield from child.flatten()

    @classmethod
    def construct(cls, elements: Iterable[Node]) -> 'Node':
        """Create an instance of this class from an iterable of child nodes."""
        if cls.arity is not None:
            return cls(*elements)
        return cls(tuple(elements))

    def __lt__(self, other):
        return self.signature < other.signature

    def __gt__(self, other):
        return self.signature > other.signature

    def __le__(self, other):
        return self.signature <= other.signature

    def __ge__(self, other):
        return self.signature >= other.signature

    @cached_property
    def signature(self) -> tuple:
        tag = self.class_tag()
        child_tags = tuple(c.signature for c in self.children())
        return tag, len(child_tags), *child_tags


def node_type(*fields: str,
              var_arity: bool = False,
              associative: bool = False,
              commutative: bool = False,
              idempotent: bool = False,
              ):
    """Decorator used to define AST nodes."""

    def decorate(cls: type[Node]):
        # Apply @dataclass(unsafe_hash=True) to the provided class.
        # This generates a fitting hash function, turning the instances hashable.
        cls = dataclass(unsafe_hash=True)(cls)
        assert issubclass(cls, Node), '@node_type used to define AST node is not subclass of Node.'

        # Set the class-level fields.
        cls.arity = None if var_arity else len(fields)
        cls.is_associative = associative
        cls.is_commutative = commutative
        cls.is_idempotent = idempotent

        assert var_arity == (associative or commutative or idempotent), \
            'Only variable arity operators can be associative, commutative or idempotent.'

        if var_arity:
            assert len(fields) == 1, 'Variable arity operators are required to have 1 instance variable.'
            name = fields[0]
            cls._children_iter = lambda self: getattr(self, name)
        else:
            names = tuple(fields)
            cls._children_iter = lambda self: (getattr(self, n) for n in names)

        __all__.append(cls.__name__)
        return cls

    return decorate


@node_type()
class Lookup(Node):
    atoms: Iterable[Atom]

    @cached_property
    @override
    def signature(self) -> tuple:
        tag = self.class_tag()
        immutable_atoms = tuple(self.atoms)
        return tag, len(immutable_atoms), *immutable_atoms



# @node_type('element')
# class Negation(Node):
#    element: Node


@node_type('elements', var_arity=True, associative=True, commutative=True, idempotent=True)
class Conjunction(Node):
    elements: Iterable[Node]


@node_type('elements', var_arity=True, associative=True, commutative=True, idempotent=True)
class Disjunction(Node):
    elements: Iterable[Node]


@node_type('elements', var_arity=True, associative=True, commutative=True, idempotent=True)
class Alternative(Node):
    elements: Iterable[Node]


@node_type('elements', var_arity=True, associative=True)
class Sequence(Node):
    elements: Seq[Node]


@node_type('lhs', 'rhs')
class Subtraction(Node):
    lhs: Node
    rhs: Node


@node_type()
class Arbitrary(Node):
    pass


@node_type()
class Epsilon(Node):
    pass
