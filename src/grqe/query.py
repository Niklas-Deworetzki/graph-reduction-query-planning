import collections
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property, reduce
from typing import Any, Callable, ClassVar, Generator, Iterable, Optional, Sequence as Seq, Tuple, override

__all__ = [
    'Cost', 'Value', 'Width', 'Atom', 'SpanAtom', 'Node',
]

Cost = float

Value = collections.abc.Set[Tuple[int, int]]


@dataclass(frozen=True, order=True)
class Width:
    """Datatype for static width computation."""
    widths: Optional[set[int]]

    @staticmethod
    def of(value: int) -> Width:
        return Width({value})

    @staticmethod
    def unbounded() -> Width:
        return Width(None)

    def is_unbounded(self) -> bool:
        return self.widths is None

    def has_fixed_width(self) -> bool:
        return not self.is_unbounded() and len(self.widths) == 1

    def __contains__(self, item):
        return not self.is_unbounded() and item in self.widths

    def __add__(self, other: Width) -> Width:
        if self.is_unbounded() or other.is_unbounded(): return Width.unbounded()
        return Width({
            x + y
            for x in self.widths
            for y in other.widths
        })

    def __and__(self, other: Width) -> Width:
        if self.is_unbounded() and other.is_unbounded(): return Width.unbounded()
        if self.is_unbounded(): return other
        if other.is_unbounded(): return self
        return Width(self.widths & other.widths)

    def __or__(self, other: Width) -> Width:
        if self.is_unbounded() or other.is_unbounded(): return Width.unbounded()
        return Width(self.widths | other.widths)

    def __len__(self):
        assert not self.is_unbounded(), 'Cannot iterate over unbounded set of widths.'
        return len(self.widths)

    def __iter__(self):
        assert not self.is_unbounded(), 'Cannot iterate over unbounded set of widths.'
        yield from self.widths


@dataclass(frozen=True, order=True)
class Atom:
    relative_position: int
    key: str
    value: str

    def shift(self, offset: int) -> Atom:
        return Atom(self.relative_position + offset, self.key, self.value)


@dataclass(frozen=True, order=True)
class SpanAtom:
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
    _profiling_info: Optional[dict[str, Any]] = __inherited_to_subclasses
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
        self._profiling_info = None

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
        super().__init_subclass__()
        Node.OPERATOR_TYPES.append(cls)

        possible_widths_is_overridden = cls.possible_widths is not Node.possible_widths
        width_is_overridden = cls.width is not Node.width
        if not (possible_widths_is_overridden or width_is_overridden):
            raise TypeError(f'{cls.__name__} must override {Node.possible_widths.__name__} or {Node.width.__name__}.')

    @abstractmethod
    def possible_widths(self) -> Width:
        """Returns all possible widths for results of this node."""
        raise NotImplementedError()

    def has_fixed_width(self) -> bool:
        """True, if there is only one possible width for results of the given node."""
        return self.possible_widths().has_fixed_width()

    def width(self) -> int:
        """Returns the width of this node or raises an exception if this node does not have a fixed width."""
        (fixed_width,) = self.possible_widths()
        return fixed_width

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
        """Cached property used to order instances of Node."""
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
        assert issubclass(cls, Node), f'@{node_type.__name__} must be used on a subclass of {Node.__name__}'

        # Set the class-level fields.
        cls.arity = None if var_arity else len(fields)
        cls.is_associative = associative
        cls.is_commutative = commutative
        cls.is_idempotent = idempotent

        assert not (associative or commutative or idempotent) or var_arity, \
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

    def __post_init__(self):
        super().__post_init__()
        assert any(atom.relative_position == 0 for atom in self.atoms), 'At least one atom must be at offset 0.'
        assert all(atom.relative_position >= 0 for atom in self.atoms), 'All atoms must have a positive offset.'

    @cached_property
    @override
    def signature(self) -> tuple:
        tag = self.class_tag()
        immutable_atoms = tuple(self.atoms)
        return tag, len(immutable_atoms), *immutable_atoms

    @override
    def possible_widths(self) -> Width:
        max_offset = max(atom.relative_position for atom in self.atoms)
        min_offset = min(atom.relative_position for atom in self.atoms)
        return Width.of((max_offset - min_offset) + 1)


@node_type()
class SpanLookup(Node):
    span: str
    atoms: Iterable[SpanAtom]

    @cached_property
    @override
    def signature(self) -> tuple:
        tag = self.class_tag()
        immutable_atoms = tuple(self.atoms)
        return tag, self.span, len(immutable_atoms), *immutable_atoms

    @override
    def possible_widths(self) -> Width:
        return Width.unbounded()


@node_type('elements', var_arity=True, associative=True, commutative=True, idempotent=True)
class Conjunction(Node):
    elements: Iterable[Node]

    @override
    def possible_widths(self) -> Width:
        widths = (element.possible_widths() for element in self.elements)
        return reduce(lambda a, b: a & b, widths)


@node_type('elements', var_arity=True, associative=True, commutative=True, idempotent=True)
class Disjunction(Node):
    elements: Iterable[Node]

    @override
    def possible_widths(self) -> Width:
        widths = (element.possible_widths() for element in self.elements)
        return reduce(lambda a, b: a | b, widths)


@node_type('elements', var_arity=True, associative=True, commutative=True, idempotent=True)
class Alternative(Node):
    elements: Iterable[Node]

    @override
    def possible_widths(self) -> Width:
        widths = (element.possible_widths() for element in self.elements)
        return reduce(lambda a, b: a | b, widths)


@node_type('elements', var_arity=True, associative=True)
class Sequence(Node):
    elements: Seq[Node]

    @override
    def possible_widths(self) -> Width:
        widths = (element.possible_widths() for element in self.elements)
        return reduce(lambda a, b: a + b, widths, initial=Width.of(0))


@node_type('lhs', 'rhs')
class Subtraction(Node):
    lhs: Node
    rhs: Node

    @override
    def possible_widths(self) -> Width:
        return self.lhs.possible_widths()


@node_type()
class Arbitrary(Node):

    @override
    def possible_widths(self) -> Width:
        return Width.of(1)


@node_type()
class Epsilon(Node):

    @override
    def possible_widths(self) -> Width:
        return Width.of(0)


@node_type('element')
class Repeat(Node):
    element: Node

    @override
    def possible_widths(self) -> Width:
        element_widths = self.element.possible_widths()
        if not element_widths.is_unbounded() and len(element_widths) == 0:
            return element_widths
        return Width.unbounded()


@node_type('element', 'container')
class Contained(Node):
    element: Node
    container: Node

    @override
    def possible_widths(self) -> Width:
        element_widths = self.element.possible_widths()
        container_widths = self.container.possible_widths()
        if element_widths.is_unbounded() or container_widths.is_unbounded():
            return element_widths

        possible_widths = {
            ew for ew in element_widths
            if any(ew < cw for cw in container_widths)
        }
        return Width(possible_widths)
