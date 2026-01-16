from hypothesis import given, strategies as st

from grqe.query import *
from grqe.transformations import *


def immutable_lists(xs):
    return st.lists(xs, min_size=1).map(tuple)


def atoms():
    return st.builds(
        Atom,
        st.integers(min_value=0, max_value=10),
        st.text(min_size=1, max_size=10),
        st.text(min_size=1, max_size=10),
    )


def nodes():
    def rec(xs):
        return st.builds(Conjunction, immutable_lists(xs)) \
            | st.builds(Disjunction, immutable_lists(xs)) \
            | st.builds(Alternative, immutable_lists(xs)) \
            | st.builds(Sequence, immutable_lists(xs)) \
            | st.builds(Negation, xs) \
            | st.builds(Subtraction, xs, xs)

    base = st.just(Arbitrary()) \
           | st.just(Epsilon()) \
           | st.builds(Lookup, immutable_lists(atoms()))
    return st.recursive(base, extend=rec)


@given(nodes())
def test_canonical_form(node: Node):
    def assert_ordered(iterable: Iterable, equal_ok=False):
        xs = list(iterable)
        for i in range(len(xs) - 1):
            if equal_ok:
                assert xs[i] <= xs[i + 1], 'Structure is not properly ordered.'
            else:
                assert xs[i] < xs[i + 1], 'Structure is not properly ordered.'

    def recursive_assert(n: Node):
        if isinstance(n, Lookup):
            assert_ordered(n.atoms, equal_ok=True)

        if n.is_commutative:
            assert_ordered(n.children())

        for child in n.children():
            recursive_assert(child)

    canonical_node = canonical(node)
    recursive_assert(canonical_node)


@given(nodes())
def test_remove_neutral_elements(node: Node):
    without_neutrals = remove_neutral_elements(node)

    sequences = (n for n in without_neutrals.flatten() if isinstance(n, Sequence))
    for sequence in sequences:
        for element in sequence.elements:
            assert not isinstance(element, Epsilon), 'Sequence still contains neutral element.'


@given(nodes())
def test_flatten(node: Node):
    flattened = flatten(node)

    for n in flattened.flatten():
        if n.is_associative:
            element_types = {type(c) for c in n.children()}
            assert type(n) not in element_types, 'Operator has not been flattened.'


@given(st.builds(Sequence, immutable_lists(nodes())))
def test_flatten_preserves_order(sequence: Sequence):
    flattened = flatten(sequence)

    original_leaf_order = [c for c in sequence.flatten() if c.arity == 0]
    flattened_leaf_order = [c for c in flattened.flatten() if c.arity == 0]

    assert original_leaf_order == flattened_leaf_order, 'Operand order has changed during flattening.'


test_canonical_form()
test_remove_neutral_elements()
test_flatten()
test_flatten_preserves_order()
