from hypothesis import given, settings, strategies as st

from grqe.query import *
from grqe.transformations import *

settings.register_profile('ast', max_examples=1000)
settings.load_profile('ast')


def immutable_lists(xs):
    return st.lists(xs, min_size=1).map(tuple)


strings = st.text(min_size=1, max_size=10)


@st.composite
def lookups(draw):
    offsets = draw(st.lists(st.integers(min_value=0, max_value=10), min_size=1))
    min_offset = min(offsets)
    normalized_offsets = (offset - min_offset for offset in offsets)

    atoms = [
        Atom(offset, draw(strings), draw(strings))
        for offset in normalized_offsets
    ]
    return Lookup(atoms)


span_lookups = st.builds(
    SpanLookup,
    st.sampled_from(['span', 's', 'p', 'text']),
    st.lists(st.builds(SpanAtom, strings, strings))
)


def nodes():
    def rec(xs):
        strats = []
        for constructing_type in Node.OPERATOR_TYPES:
            if constructing_type.arity is None:
                strats.append(st.builds(constructing_type, immutable_lists(xs)))
            elif constructing_type not in {Lookup, SpanLookup}:
                args = [xs for _ in range(constructing_type.arity)]
                strats.append(st.builds(constructing_type, *args))
        return st.one_of(strats)

    base = span_lookups | lookups()
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
    flattened = flatten_associative(node)

    for n in flattened.flatten():
        if n.is_associative:
            element_types = {type(c) for c in n.children()}
            assert type(n) not in element_types, 'Operator has not been flattened.'


@given(st.builds(Sequence, immutable_lists(nodes())))
def test_flatten_preserves_order(sequence: Sequence):
    flattened = flatten_associative(sequence)

    original_leaf_order = [c for c in sequence.flatten() if c.arity == 0]
    flattened_leaf_order = [c for c in flattened.flatten() if c.arity == 0]

    assert original_leaf_order == flattened_leaf_order, 'Operand order has changed during flattening.'


@given(nodes())
def test_sanitize_removes_epsilon(node: Node):
    sanitized = sanitize(node)

    widths = sanitized.possible_widths()
    if 0 in widths:
        assert isinstance(sanitized, Epsilon), 'Epsilon is supposed to be root after sanitize.'


test_canonical_form()
test_remove_neutral_elements()
test_flatten()
test_flatten_preserves_order()
test_sanitize_removes_epsilon()
