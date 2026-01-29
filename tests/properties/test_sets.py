from hypothesis import given, strategies as st

from grqe.sets import BucketRangeSet


def ranges(min_value=0, max_value=100000, max_diff=100):
    def to_range(x: int):
        return st.tuples(
            st.just(x),
            st.integers(
                min_value=x,
                max_value=min(max_value, x + max_diff)
            )
        )

    return st.integers(min_value, max_value).flatmap(to_range)


def set_of_ranges():
    return st.sets(ranges())


def test_set_operation(set_a, set_b, set_expected, operation):
    rset_a = BucketRangeSet.of(set_a)
    rset_b = BucketRangeSet.of(set_b)

    assert set_expected == set(operation(rset_a, rset_b)), 'Operation does not produce correct result.'
    assert set(rset_a) == set_a, 'Operation modified left operand.'
    assert set(rset_b) == set_b, 'Operation modified right operand.'


@given(set_of_ranges(), set_of_ranges())
def test_conjunction(a, b):
    test_set_operation(a, b, a & b, BucketRangeSet.conjunction)


@given(set_of_ranges(), set_of_ranges())
def test_disjunction(a, b):
    test_set_operation(a, b, a | b, BucketRangeSet.disjunction)


@given(set_of_ranges(), set_of_ranges())
def test_difference(a, b):
    test_set_operation(a, b, a - b, BucketRangeSet.difference)


@given(set_of_ranges(), set_of_ranges(), st.integers(min_value=0, max_value=100))
def test_join(a, b, distance):
    expected = {
        (ll, rr)
        for (ll, lr) in a
        for (rl, rr) in b
        if rl - lr == distance
    }
    test_set_operation(a, b, expected, lambda a, b: a.join(b, distance))


@given(set_of_ranges(), set_of_ranges(),
       st.integers(min_value=-100, max_value=100), st.integers(min_value=-100, max_value=100))
def test_extend(a, b, l_off, r_off):
    expected = {
        (l - l_off, r + r_off)
        for (l, r) in a
    }
    test_set_operation(a, b, expected, lambda a, b: a.extend(b, l_off, r_off))


def test_associativity(a, b, c, operation):
    rset_a = BucketRangeSet.of(a)
    rset_b = BucketRangeSet.of(b)
    rset_c = BucketRangeSet.of(c)

    res1 = operation(rset_a, operation(rset_b, rset_c))
    res2 = operation(rset_a, operation(rset_b, rset_c))
    assert res1 == res2, f'{operation.__name__} is not associative: a + (b + c) =/= (a + b) + c'


def test_commutativity(a, b, operation):
    rset_a = BucketRangeSet.of(a)
    rset_b = BucketRangeSet.of(b)

    res1 = operation(rset_a, rset_b)
    res2 = operation(rset_b, rset_a)
    assert res1 == res2, f'{operation.__name__} is not commutative: a + b =/= b + a'


def test_idempotence(a, operation):
    rset_a = BucketRangeSet.of(a)

    res = operation(rset_a, rset_a)
    assert rset_a == res, f'{operation.__name__} is not idempotent: a + a =/= a'


def test_distributivity(a, b, c, times, plus):
    rset_a = BucketRangeSet.of(a)
    rset_b = BucketRangeSet.of(b)
    rset_c = BucketRangeSet.of(c)

    res1 = plus(
        times(rset_a, rset_b),
        times(rset_a, rset_c),
    )
    res2 = times(
        rset_a,
        plus(rset_b, rset_c),
    )
    assert res1 == res2, f'{times.__name__} does not distribute over {plus.__name__}: (a * b) + (a * c) =/= a * (b + c)'


TEST_OPERATOR_SEMANTICS = [
    test_conjunction,
    test_disjunction,
    test_difference,
    test_join,
]


@given(set_of_ranges(), set_of_ranges(), set_of_ranges())
def test_associative_conjunction(a, b, c):
    test_associativity(a, b, c, BucketRangeSet.conjunction)


@given(set_of_ranges(), set_of_ranges(), set_of_ranges())
def test_associative_disjunction(a, b, c):
    test_associativity(a, b, c, BucketRangeSet.disjunction)


@given(set_of_ranges(), set_of_ranges(), set_of_ranges())
def test_associative_sequence(a, b, c):
    test_associativity(a, b, c, BucketRangeSet.join)


@given(set_of_ranges(), set_of_ranges())
def test_commutative_conjunction(a, b):
    test_commutativity(a, b, BucketRangeSet.conjunction)


@given(set_of_ranges(), set_of_ranges())
def test_commutative_disjunction(a, b):
    test_commutativity(a, b, BucketRangeSet.disjunction)


@given(set_of_ranges())
def test_idempotent_conjunction(a):
    test_idempotence(a, BucketRangeSet.conjunction)


@given(set_of_ranges())
def test_idempotent_disjunction(a):
    test_idempotence(a, BucketRangeSet.disjunction)


@given(set_of_ranges(), set_of_ranges(), set_of_ranges())
def test_conjunction_distributes_over_disjunction(a, b, c):
    test_distributivity(a, b, c, BucketRangeSet.conjunction, BucketRangeSet.disjunction)


@given(set_of_ranges(), set_of_ranges(), set_of_ranges())
def test_disjunction_distributes_over_conjunction(a, b, c):
    test_distributivity(a, b, c, BucketRangeSet.disjunction, BucketRangeSet.conjunction)


@given(set_of_ranges(), set_of_ranges(), set_of_ranges())
def test_sequence_distributes_over_conjunction(a, b, c):
    test_distributivity(a, b, c, BucketRangeSet.join, BucketRangeSet.conjunction)


@given(set_of_ranges(), set_of_ranges(), set_of_ranges())
def test_sequence_distributes_over_disjunction(a, b, c):
    test_distributivity(a, b, c, BucketRangeSet.join, BucketRangeSet.disjunction)


TEST_OPERATOR_LAWS = [
    test_associative_conjunction,
    test_associative_disjunction,
    test_associative_sequence,
    test_commutative_conjunction,
    test_commutative_disjunction,
    test_idempotent_conjunction,
    test_idempotent_disjunction,
    test_conjunction_distributes_over_disjunction,
    test_disjunction_distributes_over_conjunction,
    test_sequence_distributes_over_conjunction,
    test_sequence_distributes_over_disjunction,
]

for test in TEST_OPERATOR_SEMANTICS:
    test()

for test in TEST_OPERATOR_LAWS:
    test()
