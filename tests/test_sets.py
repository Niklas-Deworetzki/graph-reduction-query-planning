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


@given(set_of_ranges(), set_of_ranges(), st.integers(min_value=-100, max_value=100), st.integers(min_value=-100, max_value=100))
def test_extend(a, b, l_off, r_off):
    expected = {
        (l - l_off, r + r_off)
        for (l, r) in a
    }
    test_set_operation(a, b, expected, lambda a, b: a.extend(b, l_off, r_off))

test_conjunction()
test_disjunction()
test_difference()
test_join()
