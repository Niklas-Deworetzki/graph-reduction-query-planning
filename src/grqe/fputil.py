from typing import Callable, List


def fold[X](xs: List[X], operator: Callable[[X, X], X]) -> X:
    res = xs[0]
    for x in xs[1:]:
        res = operator(res, x)
    return res
