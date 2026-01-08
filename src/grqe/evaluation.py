from typing import Callable, Protocol

from grqe.fetch import LookupStrategy
from grqe.korp import Corpus
from grqe.query import *
from grqe.debug import LOGGER


def index_of_match[T](elements: List[T], predicate: Callable[[T], bool]) -> int:
    return next(i for i, e in elements if predicate(e))


def fold[X](xs: List[X], operator: Callable[[X, X], X]) -> X:
    res = xs[0]
    for x in xs[1:]:
        res = operator(res, x)
    return res


class Evaluator(Protocol):

    def eval_fully(self, node: Node) -> Value:
        ...


class CostGuidedEvaluator:
    set_operations = {
        Conjunction: lambda a, b: a & b,
        Disjunction: lambda a, b: a | b,
        Sequence: lambda a, b: a.join(b),
    }

    corpus: Corpus

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.index_manager = LookupStrategy.instance(corpus)

    def eval_next_in_list(self, elements: List[Node]) -> Node:
        target = min(elements, key=lambda e: e.cost if not e.is_evaluated() else math.inf)
        self.eval_step(target)
        return target

    def eval_step(self, node: Node):
        match node:
            case Alternative():
                result = self.eval_next_in_list(node.elements)
                if result.is_evaluated():
                    node.value = result.deref_value()

            case Conjunction() | Disjunction() | Sequence():
                if all(e.is_evaluated() for e in node.elements):
                    operation = self.set_operations[type(node)]

                    res = node.elements[0].deref_value()
                    for element in node.elements[1:]:
                        res = operation(res, element.deref_value(does_mutate=False))

                    node.value = res
                else:
                    self.eval_next_in_list(node.elements)

            case Subtraction():
                if node.lhs.is_evaluated() and node.rhs.is_evaluated():
                    node.value = node.lhs.deref_value() - node.rhs.deref_value(does_mutate=False)
                elif node.lhs.is_evaluated() or node.rhs.cost < node.lhs.cost:
                    self.eval_step(node.rhs)
                else:
                    self.eval_step(node.lhs)

            case Extend():
                if node.element.is_evaluated():
                    node.value = node.element.deref_value().extend(node.lhs, node.rhs)
                else:
                    self.eval_step(node.element)

            case Lookup():
                LOGGER.debug(f'Starting lookup for {len(node.atoms)} features.')
                min_off = min(atom.relative_position for atom in node.atoms)
                max_off = max(atom.relative_position for atom in node.atoms)
                lookup_positions = self.index_manager.perform_lookup(node, offset=min_off)
                node.value = BucketRangeSet({max_off - min_off: lookup_positions})
                LOGGER.debug(f'Lookup finished.')

            case _:
                raise NotImplementedError()

    # Max / Min cost + size?
    def update_costs(self, node: Node) -> float:
        if node.is_evaluated():
            node.cost = 0

        else:
            match node:
                case Alternative():
                    node.cost = min(node.elements, key=self.update_costs).cost

                case Conjunction() | Disjunction() | Sequence():
                    node.cost = sum(self.update_costs(el) for el in node.elements) + 8 * len(node.elements)

                case Subtraction():
                    node.cost = self.update_costs(node.lhs) + self.update_costs(node.rhs) + 15

                case Extend():
                    node.cost = 2 + self.update_costs(node.element)

                case Lookup():
                    node.cost = len(node.atoms)  # Lookup is 2 * log search + n intersections

        return node.cost

    def eval_fully(self, node: Node) -> Value:
        initialize_refcount(node)

        while not node.is_evaluated():
            self.update_costs(node)
            self.eval_step(node)

        return node.deref_value(does_mutate=False)

# max_matching, no weights, lazy fetching
# [+  6802 µs] INFO: Starting lookup for 4 features.
# [+  6866 µs] DEBUG: Found 5 applicable indexes.
# [+  7471 µs] INFO: Decided lookup order: {(TemplateLiteral(offset=0, feature=b'word'), TemplateLiteral(offset=2, feature=b'pos')), (TemplateLiteral(offset=0, feature=b'pos'), TemplateLiteral(offset=1, feature=b'pos'))})
# [+ 18477 µs] INFO: Lookup finished.

# weighted max matching, weights from eager fetching
# [+  5112 µs] INFO: Starting lookup for 4 features.
# [+ 34984 µs] DEBUG: Found 4 applicable indexes.
# [+ 35868 µs] INFO: Decided lookup order: {(TemplateLiteral(offset=1, feature=b'pos'), TemplateLiteral(offset=0, feature=b'pos')), (TemplateLiteral(offset=0, feature=b'word'), TemplateLiteral(offset=2, feature=b'pos'))})
# [+ 38497 µs] INFO: Lookup finished.
