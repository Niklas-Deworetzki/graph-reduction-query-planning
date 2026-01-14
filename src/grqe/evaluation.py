from typing import ClassVar, Iterable, Protocol

from grqe.debug import LOGGER, current_time
from grqe.fetch import LookupStrategy
from grqe.korp import Corpus
from grqe.query import *


class Evaluator(Protocol):

    def eval_fully(self, node: Node) -> Value:
        ...


class FullEvaluator:
    set_operations: ClassVar[dict] = {
        Conjunction: lambda a, b: a & b,
        Disjunction: lambda a, b: a | b,
        Sequence: lambda a, b: a.join(b),
        Alternative: lambda a, b: a,
    }

    corpus: Corpus

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.lookup_strategy = LookupStrategy.instance(corpus)

    def eval_node(self, node: Node):
        if node.is_evaluated():
            return

        match node:
            case Conjunction() | Disjunction() | Sequence() | Alternative():
                for child in node.elements:
                    self.eval_node(child)

                operation = self.set_operations[type(node)]

                start_time = current_time()

                res = node.elements[0].value.copy()
                for element in node.elements[1:]:
                    res = operation(res, element.value.copy())

                node.value = res

            case Subtraction():
                self.eval_node(node.lhs)
                self.eval_node(node.rhs)

                start_time = current_time()

                node.value = node.lhs.value.copy() - node.rhs.value.copy()

            case Extend():
                self.eval_node(node.element)

                start_time = current_time()
                node.value = node.element.value.copy().extend(node.lhs, node.rhs)

            case Lookup():
                start_time = current_time()

                min_off = min(atom.relative_position for atom in node.atoms)
                max_off = max(atom.relative_position for atom in node.atoms)
                lookup_positions = self.lookup_strategy.perform_lookup(node, offset=min_off)
                node.value = BucketRangeSet({max_off - min_off: lookup_positions})

            case _:
                raise NotImplementedError()

        elapsed = current_time() - start_time
        node.time = elapsed

    def eval_fully(self, node: Node) -> Value:
        self.eval_node(node)
        return node.value


class CostGuidedEvaluator:
    set_operations = {
        Conjunction: lambda a, b: a & b,
        Disjunction: lambda a, b: a | b,
        Sequence: lambda a, b: a.join(b),
    }

    corpus: Corpus

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.lookup_strategy = LookupStrategy.instance(corpus)

    def eval_next_in_list(self, elements: Iterable[Node]) -> Node:
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
                lookup_positions = self.lookup_strategy.perform_lookup(node, offset=min_off)
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
