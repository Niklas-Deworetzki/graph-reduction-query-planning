from functools import reduce
from collections import defaultdict
from typing import Protocol

from pyroaring import BitMap

from grqe.profiling import commit_profiling_data, profile, current_time
from grqe.debug import LOGGER
from grqe.fetch import LookupStrategy
from grqe.corpus import Corpus
from grqe.profiling.display import format_bytesize
from grqe.query import *
from grqe.sets import BucketRangeSet


class Evaluator(Protocol):

    def eval_fully(self, node: Node) -> ResultSet:
        ...


class FullEvaluator:
    set_operations: ClassVar[dict] = {
        Conjunction: BucketRangeSet.conjunction,
        Disjunction: BucketRangeSet.disjunction,
        Sequence: BucketRangeSet.sequence,
        Alternative: lambda *a: a[0],
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

                start_time = current_time()

                operation = self.set_operations[type(node)]
                args = (el.value for el in node.elements)
                res = operation(*args)
                node.value = res

            case Arbitrary():
                start_time = current_time()

                bitmap = BitMap()
                bitmap.add_range(0, len(self.corpus) - 1)
                node.value = BucketRangeSet({
                    1: bitmap
                })

            case Repeat():
                self.eval_node(node.element)

                start_time = current_time()

                step: ResultSet = node.element.value
                accum = BucketRangeSet.empty()
                incr = step
                while len(incr):
                    accum = accum | incr
                    incr = BucketRangeSet.sequence(incr, step)
                node.value = accum

            case Contained():
                self.eval_node(node.element)
                self.eval_node(node.container)

                start_time = current_time()

                node.value = node.element.value.covered_by(node.container.value)

            case Subtraction():
                self.eval_node(node.lhs)
                self.eval_node(node.rhs)

                start_time = current_time()

                node.value = node.lhs.value.difference(node.rhs.value)

            case Lookup():
                start_time = current_time()

                min_off = min(atom.relative_position for atom in node.atoms)
                max_off = max(atom.relative_position for atom in node.atoms)
                lookup_positions = self.lookup_strategy.perform_lookup(node, offset=min_off)
                node.value = BucketRangeSet({max_off - min_off + 1: lookup_positions})

            case SpanLookup():
                start_time = current_time()
                node.value = self.lookup_strategy.lookup_span(node)

            case _:
                raise NotImplementedError()

        elapsed = current_time() - start_time
        node._profiling_info = commit_profiling_data(
            time=elapsed,
            size=str(len(node.value)),
            serialized_bytes=format_bytesize(node.value.bytesize()),
        )

    def eval_fully(self, node: Node) -> ResultSet:
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

                case Lookup():
                    node.cost = len(node.atoms)  # Lookup is 2 * log search + n intersections

        return node.cost

    def eval_fully(self, node: Node) -> ResultSet:
        while not node.is_evaluated():
            self.update_costs(node)
            self.eval_step(node)

        return node.deref_value(does_mutate=False)
