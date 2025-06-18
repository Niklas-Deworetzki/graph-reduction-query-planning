import logging
import time
import math
import sys
from typing import Callable

from pyroaring import BitMap

from query import *
from korp.corpus import Corpus
from korp.disk import SymbolArray
from korp.index import BinaryIndex, Index, UnaryIndex
from korp.literals import Instance, Template, TemplateLiteral
from korp.util import FValue, Feature


class RelativeTimeFormatter(logging.Formatter):
    def __init__(self, start_nanos: int = time.perf_counter_ns()):
        super().__init__(fmt="[+%(relative_us)6d µs] %(levelname)s: %(message)s")
        self.start_nanos = start_nanos

    def format(self, record):
        elapsed = time.perf_counter_ns() - self.start_nanos
        record.relative_us = elapsed // 1000
        return super().format(record)


def make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = RelativeTimeFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def index_of_match[T](elements: List[T], predicate: Callable[[T], bool]) -> int:
    return next(i for i, e in elements if predicate(e))


def fold[X](xs: List[X], operator: Callable[[X, X], X]) -> X:
    res = xs[0]
    for x in xs[1:]:
        res = operator(res, x)
    return res


LOGGER = make_logger(__name__)


class IndexMatcher:
    unary_indexes: dict[Feature, Index]
    binary_indexes: dict[Tuple[TemplateLiteral, TemplateLiteral], Index]

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.unary_indexes = dict()
        self.binary_indexes = dict()

        for index in Index.indexes_for(corpus):
            LOGGER.info(f'Discovered index {index}')
            key = index.template.template
            match len(key):
                case 1:
                    assert key[0].offset == 0, 'Unary indexes are expected to have no offset!'
                    self.unary_indexes[key[0].feature] = index
                case 2:
                    self.binary_indexes[key] = index

        for unresolved_feature in set(corpus.features()) - self.unary_indexes.keys():
            LOGGER.warning(
                f'No index for feature {unresolved_feature.decode()}. Searching for it will not be possible.')

    def perform_lookup(self, lookup: Lookup, offset: int = 0) -> BitMap:
        # TODO: Use edge cover to find optimal set of indexes to use.

        unary_index_lookups = []
        for atom in lookup.atoms:
            index = self.unary_indexes[Feature(atom.key.encode())]
            symbol = self.corpus.get_symbol(
                Feature(atom.key.encode()),
                FValue(atom.value.encode())
            )

            bm = BitMap()
            bm |= index.lookup_smallset(symbol, symbol)
            bm |= index.lookup_bigset(symbol, symbol)
            if atom.relative_position:
                bm = bm.shift(atom.relative_position)
            unary_index_lookups.append(bm)

        unary_index_lookups.sort(key=len)  # Start with smallest to reduce execution time.
        result = fold(unary_index_lookups, lambda a, b: a & b)
        if offset:
            result = result.shift(offset)
        return result


class Evaluator:
    set_operations = {
        Conjunction: lambda a, b: a & b,
        Disjunction: lambda a, b: a | b,
        Sequence: lambda a, b: a * b,
    }

    corpus: Corpus

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.index_manager = IndexMatcher(corpus)

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
                LOGGER.info(f'Starting lookup for {len(node.atoms)} features.')
                min_off = min(atom.relative_position for atom in node.atoms)
                max_off = max(atom.relative_position for atom in node.atoms)
                lookup_positions = self.index_manager.perform_lookup(node, offset=min_off)
                node.value = BucketRangeSet({max_off - min_off: lookup_positions})
                LOGGER.info(f'Lookup finished.')

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
