import logging
import time
import math
import sys

import networkx as nx

from collections import defaultdict
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
    binary_indexes: dict[Tuple[Feature, int, Feature], Index] = dict()

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.unary_indexes = dict()
        self.binary_indexes = dict()

        for index in Index.indexes_for(corpus):
            key = index.template.template
            match len(key):
                case 1:
                    assert key[0].offset == 0, 'Feature in unary index is expected to be 0!'
                    self.unary_indexes[key[0].feature] = index
                case 2:
                    assert key[0].offset == 0, 'Offset of first feature in binary index is expected to be 0!'
                    assert key[1].offset >= 0, 'Second feature in binary index must be positive!'
                    self.binary_indexes[(key[0].feature, key[1].offset, key[1].feature)] = index

        for unresolved_feature in set(corpus.features()) - self.unary_indexes.keys():
            LOGGER.warning(
                f'No index for feature {unresolved_feature.decode()}. Searching for it will not be possible.')

    def perform_lookup(self, lookup: Lookup, offset: int = 0) -> BitMap:
        feature_values = {  # Lookup table for encoded features and values.
            TemplateLiteral(
                atom.relative_position - offset,  # Normalize offsets for request.
                Feature(atom.key.encode())
            ): FValue(atom.value.encode())
            for atom in lookup.atoms
        }

        # This helps us find the distance between two features when selecting binary indexes.
        relevant_feature_offsets: dict[Feature, set[int]] = defaultdict(set)
        for feature in feature_values.keys():
            relevant_feature_offsets[feature.feature].add(feature.offset)

        # Graph encoding: Binary indexes are edges between (Feature:Offset) pairs.
        graph: list[Tuple[TemplateLiteral, TemplateLiteral]] = []
        for (l_feature, distance, r_feature) in self.binary_indexes.keys():
            l_offsets = relevant_feature_offsets[l_feature]
            r_offsets = relevant_feature_offsets[r_feature]

            # See if there are combinations of features with a distance that our index can serve.
            for l_offset in l_offsets:
                if l_offset + distance in r_offsets:
                    graph.append((
                        TemplateLiteral(l_offset, l_feature),
                        TemplateLiteral(l_offset + distance, r_feature)
                    ))

        # Now find the maximal matching, a combination of binary indexes maximizing requested atoms.
        best_indexes: list[Tuple[TemplateLiteral, TemplateLiteral]] = nx.maximal_matching(nx.Graph(graph))
        LOGGER.info(f'Decided lookup order: {best_indexes})')

        index_lookups: list[BitMap] = []
        fulfilled: set[TemplateLiteral] = set()
        for l_lit, r_lit in best_indexes:
            # Keep track of which parts we've already fulfilled. We might need to add some unary indexes later.
            fulfilled.add(l_lit)
            fulfilled.add(r_lit)

            # Find which index actually serves the requested features with distance.
            index = self.binary_indexes[(
                l_lit.feature,
                r_lit.offset - l_lit.offset,
                r_lit.feature
            )]

            # Finally query the index.
            l_symbol = self.corpus.get_symbol(l_lit.feature, feature_values[l_lit])
            r_symbol = self.corpus.get_symbol(r_lit.feature, feature_values[r_lit])
            index_lookups.append(
                index.search([l_symbol, r_symbol], l_lit.offset)
            )

        for unfulfilled in feature_values.keys() - fulfilled:
            index = self.unary_indexes[unfulfilled.feature]

            symbol = self.corpus.get_symbol(unfulfilled.feature, feature_values[unfulfilled])
            index_lookups.append(
                index.search([symbol], unfulfilled.offset)
            )

        index_lookups.sort(key=len)  # Start with smallest to reduce execution time.
        return fold(index_lookups, lambda a, b: a & b)

    def perform_unary_lookup(self, lookup: Lookup, offset: int = 0) -> BitMap:
        LOGGER.info(f'Decided lookup order: {[atom.key for atom in lookup.atoms]})')
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
            if atom.relative_position + offset:
                bm = bm.shift(atom.relative_position + offset)
            LOGGER.debug(f'{atom} found {len(bm)} results')
            unary_index_lookups.append(bm)

        unary_index_lookups.sort(key=len)  # Start with smallest to reduce execution time.
        return fold(unary_index_lookups, lambda a, b: a & b)


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
                lookup_positions = self.index_manager.perform_unary_lookup(node, offset=min_off)
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
