import logging
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


def index_of_match[T](elements: List[T], predicate: Callable[[T], bool]) -> int:
    return next(i for i, e in elements if predicate(e))


def fold[X](xs: List[X], operator: Callable[[X, X], X]) -> X:
    res = xs[0]
    for x in xs[1:]:
        res = operator(res, x)
    return res


class IndexMatcher:
    unary_indexes: dict[TemplateLiteral, Index]
    binary_indexes: dict[Tuple[TemplateLiteral, TemplateLiteral], Index]

    def __init__(self, corpus: Corpus):
        self.unary_indexes = dict()
        self.binary_indexes = dict()

        for index in Index.indexes_for(corpus):
            match key := index.template.template:
                case 1:
                    self.unary_indexes[key[0]] = index
                case 2:
                    self.binary_indexes[key] = index

        all_binary_keys = {tl for tpl in self.binary_indexes.keys() for tl in tpl}
        assert all_binary_keys.issubset(self.unary_indexes.keys())

    def perform_lookup(self, lookup: Lookup) -> BitMap:
        pass  # TODO: Use edge cover to find optimal set of indexes to use.


class Evaluator:
    set_operations = {
        Conjunction: lambda a, b: a & b,
        Disjunction: lambda a, b: a | b,
        Sequence: lambda a, b: a * b,
    }

    corpus: Corpus
    indexed_features: dict[str, UnaryIndex]

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.indexed_features = {}

        for feature in corpus.features():
            str_feature = feature.decode()
            try:
                self.indexed_features[str_feature] = UnaryIndex(
                    corpus,
                    Template.parse(f'{str_feature}:0')
                )
                print(f'Read index for feature {str_feature}.')
            except FileNotFoundError:
                print(f'No index for feature {str_feature}. Searching for it will not be possible.', file=sys.stderr)

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
                unary_index_lookups = []
                for atom in node.atoms:
                    index = self.indexed_features[atom.key]
                    symbol = self.corpus.get_symbol(
                        Feature(atom.key.encode()),
                        FValue(atom.value.encode())
                    )

                    bm = BitMap()
                    unary_index_lookups.append(bm)
                    bm |= index.lookup_smallset(symbol, symbol)
                    bm |= index.lookup_bigset(symbol, symbol)
                    if atom.relative_position:
                        bm.shift(atom.relative_position)

                node.value = fold(sorted(unary_index_lookups, key=len), lambda a, b: a & b)

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
