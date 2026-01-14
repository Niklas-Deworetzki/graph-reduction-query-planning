from dataclasses import dataclass
from typing import Iterable, Protocol, Tuple

import networkx as nx
from pyroaring import BitMap

from grqe.debug import LOGGER, profile
from grqe.fputil import fold
from grqe.korp import Corpus, FValue, Feature, Index
from grqe.query import Lookup


class LookupStrategy(Protocol):

    def perform_lookup(self, lookup: Lookup, offset: int = 0) -> BitMap:
        ...

    @staticmethod
    def instance(corpus: Corpus) -> 'LookupStrategy':
        return GraphBasedIndexLookup(corpus)


@dataclass(frozen=True)
class Attribute:
    feature: Feature
    offset: int
    value: FValue


type Edge = Tuple[Attribute, Attribute]
type WeightedEdge = Tuple[Attribute, Attribute, int]


class GraphBasedIndexLookup:
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
                    assert key[0].offset == 0, 'Feature in unary index is expected to be at offset 0!'
                    self.unary_indexes[key[0].feature] = index
                case 2:
                    assert key[0].offset == 0, 'Offset of first feature in binary index is expected to be 0!'
                    assert key[1].offset >= 0, 'Second feature in binary index cannot be negative!'
                    self.binary_indexes[(key[0].feature, key[1].offset, key[1].feature)] = index

        for unresolved_feature in set(corpus.features()) - self.unary_indexes.keys():
            LOGGER.warning(
                f'No index for feature {unresolved_feature.decode()}. Searching for it will not be possible.')

    def _prefetch(self, attributes: Iterable[Attribute]) -> dict[Edge, BitMap]:
        prefetched = dict()

        for a_l in attributes:
            for a_r in attributes:
                if a_l.offset > a_r.offset:
                    continue

                distance = a_r.offset - a_l.offset
                binary_index = self.binary_indexes.get((a_l.feature, distance, a_r.feature))

                if a_l is a_r:
                    symbol = self.corpus.get_symbol(a_l.feature, a_l.value)
                    index = self.unary_indexes[a_l.feature]

                    result = index.search([symbol], a_l.offset)
                    prefetched[a_l, a_l] = result

                elif binary_index:
                    symbol_1 = self.corpus.get_symbol(a_l.feature, a_l.value)
                    symbol_2 = self.corpus.get_symbol(a_r.feature, a_r.value)

                    result = binary_index.search([symbol_1, symbol_2], a_l.offset)
                    prefetched[a_l, a_r] = result

        LOGGER.debug(f'Fetched data from {len(prefetched)} indexes.')
        return prefetched

    def perform_lookup(self, lookup: Lookup, offset: int = 0) -> BitMap:
        # Precompute attributes
        attributes = [
            Attribute(
                atom.key.encode(),
                atom.relative_position - offset,
                atom.value.encode(),
            )
            for atom in lookup.atoms
        ]

        # Prefetch index lookups
        with profile('prefetching'):
            prefetched = self._prefetch(attributes)

        # Build index selection graph
        with profile('graph solving'):
            corpus_size = len(self.corpus)
            graph = nx.Graph()
            graph.add_weighted_edges_from(
                (a1, a2, corpus_size - len(r)) for (a1, a2), r in prefetched.items()
            )

            index_selection = nx.min_edge_cover(graph)

        with profile('result allocation'):
            # Actually combine results from selected indexes.
            index_lookups: list[BitMap] = []
            for a_l, a_r in index_selection:
                if a_l.offset > a_r.offset:
                    a_l, a_r = a_r, a_l

                index_lookups.append(prefetched[a_l, a_r])

            index_lookups.sort(key=len)  # Start with smallest to reduce execution time.
            return fold(index_lookups, lambda a, b: a & b)
