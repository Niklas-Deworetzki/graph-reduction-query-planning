from dataclasses import dataclass
from functools import reduce
from typing import Iterable, Optional, Protocol, Tuple

import networkx as nx
from pyroaring import BitMap

from grqe.corpus import AnnotationsDir, Corpus, IntArray, BinaryIndex, SpanDir, UnaryIndex
from grqe.debug import LOGGER, profile
from grqe.query import Lookup, SpanLookup
from grqe.sets import Range
from grqe.type_definitions import Feature, Symbol


class LookupStrategy(Protocol):

    def perform_lookup(self, lookup: Lookup, offset: int = 0) -> BitMap:
        ...

    def lookup_span(self, lookup: SpanLookup) -> Iterable[Range]:
        ...

    @staticmethod
    def instance(corpus: Corpus) -> 'LookupStrategy':
        return GraphBasedIndexLookup(corpus)


@dataclass(frozen=True)
class Attribute:
    """feature @ offset = value"""
    feature: Feature
    offset: int
    value: Symbol


@dataclass(frozen=True)
class SpanAttribute:
    resolved_feature: AnnotationsDir
    value: Symbol


type Edge = Tuple[Attribute, Attribute]
type WeightedEdge = Tuple[Attribute, Attribute, int]


def linear_search(data: IntArray, symbol: Symbol) -> BitMap:
    bitmap = BitMap()
    for pos, value in enumerate(data):
        if symbol == value:
            bitmap.add(pos)
    return bitmap


class GraphBasedIndexLookup:
    corpus: Corpus
    unary_indexes: dict[str, UnaryIndex]
    binary_indexes: dict[tuple[str, int, str], BinaryIndex] = dict()
    features: dict[str, AnnotationsDir]

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.features = corpus.tokens()
        self.unary_indexes = {
            key.feature: value
            for key, value in corpus.unary_indexes().items()
        }
        self.binary_indexes = {
            (key.feature1, key.distance, key.feature2): value
            for key, value in corpus.binary_indexes().items()
        }

        for unresolved_feature in set(corpus.features()) - self.unary_indexes.keys():
            LOGGER.warning(f'No index for feature {unresolved_feature}. Searching will be slow.')

    @staticmethod
    def _prepare_span_attributes(lookup: SpanLookup, span: SpanDir) -> Optional[list[SpanAttribute]]:
        attributes = []
        for atom in lookup.atoms:
            annotations = span.annotations().get(atom.key)
            if not annotations:  # Unknown attribute
                return None

            symbol = annotations.to_symbol(atom.value.encode())
            if symbol == -1:
                return None  # Unknown value for attribute.
            attributes.append(SpanAttribute(annotations, symbol))
        return attributes

    def lookup_span(self, lookup: SpanLookup) -> Iterable[Range]:
        if not (span := self.corpus.spans().get(lookup.span)):
            return []

        attributes = self._prepare_span_attributes(lookup, span)
        if attributes is None:
            return []
        elif len(attributes) == 0:
            return span.ranges

        with profile('span fetching'):
            matching_span_ids = []
            for attribute in attributes:
                if attribute.resolved_feature.index is not None:
                    span_ids = attribute.resolved_feature.index.search(attribute.value)
                else:
                    span_ids = linear_search(attribute.resolved_feature.values, attribute.value)
                matching_span_ids.append(span_ids)

            span_ids = conjunct_bitmaps(matching_span_ids)
            return (span.ranges[i] for i in span_ids)

    def _prefetch(self, attributes: Iterable[Attribute]) -> dict[Edge, BitMap]:
        prefetched = dict()

        for a_l in attributes:
            for a_r in attributes:
                if a_l.offset > a_r.offset:
                    continue

                distance = a_r.offset - a_l.offset
                binary_index = self.binary_indexes.get((a_l.feature, distance, a_r.feature))

                if a_l is a_r:
                    if index := self.unary_indexes.get(a_l.feature):
                        result = index.search(a_l.value, a_l.offset)
                        prefetched[a_l, a_l] = result
                    else:
                        data = self.features[a_l.feature].values
                        prefetched[a_l, a_r] = linear_search(data, a_l.value)

                elif binary_index:
                    result = binary_index.search(a_l.value, a_r.value, a_l.offset)
                    prefetched[a_l, a_r] = result

        LOGGER.debug(f'Fetched data from {len(prefetched)} indexes.')
        return prefetched

    def _prepare_attributes(self, lookup: Lookup, offset: int) -> Optional[list[Attribute]]:
        attributes: list[Attribute] = []
        for atom in lookup.atoms:
            annotations = self.features.get(atom.key)
            if annotations is None:  # Unknown attribute.
                return None

            symbol = annotations.to_symbol(atom.value.encode())
            if symbol == -1:  # Unknown value for attribute.
                return None

            attributes.append(Attribute(atom.key, atom.relative_position - offset, symbol))
        return attributes

    def perform_lookup(self, lookup: Lookup, offset: int = 0) -> BitMap:
        # Precompute attributes
        attributes: list[Attribute] = self._prepare_attributes(lookup, offset)
        if attributes is None:
            return BitMap()

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

            return conjunct_bitmaps(index_lookups)


def conjunct_bitmaps(conjuncts: list[BitMap]) -> BitMap:
    conjuncts.sort(key=len)  # Start with smallest to reduce execution time.
    return reduce(lambda a, b: a & b, conjuncts)
