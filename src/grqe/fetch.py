import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from typing import Iterable, Optional, Protocol, Tuple, override

import networkx as nx
from cachetools import LRUCache
from pyroaring import BitMap

from grqe.corpus import AnnotationsDir, Corpus, IntArray, BinaryIndex, SpanDir, UnaryIndex
from grqe.profiling import profile
from grqe.debug import LOGGER
from grqe.query import Lookup, SpanLookup
from grqe.sets import BucketRangeSet, Range
from grqe.type_definitions import Feature, ResultSet, Symbol


class LookupStrategy(Protocol):

    def perform_lookup(self, lookup: Lookup, offset: int = 0) -> BitMap:
        ...

    def lookup_span(self, lookup: SpanLookup) -> ResultSet:
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


FILTER_FACTOR = 1.7


class SpanGoal(ABC):
    resolved_feature: AnnotationsDir

    def has_index(self) -> bool:
        return bool(self.resolved_feature.index)

    @abstractmethod
    def estimated_comparisons_for_instantiate(self) -> int:
        ...

    def prefers_filter(self, size_to_filter: int) -> bool:
        return self.estimated_comparisons_for_instantiate() > (size_to_filter / FILTER_FACTOR)

    def instantiate(self) -> BitMap:
        if self.has_index():
            return self.index_scan()
        else:
            return self.linear_scan()

    @abstractmethod
    def linear_scan(self) -> BitMap:
        ...

    @abstractmethod
    def index_scan(self) -> BitMap:
        ...

    @abstractmethod
    def filter(self, positions: BitMap) -> BitMap:
        ...


@dataclass(frozen=True)
class ValueSpanGoal(SpanGoal):
    resolved_feature: AnnotationsDir
    value: Symbol

    @override
    def linear_scan(self) -> BitMap:
        return linear_search(self.resolved_feature.values, self.value)

    @override
    def index_scan(self) -> BitMap:
        return self.resolved_feature.index.search(self.value)

    @override
    def filter(self, positions: BitMap) -> BitMap:
        matching_positions = BitMap()
        for position in positions:
            if self.resolved_feature.values[position] == self.value:
                matching_positions.add(position)
        return matching_positions

    def estimated_comparisons_for_instantiate(self) -> int:
        if self.has_index():
            return len(self.resolved_feature.index)
        else:
            return len(self.resolved_feature.values)

    def __repr__(self) -> str:
        bytestr = self.resolved_feature.symbols.from_symbol(self.value)
        return bytestr.decode()


@dataclass(frozen=True)
class RegexSpanGoal(SpanGoal):
    resolved_feature: AnnotationsDir
    pattern: re.Pattern

    @override
    def linear_scan(self) -> BitMap:
        matching_symbols = BitMap(iterate_matching_symbols(self.resolved_feature, self.pattern))

        matching_positions = BitMap()
        for position, symbol in enumerate(self.resolved_feature.values):
            if symbol in matching_symbols:
                matching_positions.add(position)
        return matching_positions

    @override
    def index_scan(self) -> BitMap:
        matching_positions = BitMap()
        for symbol in iterate_matching_symbols(self.resolved_feature, self.pattern):
            matching_positions |= self.resolved_feature.index.search(symbol)
        return matching_positions

    @override
    def filter(self, positions: BitMap) -> BitMap:
        matching_positions = BitMap()
        for position in positions:
            symbol = self.resolved_feature.values[position]
            bytestr = self.resolved_feature.symbols.from_symbol(symbol)
            if self.pattern.fullmatch(bytestr):
                matching_positions.add(position)
        return matching_positions

    def estimated_comparisons_for_instantiate(self) -> int:
        finding_symbols = len(self.resolved_feature.symbols)
        return finding_symbols if self.has_index() else finding_symbols + len(self.resolved_feature.values)

    def __repr__(self) -> str:
        return self.pattern.pattern.decode()


type Edge = Tuple[Attribute, Attribute]
type WeightedEdge = Tuple[Attribute, Attribute, int]


@dataclass(frozen=True)
class RegexTokenGoal:
    resolved_feature: AnnotationsDir
    offset: int
    pattern: re.Pattern

    def instantiate(self) -> BitMap:
        matching_positions = BitMap()
        matching_symbols = iterate_matching_symbols(self.resolved_feature, self.pattern)

        if index := self.resolved_feature.index:
            for symbol in matching_symbols:
                matching_positions |= index.search(symbol)

        else:
            matching_symbols = BitMap(matching_symbols)
            for position, symbol in enumerate(self.resolved_feature.values):
                if symbol in matching_symbols:
                    matching_positions.add(position)

        if self.offset:
            matching_positions.shift(-self.offset)
        return matching_positions


    def filter(self, positions: BitMap) -> BitMap:
        first_invalid_index = len(self.resolved_feature.values) - self.offset
        for position in list(positions.iter_equal_or_larger(first_invalid_index)):
            positions.discard(position)

        matching_positions = BitMap()
        for position in positions:
            symbol = self.resolved_feature.values[position + self.offset]
            bytestr = self.resolved_feature.symbols.from_symbol(symbol)

            if self.pattern.fullmatch(bytestr):
                matching_positions.add(position)
        return matching_positions

    def __repr__(self):
        return self.pattern.pattern.decode()


class Prefetch(ABC):
    values: BitMap = field(init=False)

    def __post_init__(self):
        self.values = self.materialize()

    @abstractmethod
    def materialize(self) -> BitMap:
        ...

    def estimated_cost(self) -> float:
        return len(self.values)


@dataclass(frozen=True)
class BinaryPrefetch(Prefetch):
    a_l: Attribute
    a_r: Attribute
    resolved_index: BinaryIndex

    def materialize(self) -> BitMap:
        return self.resolved_index.search(self.a_l.value, self.a_r.value, self.a_l.offset)


@dataclass(frozen=True)
class UnaryPrefetch(Prefetch):
    attribute: Attribute
    resolved_index: UnaryIndex

    def materialize(self) -> BitMap:
        return self.resolved_index.search(self.attribute.value, self.attribute.offset)


@dataclass(frozen=True)
class LinearScan(Prefetch):
    attribute: Attribute
    data: IntArray

    def materialize(self) -> BitMap:
        return linear_search(self.data, self.attribute.value, self.attribute.offset)


def linear_search(data: IntArray, symbol: Symbol, offset: int = 0) -> BitMap:
    bitmap = BitMap()
    for pos, value in enumerate(data):
        if symbol == value:
            bitmap.add(pos)

    if offset:
        bitmap.shift(-offset)
    return bitmap


def iterate_matching_symbols(feature: AnnotationsDir, pattern: re.Pattern) -> Iterable[Symbol]:
    for symbol, bytestr in enumerate(feature.symbols):
        if pattern.fullmatch(bytestr):
            yield symbol


class GraphBasedIndexLookup:
    corpus: Corpus
    unary_indexes: dict[str, UnaryIndex]
    binary_indexes: dict[tuple[str, int, str], BinaryIndex] = dict()
    features: dict[str, AnnotationsDir]

    cached_full_spans: dict[str, ResultSet]
    cached_span_lookups: LRUCache
    cached_token_lookups: LRUCache

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

        self.cached_full_spans = {}
        self.cached_span_lookups = LRUCache(maxsize=5)
        self.cached_token_lookups = LRUCache(maxsize=15)

        for unresolved_feature in set(corpus.features()) - self.unary_indexes.keys():
            LOGGER.warning(f'No index for feature {unresolved_feature}. Searching will be slow.')

    @staticmethod
    def _prepare_span_attributes(lookup: SpanLookup, span: SpanDir) -> Optional[list[SpanGoal]]:
        goals: list[SpanGoal] = []

        for atom in lookup.atoms:
            annotations = span.annotations().get(atom.key)
            if not annotations:  # Unknown attribute
                return None

            if not atom.is_regex:
                symbol = annotations.to_symbol(atom.value.encode())
                if symbol == -1:
                    return None  # Unknown value for attribute.
                goals.append(ValueSpanGoal(annotations, symbol))
            else:
                pattern = regex_to_pattern(atom.value)
                goals.append(RegexSpanGoal(annotations, pattern))

        return goals

    def span_from_cache(self, span: SpanDir, name: str) -> ResultSet:
        if name in self.cached_full_spans:
            with profile('span.from_full_cache'):
                return self.cached_full_spans[name]

        with profile('span.io'):
            materialized_result = list(span.ranges)
        with profile('span.to_bitmap'):
            result = BucketRangeSet.of(materialized_result)

        self.cached_full_spans[name] = result
        return result

    def lookup_span(self, lookup: SpanLookup) -> Iterable[Range]:
        if not (span := self.corpus.spans().get(lookup.span)):
            return []

        goals = self._prepare_span_attributes(lookup, span)
        if goals is None:
            return []

        if len(goals) == 0:
            return self.span_from_cache(span, lookup.span)

        cache_key = lookup
        if cache_key in self.cached_span_lookups:
            with profile('span.from_lookup_cache'):
                return self.cached_span_lookups[cache_key]

        with profile('span.search'):
            first_goal, *remaining_goals = sorted(goals, key=SpanGoal.estimated_comparisons_for_instantiate)
            span_ids = first_goal.instantiate()
            for goal in remaining_goals:
                with profile(f'span.{goal}'):
                    if goal.prefers_filter(len(span_ids)):
                        span_ids = goal.filter(span_ids)
                    else:
                        span_ids &= goal.instantiate()

        with profile('span.materialize'):
            materialized_result = list(span.ranges[i] for i in span_ids)
        with profile('span.to_bitmap'):
            result = BucketRangeSet.of(materialized_result)

        self.cached_span_lookups[cache_key] = result
        return result

    def _prefetch(self, attributes: Iterable[Attribute]) -> dict[Edge, Prefetch]:
        prefetched: dict[Edge, Prefetch] = dict()

        for a_l in attributes:
            for a_r in attributes:
                if a_l.offset > a_r.offset:
                    continue

                distance = a_r.offset - a_l.offset
                binary_index = self.binary_indexes.get((a_l.feature, distance, a_r.feature))

                if a_l is a_r:
                    if index := self.unary_indexes.get(a_l.feature):
                        prefetched[a_l, a_l] = UnaryPrefetch(a_l, index)
                    else:
                        data = self.features[a_l.feature].values
                        prefetched[a_l, a_r] = LinearScan(a_l, data)

                elif binary_index:
                    prefetched[a_l, a_r] = BinaryPrefetch(a_l, a_r, binary_index)

        LOGGER.debug(f'Fetched data from {len(prefetched)} indexes.')
        return prefetched

    def _prepare_attributes(self, lookup: Lookup, offset: int) -> tuple[
        bool,
        Iterable[Attribute],
        Iterable[RegexTokenGoal],
    ]:
        attributes: list[Attribute] = []
        regexes: list[RegexTokenGoal] = []

        for atom in lookup.atoms:
            annotations = self.features.get(atom.key)
            if annotations is None:  # Unknown attribute.
                return False, [], []

            if not atom.is_regex:
                symbol = annotations.to_symbol(atom.value.encode())
                if symbol == -1:  # Unknown value for attribute.
                    return False, [], []

                attributes.append(Attribute(atom.key, atom.relative_position - offset, symbol))

            else:
                pattern = regex_to_pattern(atom.value)
                regexes.append(RegexTokenGoal(annotations, atom.relative_position - offset, pattern))

        return True, attributes, regexes

    def perform_lookup(self, lookup: Lookup, offset: int = 0) -> BitMap:
        # Precompute attributes
        passed_preliminary_checks, attributes, regexes = self._prepare_attributes(lookup, offset)
        if not passed_preliminary_checks:
            return BitMap()

        cache_key = (tuple(attributes), tuple(regexes))
        if cached_result := self.cached_token_lookups.get(cache_key):
            with profile('lookup.cached'):
                return cached_result

        # Prefetch index lookups
        with profile('leaf.prefetch'):
            prefetched = self._prefetch(attributes)

        # Build index selection graph
        with profile('leaf.index_plan'):
            corpus_size = len(self.corpus)
            graph = nx.Graph()
            graph.add_weighted_edges_from(
                (a1, a2, corpus_size - r.estimated_cost())
                for (a1, a2), r in prefetched.items()
            )

            index_selection = nx.min_edge_cover(graph)

        with profile('leaf.materialize_lookups'):
            # Actually combine results from selected indexes.
            index_lookups: list[BitMap] = []
            for a_l, a_r in index_selection:
                if a_l.offset > a_r.offset:
                    a_l, a_r = a_r, a_l

                index_lookups.append(prefetched[a_l, a_r].materialize())

        if index_lookups:
            result = conjunct_bitmaps(index_lookups)
        else:
            first_regex, *regexes = regexes
            with profile(f'regex.instantiate.{first_regex}'):
                result = first_regex.instantiate()

        with profile('regex.filter'):
            for regex in regexes:
                with profile(f'regex.filter.{regex}'):
                    result = regex.filter(result)

        self.cached_token_lookups[cache_key] = result
        return result


def conjunct_bitmaps(conjuncts: Iterable[BitMap]) -> BitMap:
    sorted(conjuncts, key=len)  # Start with smallest to reduce execution time.
    return reduce(lambda a, b: a & b, conjuncts)

def regex_to_pattern(regex: str) -> re.Pattern:
    return re.compile(regex.encode())
