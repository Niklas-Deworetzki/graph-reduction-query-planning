import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from .disk import IntArray, Symbol, SymbolArray, SymbolCollection, SymbolList, SymbolRange
from .util import FValue, Feature, add_suffix, binsearch_last


################################################################################
## Corpus

class Corpus:
    dir_suffix = '.corpus'
    features_file = 'features.cfg'
    feature_prefix = 'feature:'
    sentences_path = 'sentences'

    tokens: dict[Feature, SymbolArray]
    sentence_pointers: IntArray
    id: str
    name: str
    path: Path
    base_dir: Path

    def __init__(self, corpus: str, base_dir: Path = Path()) -> None:
        self.name = corpus
        self.base_dir = Path(base_dir)
        self.path = self.base_dir / corpus
        self.id = str(self.path)
        if self.path.suffix != self.dir_suffix:
            self.path = add_suffix(self.path, self.dir_suffix)
        self.sentence_pointers = IntArray(self.path / self.sentences_path)
        self.tokens = {
            feature: SymbolArray(self.indexpath(self.path, feature))
            for feature in self.features()
        }
        assert all(
            len(self) == len(arr) for arr in self.tokens.values()
        )

    def __str__(self) -> str:
        return f"[Corpus: {self.name}]"

    def __repr__(self) -> str:
        return f"Corpus({self.name}, base={self.base_dir})"

    def __len__(self) -> int:
        return len(self.tokens[self.features()[0]])

    def default_feature(self) -> Feature:
        return self.features()[0]

    def features(self) -> list[Feature]:
        with open(self.path / self.features_file, 'r') as IN:
            return [feat.encode() for feat in json.load(IN)]

    def symbols(self, feature: Feature) -> SymbolCollection:
        return self.tokens[feature].symbols  # type: ignore

    def get_symbol(self, feature: Feature, value: FValue) -> Symbol:
        return self.tokens[feature].symbols.to_symbol(value)

    def get_symbol_range(self, feature: Feature, prefix: FValue) -> SymbolRange:
        return self.tokens[feature].symbols.to_symbol_range(prefix)

    def get_matches(self, feature: Feature, regex: str) -> SymbolList:
        symbols = self.symbols(feature)
        return SymbolList(tuple(symbols.finditer(regex.encode())))

    def lookup_symbol(self, feature: Feature, sym: Symbol) -> FValue:
        return FValue(self.tokens[feature].symbols.to_name(sym))

    def num_sentences(self) -> int:
        return len(self.sentence_pointers) - 1

    def sentences(self) -> Iterator[range]:
        sents = self.sentence_pointers.array
        for start, end in zip(sents[1:], sents[2:]):
            yield range(start, end)
        yield range(sents[len(sents) - 1], len(self))

    def sentence_positions(self, n: int) -> range:
        sents = self.sentence_pointers.array
        start = sents[n]
        nsents = len(sents)
        end = sents[n + 1] if n + 1 < nsents else len(self)
        return range(start, end)

    def render_sentence(self, sent: int, pos: int = -1, offset: int = -1,
                        features: Sequence[Feature] = (), context: int = -1) -> str:
        if not features:
            features = [self.default_feature()]
        tokens: list[str] = []
        positions = self.sentence_positions(sent)
        for p in positions:
            if p < 0 or p >= len(self):
                continue
            if context >= 0:
                if p < pos - context:
                    continue
                if p == pos - context and p > positions.start:
                    tokens.append('...')
            if p == pos:
                tokens.append('[')
            tokens.append('/'.join(
                symbol_array.symbols.to_name(symbol_array[p]).decode()
                for feat in features
                for symbol_array in [self.tokens[feat]]
            ))
            if p == pos + offset:
                tokens.append(']')
            if context >= 0:
                if p == pos + offset + context and p < positions.stop:
                    tokens.append('...')
                    break
        return ' '.join(tokens)

    def get_sentence_from_position(self, pos: int) -> int:
        ptrs = self.sentence_pointers.array
        return binsearch_last(0, len(ptrs) - 1, pos, lambda k: ptrs[k], error=False)

    def __enter__(self) -> 'Corpus':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        for sa in self.tokens.values():
            sa.close()
        IntArray.close(self.sentence_pointers)

    @staticmethod
    def indexpath(basepath: Path, feature: Feature) -> Path:
        return basepath / (Corpus.feature_prefix + feature.decode()) / feature.decode()
