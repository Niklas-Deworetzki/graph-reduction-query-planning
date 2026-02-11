from grqe.corpus.corpus import Corpus


def frequency_of_token(corpus: Corpus, feature: str, value: bytes) -> int:
    tokens = corpus.tokens()[feature]
    symbol = tokens.to_symbol(value)

    if index := corpus.unary_index(feature):
        return len(index.search(symbol))

    return sum(symbol == at_position for at_position in tokens.values)
