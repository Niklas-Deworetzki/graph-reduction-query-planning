import re
from bisect import bisect
from datetime import datetime
from pathlib import Path
from typing import Iterable

from grqe.corpus import AnnotationsDir, Corpus
from grqe.server import query_server
from grqe.type_definitions import Symbol

queries = [
    '[form="jämställdhet"]',
    '[form!="jämställdhet"]',
    '[form="att"]',
    '[form="covid"] | [form="corona"] | [form="covid-19"]',
    '[form="läkare"] | [form="doktor"]'
    '[lemma="innebära", upos="VERB"]',
    '[lemma="innebära", upos="VERB", form!="innebär"]',
    '[upos="ADJ"] [form="områden"]',
    '[form="in"] [form="under"] [form="skinnet"]',
    '[form="NATO"] [form="medlem"] [form="Sverige"]',
    '[form="hen", upos="PRON"] | [form="hen", upos="NOUN"]',
    '[form="arbetslös", upos="ADJ"]',
    '[form="dag"] [] [upos="NOUN"]',
    '[form="skog"] [form="af"]',
    '[lemma="ge"] [form="råd"] [form="för"]',
    '[form="är"] [form="i"] [form="avsaknad"] [form="av"]',
    '[lemma="analys"] | [lemma="ute"] | [lemma="smärtsam"]',
    '[upos="NOUN"]',
    '[upos="NOUN"]+',
    '[upos="NOUN"]+ within s',
    '[form="a.*"]',
    '[form="a.*"]+',
    '[upos="ADJ"] [upos="NOUN"]',
    '[word="den"] [upos="ADJ"] [upos="NOUN"] within s',
    '[word="den"] [upos="ADJ"] [lemma="häst"]',
    '[word="den"] [upos="ADJ"] [lemma="häst"] within s'
]

queries += [' '.join(['[]'] * n) for n in range(1, 10)]


def main():
    query_server(
        Path('/home/niklas/encoded/'),
        'wikipedia',
        Path('traces') / datetime.now().strftime('%Y%m%d%H%M%S'),
        (query for query in queries),
        lambda *_: ...,
    )


def iterate_matching_symbols(feature: AnnotationsDir, pattern: re.Pattern) -> Iterable[Symbol]:
    for symbol, bytestr in enumerate(feature.symbols):
        if pattern.fullmatch(bytestr):
            yield symbol


def bisect_matching_symbols(feature: AnnotationsDir, pattern: re.Pattern) -> Iterable[Symbol]:
    regex = pattern.pattern.replace(b'.', '[^\x00]'.encode())
    regex = b'\x00' + regex + b'(?=\x00)'
    pattern = re.compile(regex)

    symboltable = feature.symbols._bytesarray._rawdata
    starts = feature.symbols._bytesarray._starts

    for match in pattern.finditer(symboltable):
        yield bisect(starts, match.start() + 1) - 1


if __name__ == '__main__':
    with Corpus('wikipedia', Path('/home/niklas/encoded/')).lock() as corpus:
        feat = corpus.feature('form')
        pat = re.compile(b'a.*')

        a = set(iterate_matching_symbols(feat, pat))
        b = set(bisect_matching_symbols(feat, pat))

        print('Missing in A')
        for missing_in_a in b - a:
            print(f'{missing_in_a} => {feat.from_symbol(missing_in_a)}')

        print('Missing in B')
        for missing_in_b in a - b:
            print(f'{missing_in_b} => {feat.from_symbol(missing_in_b)}')
