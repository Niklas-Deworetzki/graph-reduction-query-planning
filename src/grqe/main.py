from pathlib import Path

from query import *
from evaluation import *

corpus = Corpus(
    'wikipedia',
    Path('/home/niklas/git/korpsearch/corpora')
)
ev = Evaluator(corpus)

hund = Lookup([
    Atom(0, "word", "äger")
])
huhund = Lookup([
    Atom(0, "pos", "NN")
])

res = ev.eval_fully(
    Alternative([hund, huhund])
)

