from pathlib import Path

from query import *
from evaluation import *

import tracemalloc

tracemalloc.start()
snapshot1 = tracemalloc.take_snapshot()

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

print(len(res))

snapshot2 = tracemalloc.take_snapshot()

top_stats = snapshot2.compare_to(snapshot1, 'lineno')
for stat in top_stats[:10]:
    print(stat)
