import time
from pathlib import Path

from evaluation import *
from grqe.debug import stopwatch
from parser import parse

corpus = Corpus(
    'wikipedia',
    Path('/home/niklas/git/korpsearch/corpora')
)
ev = FullEvaluator(corpus)

# A = [word@0 = "äger"];
# B = [pos@0 = "NN" pos@1 = "NN" pos@2 = "NN" pos@3 = "NN"];
# B = [pos@0 = "NN" pos@1 = "NN" pos@2 = "NN" pos@3 = "NN"]; A = alt B B;

while True:
    raw_text = input('query: ')
    with stopwatch('parsing'):
        query = parse(raw_text)
    with stopwatch('eval'):
        ev.eval_fully(query)

    time.sleep(0.1)
