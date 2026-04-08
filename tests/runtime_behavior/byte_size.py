import sys

from tqdm import tqdm

from grqe.util import take
from tests.runtime_behavior.arbitrary_set import random_sets

N_SAMPLES = 100000

SERIAL_BYTECOUNT = 0
SYS_BYTECOUNT = 0

for sample in tqdm(
    iterable=take(random_sets(10, 1000, 100000), N_SAMPLES),
    total=N_SAMPLES,
    desc='Calculating'
):
    SERIAL_BYTECOUNT += sample.bytesize()
    SYS_BYTECOUNT += sys.getsizeof(sample)

print(f'serialization: {SERIAL_BYTECOUNT // N_SAMPLES}')
print(f'sys: {SYS_BYTECOUNT // N_SAMPLES}')
