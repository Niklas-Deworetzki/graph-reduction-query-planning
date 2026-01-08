import logging
import time
from contextlib import AbstractContextManager
from typing import Callable

current_time: Callable[[], int] = time.perf_counter_ns


def make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = make_logger(__name__)


class Stopwatch(AbstractContextManager):
    task: str

    no_runs: int = 0
    accumulated_runtime: int = 0

    def __init__(self, task: str):
        self.task = task
        self.no_runs = 0
        self.accumulated_runtime = 0

    def __enter__(self):
        self._start = current_time()
        return self

    def __exit__(self, *exc):
        elapsed = current_time() - self._start
        LOGGER.debug(f'Finished {self.task} in {elapsed // 1000} µs')

        self.accumulated_runtime += elapsed
        self.no_runs += 1


_WATCHES = {}


def stopwatch(task: str) -> Stopwatch:
    watch = _WATCHES.get(task)
    if watch is None:
        watch = Stopwatch(task)
        _WATCHES[task] = watch
    return watch
