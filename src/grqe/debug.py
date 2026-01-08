import logging
import time
from contextlib import AbstractContextManager
from typing import Callable

current_time: Callable[[], int] = time.perf_counter_ns


class RelativeTimeFormatter(logging.Formatter):
    start_nanos: int

    def __init__(self):
        super().__init__(fmt="[+%(relative_us)6d µs] %(levelname)s: %(message)s")
        self.reset()

    def format(self, record):
        elapsed = current_time() - self.start_nanos
        record.relative_us = elapsed // 1000
        return super().format(record)

    def reset(self):
        self.start_nanos = current_time()


def make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = RelativeTimeFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    logger.reset = formatter.reset
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
        LOGGER.debug(f'Starting {self.task}')
        return self

    def __exit__(self, *exc):
        elapsed = current_time() - self._start
        LOGGER.debug(f'Finished {self.task}')

        self.accumulated_runtime += elapsed
        self.no_runs += 1


_WATCHES = {}


def stopwatch(task: str) -> Stopwatch:
    watch = _WATCHES.get(task)
    if watch is None:
        watch = Stopwatch(task)
        _WATCHES[task] = watch
    return watch
