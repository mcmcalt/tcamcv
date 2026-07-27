import random
from random import randrange
from typing import Callable, Optional

from tcamcv import Frame


def attempt_random_position(
    weights: Frame,
    probability: float,
    resolution: Optional[tuple[int, int]] = None,
    randomsource: Optional[Callable[[], float]] = None,
):
    """Get semi-random starts, intended to be queued in advance and further
    shuffled to improve randomness. This is intended to be used periodically
    over time, to continuously generate roughly-random points. In particular,
    this function may yield the same point mulitple times consecutively in
    order to better target the desired overall distribution, as a compromise
    to reduce the cycle time spent distributing values by weight."""

    if randomsource is None:
        randomsource = random.random

    xres, yres = weights.shape if resolution is None else resolution
    x = int(randomsource() * xres)
    y = int(randomsource() * yres)

    num_rolls = weights[x, y]

    # The weight in each position corresponds to the number of rolls to try
    # selecting that position, using the provided per-roll selection probability.
    for i in range(num_rolls):
        # TODO: If warranted, it might be possible to optimize this by
        # calculating the probability of m successes in n rolls. That might
        # also allow float weights.
        if randomsource() < probability:
            yield x, y


def select_at_least(
    min_count: int,
    weights: Frame,
    probability: float,
):
    count = 0

    shape = weights.shape

    while count < min_count:
        for pos in attempt_random_position(weights, probability, shape):
            count += 1
            yield pos


class Spawner:
    def __init__(self, weights: Frame, probability: float, min_queue: int):
        self.weights = weights
        self.probability = probability

        # min_queue only applies at next selection, not at rest.
        self.min_queue = min_queue
        self.queue = []

    def use(
        self,
        weights: Optional[Frame] = None,
        probability: Optional[float] = None,
        min_queue: Optional[int] = None,
    ):
        if weights is not None:
            self.weights = weights

        if probability is not None:
            self.probability = probability

        if min_queue is not None:
            self.min_queue = min_queue

        return self

    def ensure_min_length(self):
        deficit = self.min_queue - len(self.queue)
        if deficit > 0:
            # random.choices or a waiting area using counters (selection by
            # cumulative remainders?) are other options.
            self.queue.extend(select_at_least(deficit, self.weights, self.probability))

    def __next__(self):
        # Ensure the queue is long enough to mitigate repeated selections.
        self.ensure_min_length()
        return self.queue.pop(randrange(len(self.queue)))
