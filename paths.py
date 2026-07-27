import itertools
from math import pow
import cv2
import numpy as np
import random
from typing import Callable, Iterable, Optional

from tcamcv import Frame, FrameSource, iterframes, present_with, show_frame


def yield_random_starts(
    frame: Frame,
    probability: float,
    randomsource: Optional[Callable[[], float]] = None,
):
    # The value in each position corresponds to the number of "rolls" to try selecting that position,
    # using the provided per-roll selection probability.
    # If the chance of selection on each roll is 20%, with 5 attempts, then the
    # probability of *not* being selected is .8^5. The effective selection
    # chance is thus 1 - (1-p)^n.

    if randomsource is None:
        randomsource = random.random

    for position, value in np.ndenumerate(frame):
        skip_chance = pow(1 - probability, value)
        if randomsource() > skip_chance:
            yield position


def positions_as_frame(positions: Iterable[tuple[int, int]], shape: tuple[int, int]):
    frame = np.full(shape, 0, np.uint8)
    for pos in positions:
        frame[pos] = 255

    return frame


def yield_ascents(positions: Iterable[tuple[int, int]], values: Frame):
    shape = values.shape
    xmax = shape[0] - 1
    ymax = shape[1] - 1

    for pos in positions:
        x, y = pos

        # Build search ranges.
        # Assume positions in range, and width and height > 1.
        if x == 0:
            xrange = (0, 1)
        elif x == xmax:
            xrange = (xmax - 1, xmax)
        else:
            xrange = (x - 1, x, x + 1)

        if y == 0:
            yrange = (0, 1)
        elif y == ymax:
            yrange = (ymax - 1, ymax)
        else:
            yrange = (y - 1, y, y + 1)

        # If the initial position is higher than all its neighbors, forcing a
        # move would cause isolation unless that neighbor hasn't been traversed
        # yet and has an even higher exclusive neighbor.
        # candidates = [(x, y) for x, y in itertools.product(xrange, yrange) if x != 0 or y != 0]
        candidates = itertools.product(xrange, yrange)
        pairings = ((c, values[c]) for c in candidates)
        move_to, _ = max(pairings, key=lambda t: t[1])
        yield pos, move_to


def ascend(positions: Iterable[tuple[int, int]], values: Frame):
    # TODO: Could cache and drop values that ascend to an already-tested position.
    return {move_to for pos, move_to in yield_ascents(positions, values)}


def yield_ascent_frames(starts: Iterable[tuple[int, int]], values: Frame):
    shape = values.shape
    positions = starts

    while True:
        yield positions_as_frame(positions, shape)
        positions = ascend(positions, values)


def blend_background_with(source: FrameSource, background: Frame):
    for frame in iterframes(source):
        yield cv2.addWeighted(frame, 1, background, 1, 0)


def make_background_blender(source):
    pass


# Search will sometimes get stuck at a premature maximum or take a side path.
# TODO: Add random noise or combine with running frame averaging.
def present_ascents(
    starts: Iterable[tuple[int, int]], values: Frame, background: Optional[Frame] = None
):
    # present_with(yield_ascent_frames(starts, values), lambda blend_background_with())

    for frame in yield_ascent_frames(starts, values):
        blended = (
            cv2.addWeighted(frame, 1, background, 1, 0)
            if background is not None
            else frame
        )
        show_frame(blended)
