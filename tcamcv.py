"""Traffic camera computer vision experimentation"""

from collections import deque
from itertools import islice

import cv2
from cv2 import VideoCapture
import numpy as np
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

# A video frame.
Frame = np.ndarray

# A (presumed) source of video frames.
FrameSource = VideoCapture | Iterable[Frame] | Path | str


class Playlist:
    """A list of video files."""

    def __init__(self, files: Iterable[str | Path]):
        self.files = [Path(file) for file in files]

    def __iter__(self):
        for file in self.files:
            vc = cv2.VideoCapture(file)
            yield from iterframes(vc)

    @classmethod
    def rglob(cls, base: Path | str, pattern: str = "*"):
        return cls(Path(base).rglob(pattern))


def show_frame(frame, wait=1, title="Frame"):
    cv2.imshow(title if title else "Frame", frame)
    return cv2.waitKey(wait if wait is not None else 1)


def itervc(vc: VideoCapture, ms: Optional[int] = None) -> Iterator[Frame]:
    if ms is not None:
        vc.set(cv2.CAP_PROP_POS_MSEC, ms)

    while True:
        status, frame = vc.read()
        if status:
            yield frame
        else:
            return


def iterframes(source: FrameSource, start: Optional[int] = None) -> Iterator[Frame]:
    if isinstance(source, (str | Path)):
        # FIXME: Handle bad path appropriately.
        source = cv2.VideoCapture(source)

    if isinstance(source, VideoCapture):
        return itervc(source, start)
    else:
        return iter(source)


def iter_gray_frames(source: FrameSource, conversion=cv2.COLOR_BGR2GRAY):
    for frame in iterframes(source):
        yield cv2.cvtColor(frame, conversion)


itergrays = iter_gray_frames
grays = iter_gray_frames


class Differencer:
    def __init__(self, prior_frame: Optional[Frame] = None):
        self._prior_frame = prior_frame

    def feed(self, frame: Frame) -> Frame | None:
        """Send priming or running inputs."""
        if self._prior_frame is None:
            self._prior_frame = frame
            return None
        else:
            return self.next(frame)

    def prime(self, frame: Frame):
        if self._prior_frame is None:
            self._prior_frame = frame
        else:
            raise RuntimeError("Differencer has already been primed")

    def prime_from(self, source: Iterator[Frame]):
        """Consume items from the source until primed."""
        consumed = 0
        while not self.primed():
            self.prime(next(source))
            consumed += 1
        return consumed

    def next(self, frame: Frame):
        """Send running inputs (after priming) and get resulting outputs."""
        assert self._prior_frame is not None
        diff = cv2.absdiff(frame, self._prior_frame)
        self._prior_frame = frame
        return diff

    def primed(self):
        return self._prior_frame is not None


def iter_deltas(source: FrameSource, first_frame: Optional[Frame] = None):
    frames = iterframes(source)
    differ = Differencer(first_frame)
    differ.prime_from(frames)

    for frame in frames:
        yield differ.next(frame)


def present_with(
    source: FrameSource,
    func: Callable[[FrameSource], Iterator[Frame]],
    wait: Optional[int] = None,
    **kwargs
):
    wait = 1 if wait is None else wait
    for frame in func(source, **kwargs):
        show_frame(frame, wait)


def present_frames(source: FrameSource, wait: Optional[int] = None):
    present_with(source, iterframes, wait)


def present_diffs(source: FrameSource, wait: Optional[int] = None):
    present_with(source, iter_deltas, wait)


def blend(a, b, alpha: float):
    return cv2.addWeighted(a, alpha, b, 1 - alpha, 0)


def mask_frame(mask, frame):
    return cv2.bitwise_and(frame, frame, mask=mask)


# recommend alpha = .97 for pathways? .8 to .85 for tracking?
def yield_blends(source: FrameSource, alpha: float):
    iterator = iter_deltas(source)
    prior = next(iterator)

    for frame in iterator:
        blended = blend(prior, frame, alpha)
        yield blended
        prior = blended


def present_blends(source: FrameSource, alpha: float):
    present_with(source, yield_blends, alpha=alpha)


def blur(frame: Frame, radius: int = 20):
    diameter = radius * 2 + 1
    return cv2.GaussianBlur(frame, (diameter, diameter), 0)


# TODO: Use high-persistence frame blend to get pathways, then use
# that to shape the detection blur along lane and reduce adjacent lane influence?


def iter_blurred_blends(source: FrameSource, alpha: float = 0.8, blursize: int = 20):
    diameter = blursize * 2 + 1
    for blend in yield_blends(source, alpha):
        yield cv2.GaussianBlur(blend, (diameter, diameter), 0)


def present_blurred_blends(source: FrameSource, alpha: float = 0.8, blursize: int = 20):
    present_with(
        source,
        iter_blurred_blends,
        alpha=alpha,
        blursize=blursize,
    )


def blur_peaks(frame, blursize=2):
    diameter = blursize * 2 + 1
    blurred = cv2.GaussianBlur(frame, (diameter, diameter), 0)
    # TODO: Fix range underflow.
    return frame - blurred


# If frames are (blended) frame-differences, positive values represent
# activity and negative values represent stable areas around activity.
def delta_blur(frame, blursize1=20, blursize2=80):
    diameter1 = blursize1 * 2 + 1
    diameter2 = blursize2 * 2 + 1
    blurred1 = cv2.GaussianBlur(frame, (diameter1, diameter1), 0)
    blurred2 = cv2.GaussianBlur(frame, (diameter2, diameter2), 0)
    # Would it be better to rescale either blur's output range separately first?
    return blurred1.astype(np.int16) - blurred2.astype(np.int16)


class FrameHistory:
    def __init__(self, length: int, source: FrameSource | None = None):
        self.history = deque[Frame](
            # The deque constructor would iterate the whole input and only
            # save the end.
            [] if source is None else islice(iterframes(source), length),
            length,
        )

    def save(self, frame: Frame):
        self.history.append(frame)

    def get(self, age: int | None = None, fill: bool = True):
        try:
            return self.history[0 if age is None else -age - 1]
        except IndexError as err:
            if fill:
                # might not have any frames though.
                return self.history[0]


class MaskGenerator:
    def __init__(
        self,
        first_frame: Frame,
        alpha: float = 0.8,  # .1 to .12 produce an interesting snowplow-like effect
        innerblur: int = 15,
        outerblur: int = 20,
        fine_coeff: float = 6,
        coarse_coeff: float = -2,
        bias: int = 64,
        beta: float = 0.8,
        threshold: float = 80,
        mask_blend_factor=0.5,
        mask_blur_radius=5,
    ):
        # Lots of trial and error here, there are likely redundant
        # and inefficient parts.

        self.alpha = alpha
        self.innerblur = innerblur
        self.outerblur = outerblur
        self.fine_coeff = fine_coeff
        self.coarse_coeff = coarse_coeff
        self.bias = bias
        self.beta = beta
        self.threshold = threshold
        self.mask_blend_factor = mask_blend_factor
        self.mask_blur_radius = mask_blur_radius

        self.differ = Differencer(first_frame)

        self.prior_blend = None

        self.blended_blur = None
        self.blended_mask = None

    def _get_fine_blur(self, frame: Frame):
        return blur(frame, self.innerblur)

    def _get_coarse_blur(self, frame: Frame):
        return blur(frame, self.outerblur)

    def next(self, frame: Frame) -> Frame:
        this_diff = self.differ.next(frame)

        # TODO: Use generators that can be sent the shared frame and return
        # their respective next output.

        if self.prior_blend is None:
            this_blend = this_diff
        else:
            this_blend = blend(this_diff, self.prior_blend, self.alpha)

        self.prior_blend = this_blend

        fine_blur = self._get_fine_blur(this_blend)
        # Would it be more efficient to apply additional blur to fine_blur?
        coarse_blur = self._get_coarse_blur(this_blend)

        this_blur_delta = cv2.addWeighted(
            fine_blur, self.fine_coeff, coarse_blur, self.coarse_coeff, self.bias
        )

        # show_frame(this_blur_delta)

        # print(this_blur_delta)
        # biased = this_blur_delta + 128
        # clamped = np.clip(biased, 0, 255)
        # unsigned = clamped.astype(np.uint8)
        # show_frame(unsigned)
        # show_frame(np.where(this_blur_delta < 0, 128, this_blur_delta))
        # this_blur_delta = this_blur_delta.astype(np.uint8)

        blended_blur = self.blended_blur

        if blended_blur is None:
            blended_blur = this_blur_delta
        else:
            blended_blur = blend(blended_blur, this_blur_delta, self.beta)

        self.blended_blur = blended_blur

        # Create mask with full-scale values where the image should have a mask
        # (blocked out or diminished).
        # Maybe apply a sigmoid function instead of a binary threshold?
        _, mask = cv2.threshold(
            blended_blur, self.threshold, 255, cv2.THRESH_BINARY
        )  # cv2.THRESH_BINARY_INV)

        blended_mask = self.blended_mask

        if blended_mask is None:
            blended_mask = mask
        else:
            blended_mask = blend(mask, blended_mask, self.mask_blend_factor)

        self.blended_mask = blended_mask

        blurred_mask = blur(blended_mask, self.mask_blur_radius)
        # show_frame(blurred_mask)

        mask = blurred_mask

        # mask = cv2.cvtColor(this_blur_delta, cv2.COLOR_BGR2GRAY)
        # masked = mask_frame(mask, prior_frame)
        # show_frame(masked)

        return mask


class Overlayer:
    def __init__(
        self,
        first_frame: Frame,
        mask_func: Callable[[Frame], Frame],
        delay: int | None = None,
    ):
        if delay is None:
            delay = 6

        self.mask_func = mask_func

        # Keep delayed version of frames if needed to align with mask.
        # TODO: convert to gray after iteration so history can get the colored frame.
        history = FrameHistory(delay)
        self.history = history
        history.save(first_frame)

    def next(self, frame: Frame):
        self.history.save(frame)
        mask = self.mask_func(frame)

        # Reduce values outside focus areas by 1/2, with areas at full scale
        # in the mask staying unchanged.
        overlay = 0.5 + mask.astype(np.float16) * (0.5 / 255)

        frame_to_mask = self.history.get(None, True)
        assert frame_to_mask is not None
        # TODO: Use cv2.bitwise_and if the mask is binary
        overlaid = frame_to_mask.astype(np.float16) * (
            overlay if len(frame_to_mask.shape) == 2 else overlay[:, :, np.newaxis]
        )
        # FIXME: clip?
        return overlaid.astype(np.uint8)

    def threshold(self, frame: Frame, mask_threshold: int = 128):
        self.history.save(frame)
        mask = self.mask_func(frame)

        overlay = np.zeros_like(mask)
        overlay[mask < mask_threshold] = 255

        frame_to_mask = self.history.get(None, True)
        assert frame_to_mask is not None
        overlaid = cv2.addWeighted(frame_to_mask, 1, overlay, 0.5, 0)

        self.prior_frame = frame
        return overlaid


def present_masked_frames(
    source: FrameSource,
    delay: int | None = None,
    conversion=cv2.COLOR_BGR2GRAY,
    *args,
    **kwargs
):
    frames = iterframes(source)
    first_frame = next(frames)

    mask_generator = MaskGenerator(
        (
            first_frame
            if conversion is None
            else cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        ),
        *args,
        **kwargs
    )

    def mask_func(frame: Frame):
        return mask_generator.next(
            frame if conversion is None else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )

    overlayer = Overlayer(first_frame, mask_func, delay)

    for frame in frames:
        overlaid = overlayer.next(frame)
        show_frame(overlaid)


def get_background(source: FrameSource, alpha: float = 0.01):
    frames = iter(iterframes(source))

    # need to change type to avoid an assertion error
    accumulator = next(frames).astype(np.float32)

    for frame in frames:
        accumulator = cv2.accumulateWeighted(frame, accumulator, alpha)

    return cv2.convertScaleAbs(accumulator)


def background_accumulator(initial: Frame, alpha: float = 0.01, mask=None):
    blended = initial.astype(np.float32)
    yield blended  # Would get error if not priming with None

    # Use .close() on the generator to trigger GeneratorExit as the
    # proper wait to end it.
    while True:
        frame = yield cv2.convertScaleAbs(blended)
        blended = cv2.accumulateWeighted(frame, blended, alpha, mask)


def constant_accumulator(frame: Frame):
    """Placeholder "accumulator" that actually returns the initial frame each time."""
    while True:
        yield frame


def present_foregrounds(source: FrameSource):
    frames = iter(iterframes(source))
    first_frame = next(frames)
    bg_gen = background_accumulator(first_frame)
    # Avoid the error from sending a value to a just-started generator.
    next(bg_gen)
    foreground = first_frame
    # show_frame(foreground)

    for frame in frames:
        bg = bg_gen.send(frame)
        foreground = cv2.absdiff(frame, bg)
        show_frame(foreground)


def average_pixel_values(source: FrameSource):
    # NOTE: Assumes np.uint8 elements
    frames = iterframes(source)
    running_total = next(frames).astype(np.uint32)
    count = 1
    for frame in frames:
        running_total += frame
        count += 1

    return (running_total / count).astype(np.uint8)


def average_mask(source: FrameSource, *args, **kwargs) -> Frame:
    frames = iter_gray_frames(source)
    first_frame = next(frames)
    mask_generator = MaskGenerator(first_frame, *args, **kwargs)
    return average_pixel_values(mask_generator.next(frame) for frame in frames)
