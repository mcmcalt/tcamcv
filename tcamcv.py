"""Traffic camera computer vision experimentation"""

import cv2
from cv2 import VideoCapture
import numpy as np
from pathlib import Path
from typing import Iterable, Iterator, Optional

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
            yield from yield_frames(vc)

    @classmethod
    def rglob(cls, base: Path | str, pattern: str = "*"):
        return cls(Path(base).rglob(pattern))


def show_frame(frame, wait=1, title="Frame"):
    cv2.imshow(title if title else "Frame", frame)
    return cv2.waitKey(wait if wait is not None else 1)


def yield_vc_frames(vc: VideoCapture, ms: Optional[int] = None) -> Iterator[Frame]:
    if ms is not None:
        vc.set(cv2.CAP_PROP_POS_MSEC, ms)

    while True:
        status, frame = vc.read()
        if status:
            yield frame
        else:
            return


def yield_frames(source: FrameSource, start: Optional[int] = None) -> Iterator[Frame]:
    if isinstance(source, (str | Path)):
        source = cv2.VideoCapture(source)

    if isinstance(source, VideoCapture):
        return yield_vc_frames(source, start)
    else:
        return iter(source)


def yield_gray_frames(source: FrameSource, conversion=cv2.COLOR_BGR2GRAY):
    for frame in yield_frames(source):
        yield cv2.cvtColor(frame, conversion)


def yield_deltas(source: FrameSource, first_frame=None):
    frames = yield_frames(source)
    prior_frame = next(frames) if first_frame is None else first_frame

    for next_frame in frames:
        diff = cv2.absdiff(prior_frame, next_frame)
        yield diff
        prior_frame = next_frame


def present_diffs(source: FrameSource):
    for frame in yield_deltas(source):
        show_frame(frame, 0)


def blend(a, b, alpha: float):
    return cv2.addWeighted(a, alpha, b, 1 - alpha, 0)


def mask_frame(mask, frame):
    return cv2.bitwise_and(frame, frame, mask=mask)


# recommend alpha = .97 for pathways? .8 to .85 for tracking?
def yield_blends(source: FrameSource, alpha: float):
    iterator = yield_deltas(source)
    prior = next(iterator)

    for frame in iterator:
        blended = blend(prior, frame, alpha)
        yield blended
        prior = blended


def present_blends(source: FrameSource, alpha: float):
    for blended in yield_blends(source, alpha):
        show_frame(blended, 0)


# TODO: Use high-persistence frame blend to get pathways, then use
# that to shape the detection blur along lane and reduce adjacent lane influence?


def present_blurred_blends(source: FrameSource, alpha: float = 0.8, blursize: int = 51):
    # TODO: Give better error message indicating blursize must be odd?
    # Or have argument be radius, then double + 1?
    for blended in yield_blends(source, alpha):
        blurred = cv2.GaussianBlur(blended, (blursize, blursize), 0)
        show_frame(blurred, 0)


def blur_peaks(frame, blursize=2):
    diameter = blursize * 2 + 1
    blurred = cv2.GaussianBlur(frame, (diameter, diameter), 0)
    return frame - blurred


# If frames are (blended) frame-differences, positive values represent
# activity and negative values represent stable areas around activity.
def delta_blur(frame, blursize1=20, blursize2=80):
    diameter1 = blursize1 * 2 + 1
    diameter2 = blursize2 * 2 + 1
    blurred1 = cv2.GaussianBlur(frame, (diameter1, diameter1), 0)
    blurred2 = cv2.GaussianBlur(frame, (diameter2, diameter2), 0)
    return blurred2 - blurred1


def present_masked_frames(
    source: FrameSource,
    alpha: float = 0.8,
    innerblur: int = 20,
    outerblur: int = 80,
    beta: float = 0.8,
    threshold: float = 160,
):
    iterator = yield_gray_frames(source)
    first_frame = next(iterator)
    second_frame = next(iterator)
    first_diff = cv2.absdiff(first_frame, second_frame)

    prior_frame = second_frame
    prior_diff = first_diff
    prior_blend = first_diff

    blended_blur = None

    # FIXME: Keep delayed version of frames if needed to align with mask

    for this_frame in iterator:
        this_diff = cv2.absdiff(this_frame, prior_frame)
        # TODO: Use generators that can be sent the shared frame and return
        # their respective next output.
        this_blend = blend(this_diff, prior_blend, alpha)
        this_blur_delta = delta_blur(this_blend, innerblur, outerblur)
        # show_frame(this_blur_delta)

        if blended_blur is None:
            blended_blur = this_blur_delta
        else:
            blended_blur = blend(blended_blur, this_blur_delta, beta)

        # FIXME: Need to clamp properly to positive values. Otherwise
        # negative values will still act as transparent areas.
        # mask = np.clip(this_blur_delta, 0, 255)
        # mask = np.clip(blended_blur, 0, 255)

        # print(this_blur_delta)
        # show_frame(this_blur_delta)
        # show_frame(blended_blur)

        # mask = blended_blur > 0

        # Create mask with full-scale values where the image should have a mask
        # (blocked out or diminished).
        _, mask = cv2.threshold(blended_blur, threshold, 255, cv2.THRESH_BINARY_INV)

        overlay = np.zeros_like(mask)
        overlay[mask == 255] = 255

        # mask = cv2.cvtColor(this_blur_delta, cv2.COLOR_BGR2GRAY)
        # masked = mask_frame(mask, prior_frame)
        # show_frame(masked)

        overlaid = cv2.addWeighted(prior_frame, 1, overlay, 0.5, 0)
        show_frame(overlaid)

        prior_frame = this_frame
        prior_diff = this_diff
        prior_blend = this_blend


def get_background(source: FrameSource, alpha: float = 0.01):
    frames = iter(yield_frames(source))

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
    frames = iter(yield_frames(source))
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
