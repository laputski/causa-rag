"""The smallest difference a question set can tell from noise.

A number reported to three decimal places on twelve questions carries an
authority the twelve questions cannot support, and nothing said so. The
platform calls a five per cent drop a regression; whether the set it is
measured on can distinguish five per cent at all is a separate question, and
until now an unasked one.

The estimate is the half-width of the ninety-five per cent interval around
the mean, and it is computed in closed form. The plan this came from said
bootstrap, and for a mean a bootstrap converges on exactly this number while
costing a thousand resamples per metric per read, which would move these
checks out of the class that costs a run nothing. The closed form is used
and the deviation is written down here, and not left for a reader to
notice.

What this cannot decide is which difference matters. That is a property of
the decision somebody is making and not of the data, so it is taken from the
thresholds the platform already applies when it calls a change a regression,
and never invented here.

Pure arithmetic: no I/O, no storage.
"""
from __future__ import annotations

import statistics

#: The interval this reports. Two-sided, ninety-five per cent, which is the
#: convention the reader of a metric will assume unless told otherwise.
_Z = 1.96


def resolution(values: list[float]) -> float | None:
    """Half the width of the interval around the mean of these values.

    A difference smaller than this cannot be told from the scatter of the
    questions themselves. None when fewer than two values are present: one
    question has no scatter to measure, and reporting zero would say the set
    can distinguish anything.
    """
    if len(values) < 2:
        return None
    return float(_Z * statistics.stdev(values) / len(values) ** 0.5)


def can_distinguish(values: list[float], difference: float) -> bool | None:
    """Whether a difference of this size would be visible on this set.

    None when the resolution cannot be estimated, which is a third answer and
    not a "no": a set of one question does not fail to distinguish the
    difference, it says nothing about it.
    """
    half_width = resolution(values)
    if half_width is None:
        return None
    return difference >= half_width
