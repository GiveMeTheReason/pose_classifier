import fractions
import typing as tp


def frequency_controller(
    target_ratio: float,
    max_len: int,
) -> tp.Generator[int, None, None]:
    ratio = fractions.Fraction(target_ratio).limit_denominator()

    phase = max(fractions.Fraction(0), 1 - ratio)
    for period in range(max_len):
        phase += ratio
        while phase >= 1:
            phase -= 1
            yield period
