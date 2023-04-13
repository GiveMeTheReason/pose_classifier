import fractions
import typing as tp


def frequency_controller(
    target_ratio: float,
    data_len: int,
) -> tp.Generator[int, None, None]:
    ratio_as_fraction = fractions.Fraction(target_ratio)
    current_frame = max(0, 1 - ratio_as_fraction)
    for frame in range(data_len):
        current_frame += ratio_as_fraction
        while current_frame >= 1:
            current_frame -= 1

            yield frame
