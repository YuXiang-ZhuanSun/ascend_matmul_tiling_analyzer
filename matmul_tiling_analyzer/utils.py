from math import sqrt


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def ceil_align(value: int, align: int) -> int:
    return ceil_div(value, align) * align


def floor_align(value: int, align: int) -> int:
    return (value // align) * align


def min_sqrt_factor(value: int) -> int:
    root = int(sqrt(value))
    for factor in range(root, 0, -1):
        if value % factor == 0:
            return factor
    return 1
