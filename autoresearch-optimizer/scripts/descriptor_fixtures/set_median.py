"""Return the input string with the smallest total distance to all inputs."""

from median_string.metrics import sum_distance


def solve(instance) -> str:
    return min(instance.strings, key=lambda s: sum_distance(s, instance.strings, metric=instance.metric))
