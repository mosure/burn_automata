"""Small assertion and report access helpers for validation scripts."""

from __future__ import annotations

from typing import Any


def nested_get(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def expect_equal(failures: list[str], field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        failures.append(f"{field}: expected {expected!r}, got {actual!r}")


def expect_close(
    failures: list[str],
    field: str,
    actual: Any,
    expected: float,
    tolerance: float,
) -> None:
    if not isinstance(actual, int | float) or abs(float(actual) - expected) > tolerance:
        failures.append(
            f"{field}: expected {expected!r} +/- {tolerance}, got {actual!r}"
        )


def expect_true(failures: list[str], field: str, actual: Any) -> None:
    if actual is not True:
        failures.append(f"{field}: expected true, got {actual!r}")


def expect_false(failures: list[str], field: str, actual: Any) -> None:
    if actual is not False:
        failures.append(f"{field}: expected false, got {actual!r}")


def expect_bool(failures: list[str], field: str, actual: Any) -> None:
    if not isinstance(actual, bool):
        failures.append(f"{field}: expected boolean, got {actual!r}")


def expect_le(failures: list[str], field: str, actual: Any, maximum: Any) -> None:
    if (
        not isinstance(actual, int | float)
        or not isinstance(maximum, int | float)
        or float(actual) > float(maximum)
    ):
        failures.append(f"{field}: expected <= {maximum!r}, got {actual!r}")


def expect_ge(failures: list[str], field: str, actual: Any, minimum: Any) -> None:
    if (
        not isinstance(actual, int | float)
        or not isinstance(minimum, int | float)
        or float(actual) < float(minimum)
    ):
        failures.append(f"{field}: expected >= {minimum!r}, got {actual!r}")


def expect_finite(failures: list[str], field: str, actual: Any) -> None:
    if not isinstance(actual, int | float) or not float("-inf") < float(actual) < float("inf"):
        failures.append(f"{field}: expected finite number, got {actual!r}")
