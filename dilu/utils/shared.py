"""Small value and output helpers shared by runtime modules."""

import sys
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a scalar to ``float`` and use ``default`` for missing values."""
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def float_or_default(value: Any, default: float) -> float:
    """Convert an optional numeric value while preserving an observed zero."""
    return safe_float(value, default)


def clip_unit_interval(value: float) -> float:
    """Clamp a scalar to the closed unit interval."""
    return float(min(1.0, max(0.0, float(value))))


def print_safe(*args: object, **kwargs: object) -> None:
    """Print plain text without Rich markup on Windows consoles."""
    text = " ".join(str(arg) for arg in args)
    for tag in (
        "[bold cyan]", "[/bold cyan]", "[bold green]", "[/bold green]",
        "[dim]", "[/dim]", "[yellow]", "[/yellow]", "[cyan]", "[/cyan]",
        "[green]", "[/green]", "[red]", "[/red]",
    ):
        text = text.replace(tag, "")
    print(text, file=kwargs.get("file", sys.stdout), flush=bool(kwargs.get("flush", False)))
