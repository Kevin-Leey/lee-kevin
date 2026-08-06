"""Retired latency-surface entry point.

The old surface combined incompatible evidence generations.  It remains as a
fail-closed command so archived automation receives an actionable diagnostic
instead of silently producing a paper-facing artifact.
"""

from __future__ import annotations

import argparse
from typing import Sequence


DEPRECATION_MESSAGE = (
    "analyze_rgd_latency_surface is retired; use the versioned release-state "
    "analysis and its matching provenance manifest instead"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-root")
    parser.add_argument("--output")
    parser.error(DEPRECATION_MESSAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
