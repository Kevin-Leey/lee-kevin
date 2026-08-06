# Formal-protocol default thresholds and constant structures used across
# evaluation tools and the core metrics pipeline.

from typing import Dict

DEFAULT_RGD_SUBORDINATE_RUNTIME_PROFILE: Dict = {}

# Minimum fraction of frames with a gate snapshot record required before
# collapse audit numbers are deemed publication-grade.
DEFAULT_MAIN_TEXT_MIN_GATE_SNAPSHOT_FRAME_RATIO: float = 0.80
