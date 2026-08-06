"""Evaluation components for the Recoverability-Gated Deliberation formal analysis pipeline.

This package exposes the retained experiment surface used to audit:

- Physical telemetry required by the formal protocol.
- Reasoning-route provenance and failure analysis records.
- Fixed-policy RGD aggregation and export for formal result bundles.
"""

from .physical_metrics import PhysicalMetrics, PhysicalMetricsRecorder
from .reasoning_recorder import ReasoningRecord, ReasoningRecorder

from .metrics_aggregator import MetricsAggregator

__all__ = [
    "PhysicalMetricsRecorder",
    "PhysicalMetrics",
    "ReasoningRecorder",
    "ReasoningRecord",
    "MetricsAggregator",
]
