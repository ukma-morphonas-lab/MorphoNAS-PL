"""
Experiment tracking utilities for monitoring evolution progress.

This module provides trackers that can be composed with MorphoNAS experiments
to capture additional metrics during evolution without modifying core code.
"""

from .rollout_tracker import RolloutTracker

__all__ = [
    'RolloutTracker',
]
