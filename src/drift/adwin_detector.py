"""
ADWIN Concept Drift Detection Module
======================================
Monitors multiple data streams for distributional changes using the
ADWIN (Adaptive Windowing) algorithm from the River library.
When drift is detected, triggers model adaptation procedures.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ADWINDriftDetector:
    """
    Concept drift detector using ADWIN from the River library.

    Monitors multiple signals (prediction error, risk scores, feature
    distributions) and triggers adaptation when statistically significant
    distributional changes are detected.

    Parameters
    ----------
    delta : float
        ADWIN confidence parameter. Lower values = fewer false alarms
        but slower detection. Default 0.002.
    monitored_signals : list of str
        Names of signals to monitor.
    drift_check_interval : int
        Number of observations between drift checks.
    """

    def __init__(
        self,
        delta: float = 0.002,
        monitored_signals: Optional[List[str]] = None,
        drift_check_interval: int = 1000,
    ):
        self.delta = delta
        self.monitored_signals = monitored_signals or [
            "prediction_error",
            "risk_score",
            "transaction_amount",
        ]
        self.drift_check_interval = drift_check_interval

        # Initialize ADWIN detectors for each signal
        self.detectors: Dict = {}
        self._initialize_detectors()

        # Drift event log
        self.drift_events: List[Dict] = []
        self.observation_count = 0
        self.total_drifts_detected = 0

    def _initialize_detectors(self) -> None:
        """Initialize ADWIN detector for each monitored signal."""
        from river import drift

        for signal_name in self.monitored_signals:
            self.detectors[signal_name] = drift.ADWIN(delta=self.delta)

        logger.info(f"  Initialized ADWIN detectors for {len(self.detectors)} signals")
        logger.info(f"  Delta: {self.delta}, Check interval: {self.drift_check_interval}")

    def update(self, signal_values: Dict[str, float]) -> Dict[str, bool]:
        """
        Update drift detectors with new observations.

        Parameters
        ----------
        signal_values : dict
            Dictionary mapping signal names to their current values.
            e.g., {"prediction_error": 0.15, "risk_score": 0.42}

        Returns
        -------
        dict
            Dictionary mapping signal names to drift status (True = drift detected).
        """
        self.observation_count += 1
        drift_status = {}

        for signal_name, detector in self.detectors.items():
            if signal_name in signal_values:
                value = signal_values[signal_name]

                # Update the ADWIN detector
                detector.update(value)

                # Check for drift
                if detector.drift_detected:
                    drift_status[signal_name] = True
                    self.total_drifts_detected += 1

                    # Log the drift event
                    event = {
                        "signal": signal_name,
                        "observation": self.observation_count,
                        "value": value,
                        "timestamp": datetime.now().isoformat(),
                        "window_size": detector.width,
                    }
                    self.drift_events.append(event)

                    logger.warning(
                        f"  [!] DRIFT DETECTED in '{signal_name}' at observation "
                        f"{self.observation_count} (value={value:.4f}, "
                        f"window={detector.width})"
                    )
                else:
                    drift_status[signal_name] = False

        return drift_status

    def update_batch(
        self, signal_batch: Dict[str, np.ndarray]
    ) -> List[Dict[str, bool]]:
        """
        Update detectors with a batch of observations.

        Parameters
        ----------
        signal_batch : dict
            Dictionary mapping signal names to arrays of values.

        Returns
        -------
        list of dict
            Drift status for each observation in the batch.
        """
        batch_size = max(len(v) for v in signal_batch.values())
        all_drift_status = []

        for i in range(batch_size):
            values = {}
            for signal_name, signal_array in signal_batch.items():
                if i < len(signal_array):
                    values[signal_name] = float(signal_array[i])
            drift_status = self.update(values)
            all_drift_status.append(drift_status)

        return all_drift_status

    def check_and_adapt(
        self,
        signal_values: Dict[str, float],
        adaptation_callbacks: Optional[Dict[str, Callable]] = None,
    ) -> bool:
        """
        Check for drift and trigger adaptation if detected.

        Parameters
        ----------
        signal_values : dict
            Current signal values.
        adaptation_callbacks : dict, optional
            Functions to call when drift is detected for each signal.
            Keys are signal names, values are callables.

        Returns
        -------
        bool
            True if any drift was detected.
        """
        drift_status = self.update(signal_values)
        any_drift = any(drift_status.values())

        if any_drift and adaptation_callbacks:
            for signal_name, is_drifted in drift_status.items():
                if is_drifted and signal_name in adaptation_callbacks:
                    logger.info(f"  Triggering adaptation for '{signal_name}'...")
                    adaptation_callbacks[signal_name]()

        return any_drift

    def get_drift_summary(self) -> Dict:
        """
        Get a summary of all detected drift events.

        Returns
        -------
        dict
            Summary including total drifts, per-signal counts, and event log.
        """
        from collections import Counter

        signal_counts = Counter(event["signal"] for event in self.drift_events)

        summary = {
            "total_observations": self.observation_count,
            "total_drifts": self.total_drifts_detected,
            "drifts_per_signal": dict(signal_counts),
            "drift_events": self.drift_events,
            "current_window_sizes": {
                name: det.width for name, det in self.detectors.items()
            },
        }

        return summary

    def reset(self, signal_name: Optional[str] = None) -> None:
        """
        Reset drift detector(s).

        Parameters
        ----------
        signal_name : str, optional
            If provided, reset only this signal's detector.
            If None, reset all detectors.
        """
        from river import drift

        if signal_name:
            if signal_name in self.detectors:
                self.detectors[signal_name] = drift.ADWIN(delta=self.delta)
                logger.info(f"  Reset drift detector for '{signal_name}'")
        else:
            self._initialize_detectors()
            logger.info("  Reset all drift detectors")

    def simulate_drift(
        self,
        pre_drift_data: np.ndarray,
        post_drift_data: np.ndarray,
        signal_name: str = "test_signal",
    ) -> Dict:
        """
        Simulate drift detection for testing/validation.

        Parameters
        ----------
        pre_drift_data : np.ndarray
            Data before the drift point.
        post_drift_data : np.ndarray
            Data after the drift point.
        signal_name : str
            Name for the test signal.

        Returns
        -------
        dict
            Simulation results including detection point and delay.
        """
        from river import drift

        logger.info(f"  Simulating drift detection for '{signal_name}'...")
        logger.info(f"    Pre-drift: {len(pre_drift_data)} samples, "
                     f"mean={pre_drift_data.mean():.4f}")
        logger.info(f"    Post-drift: {len(post_drift_data)} samples, "
                     f"mean={post_drift_data.mean():.4f}")

        test_detector = drift.ADWIN(delta=self.delta)
        combined_data = np.concatenate([pre_drift_data, post_drift_data])
        drift_point = len(pre_drift_data)

        detected_at = None
        for i, value in enumerate(combined_data):
            test_detector.update(value)
            if test_detector.drift_detected and detected_at is None:
                detected_at = i

        result = {
            "drift_point": drift_point,
            "detected_at": detected_at,
            "detection_delay": detected_at - drift_point if detected_at else None,
            "detected": detected_at is not None,
            "total_samples": len(combined_data),
        }

        if detected_at:
            logger.info(f"    [OK] Drift detected at index {detected_at} "
                       f"(delay: {result['detection_delay']} samples)")
        else:
            logger.warning(f"    [X] Drift not detected")

        return result
