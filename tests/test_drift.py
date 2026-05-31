"""Tests for ADWIN concept drift detection."""

import numpy as np
import pytest

from src.drift.adwin_detector import ADWINDriftDetector


class TestADWINDriftDetector:
    """Tests for the ADWIN-based drift detector."""

    def test_initialization(self):
        """Detector should initialize with default signals."""
        detector = ADWINDriftDetector()

        assert len(detector.detectors) == 3  # default 3 signals
        assert detector.observation_count == 0
        assert detector.total_drifts_detected == 0

    def test_custom_signals(self):
        """Should initialize with custom signal names."""
        detector = ADWINDriftDetector(
            monitored_signals=["signal_a", "signal_b"]
        )
        assert len(detector.detectors) == 2
        assert "signal_a" in detector.detectors

    def test_update_no_drift(self):
        """Stable data should not trigger drift."""
        detector = ADWINDriftDetector(
            monitored_signals=["test"],
            delta=0.002,
        )

        # Feed stationary data
        for _ in range(100):
            drift_status = detector.update({"test": np.random.normal(0, 0.1)})
            # Most updates should not trigger drift
            # (We can't guarantee none will, so just check format)
            assert "test" in drift_status
            assert isinstance(drift_status["test"], bool)

    def test_detects_sudden_drift(self):
        """Should detect a sudden distribution change."""
        detector = ADWINDriftDetector(
            monitored_signals=["test"],
            delta=0.01,  # slightly more sensitive for testing
        )

        # Stationary phase
        for _ in range(500):
            detector.update({"test": np.random.normal(0, 0.1)})

        # Drift phase
        drift_detected = False
        for _ in range(500):
            status = detector.update({"test": np.random.normal(5, 0.1)})
            if status.get("test", False):
                drift_detected = True
                break

        assert drift_detected, "ADWIN should detect sudden drift"

    def test_drift_event_logging(self):
        """Drift events should be logged with metadata."""
        detector = ADWINDriftDetector(
            monitored_signals=["test"],
            delta=0.01,
        )

        # Force drift
        for _ in range(300):
            detector.update({"test": 0.0})
        for _ in range(300):
            detector.update({"test": 5.0})

        if detector.drift_events:
            event = detector.drift_events[0]
            assert "signal" in event
            assert "observation" in event
            assert "timestamp" in event
            assert event["signal"] == "test"

    def test_batch_update(self):
        """Batch updates should process all values."""
        detector = ADWINDriftDetector(
            monitored_signals=["sig1"],
        )

        batch = {"sig1": np.random.normal(0, 1, 100)}
        results = detector.update_batch(batch)

        assert len(results) == 100
        assert detector.observation_count == 100

    def test_get_drift_summary(self):
        """Summary should contain expected keys."""
        detector = ADWINDriftDetector(
            monitored_signals=["a", "b"],
        )

        for _ in range(50):
            detector.update({"a": 0.5, "b": 0.5})

        summary = detector.get_drift_summary()
        assert "total_observations" in summary
        assert "total_drifts" in summary
        assert "drifts_per_signal" in summary
        assert "current_window_sizes" in summary
        assert summary["total_observations"] == 50

    def test_reset_single_signal(self):
        """Should reset specific signal detector."""
        detector = ADWINDriftDetector(
            monitored_signals=["a", "b"],
        )

        for _ in range(100):
            detector.update({"a": 1.0, "b": 2.0})

        detector.reset("a")
        # After reset, detector for 'a' should have width of 0
        assert detector.detectors["a"].width == 0
        # Detector for 'b' should still have data
        assert detector.detectors["b"].width > 0

    def test_reset_all(self):
        """Should reset all detectors."""
        detector = ADWINDriftDetector(
            monitored_signals=["a", "b"],
        )

        for _ in range(100):
            detector.update({"a": 1.0, "b": 2.0})

        detector.reset()
        assert detector.detectors["a"].width == 0
        assert detector.detectors["b"].width == 0

    def test_simulate_drift(self):
        """Drift simulation should return detection results."""
        detector = ADWINDriftDetector(delta=0.01)

        pre = np.random.normal(0, 0.1, 300)
        post = np.random.normal(5, 0.1, 300)

        result = detector.simulate_drift(pre, post)

        assert "drift_point" in result
        assert "detected_at" in result
        assert "detected" in result
        assert result["drift_point"] == 300
        assert result["detected"] is True
        assert result["total_samples"] == 600

    def test_check_and_adapt(self):
        """check_and_adapt should call callback on drift."""
        detector = ADWINDriftDetector(
            monitored_signals=["test"],
            delta=0.01,
        )

        callback_called = {"value": False}

        def on_drift():
            callback_called["value"] = True

        # Stationary
        for _ in range(300):
            detector.check_and_adapt(
                {"test": 0.0},
                adaptation_callbacks={"test": on_drift},
            )

        # Drift
        for _ in range(300):
            detector.check_and_adapt(
                {"test": 5.0},
                adaptation_callbacks={"test": on_drift},
            )

        assert callback_called["value"] is True
