import contextlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from src.data_processor import DataProcessor


class PeakDetectionTests(unittest.TestCase):
    def processor(self, **settings):
        processor = DataProcessor(np.arange(20_000), np.zeros(20_000))
        processor.config["detect_peaks"] = settings
        processor.dft_freq = np.fft.rfftfreq(20_000, d=1e-6)
        # A known noise floor with three peaks: two separated by 0.30 kHz.
        processor.dft_mag = np.linspace(0.8, 1.2, 10_001)
        processor.dft_mag[[2000, 2006, 4000]] = [10.0, 6.0, 3.0]
        return processor

    def test_reference_spacing_suppresses_nearby_weaker_peak(self):
        processor = self.processor(min_peak_distance_khz=0)
        self.assertEqual(len(processor.detect_peaks()), 3)
        processor.config["detect_peaks"].pop("min_peak_distance_khz")
        peaks = processor.detect_peaks()
        np.testing.assert_allclose([p["freq"] for p in peaks], [100, 200], atol=0.05)
        self.assertIs(processor.peaks, peaks)
        self.assertTrue(all(p["width_kHz"] > 0 for p in peaks))

    def test_sigma_settings_independently_reject_weaker_peaks(self):
        for key in ("height_sigma", "prominence_sigma"):
            with self.subTest(setting=key):
                processor = self.processor(min_peak_distance_khz=0)
                self.assertEqual(len(processor.detect_peaks()), 3)
                processor.config["detect_peaks"][key] = 40
                peaks = processor.detect_peaks()
                self.assertEqual(len(peaks), 1)
                self.assertAlmostEqual(peaks[0]["freq"], 100, delta=0.05)

    def test_invalid_sensitivity_settings_report_the_key(self):
        for key in ("min_peak_distance_khz", "height_sigma", "prominence_sigma"):
            for value in (-1, float("nan"), float("inf"), True, None, "5 kHz"):
                with self.subTest(setting=key, value=value):
                    processor = self.processor(**{key: value})
                    with self.assertRaisesRegex(ValueError, rf"detect_peaks\.{key}"):
                        processor.detect_peaks()

    def test_flat_signal_has_no_peaks(self):
        processor = DataProcessor(np.arange(100), np.zeros(100))
        self.assertEqual(processor.detect_peaks(), [])

    def test_default_config_loads_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.chdir(directory):
                processor = DataProcessor(np.arange(100), np.zeros(100))
        self.assertEqual(processor.config["detect_peaks"]["min_peak_distance_khz"], 0.5)

    def test_custom_yaml_controls_loaded_capture_detection(self):
        config = DataProcessor.load_yaml()
        config["detect_peaks"] = {"height_sigma": 1e9}
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "custom.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            capture_path = Path(directory) / "capture.csv"
            time = np.arange(4096)
            signal = np.sin(2 * np.pi * time / 16)
            signal += np.random.default_rng(42).normal(0, 0.1, time.size)
            np.savetxt(
                capture_path, np.column_stack([time * 1e-6, signal]),
                delimiter=",", header="time_s,voltage_v", comments="",
            )
            processor = DataProcessor.load_data(capture_path, config_path=config_path)
        self.assertEqual(processor.config["detect_peaks"]["height_sigma"], 1e9)
        self.assertEqual(processor.detect_peaks(), [])
        processor.config["detect_peaks"]["height_sigma"] = 5
        self.assertGreater(len(processor.detect_peaks()), 0)


if __name__ == "__main__":
    unittest.main()
