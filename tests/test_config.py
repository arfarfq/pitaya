from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import yaml

from src.config import DEFAULTS, load_config
from main import analyze
from src.data_processor import DataProcessor
import main as workflow


class ConfigTests(unittest.TestCase):
    def test_example_covers_defaults_and_loads_custom_settings(self):
        example = Path(__file__).resolve().parents[1] / "config.yaml"
        supplied = yaml.safe_load(example.read_text())
        self.assertEqual(supplied.keys(), DEFAULTS.keys())
        for section, settings in DEFAULTS.items():
            self.assertEqual(supplied[section].keys(), settings.keys())
        config = load_config(example)
        self.assertEqual(config["harmonics"], supplied["harmonics"])
        self.assertEqual(config["fft"], supplied["fft"])

    def test_partial_config_merges_and_resolves_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.yaml"
            path.write_text("fft:\n  window: hann\n  padding_factor: 2\nplot:\n  save: plots/fft.png\n")
            config = load_config(path)
            self.assertEqual(config["fft"]["window"], "hann")
            self.assertEqual(config["fft"]["cutoff_khz"], 500)
            self.assertEqual(config["plot"]["save"], str((Path(directory) / "plots/fft.png").resolve()))

    def test_window_padding_and_detrending_change_transform(self):
        time = np.arange(1024) / 1e6
        volts = 3 + 0.1 * np.arange(1024) + 2 * np.cos(2 * np.pi * 125e3 * time)
        box = DataProcessor(time, volts, window="boxcar", detrend="none", padding_factor=1)
        freq, amplitude = box.dft()
        self.assertAlmostEqual(amplitude[0], volts.mean())
        clean = DataProcessor(time, volts, window="hann", window_symmetric=False,
                              detrend="linear", fft_length=4096, padding_factor=1)
        freq_clean, amplitude_clean = clean.dft()
        self.assertAlmostEqual(freq_clean[1] - freq_clean[0], (freq[1] - freq[0]) / 4)
        self.assertAlmostEqual(amplitude_clean[np.argmin(abs(freq_clean - 125))], 2, places=4)
        self.assertFalse(clean.y_analysis.flags.writeable)
        kaiser = DataProcessor(time, volts, window="kaiser", window_parameter=8.6)
        self.assertTrue(np.isfinite(kaiser.dft()[1]).all())

    def test_config_controls_calibration_and_detection(self):
        config = deepcopy(DEFAULTS)
        config["fft"].update(window="boxcar", padding_factor=1)
        config["calibration"].update(ref_khz=125, ref_mz=100, mz_range=None)
        config["harmonics"]["max_harmonic"] = 1
        time = np.arange(1024) / 1e6
        processor = DataProcessor(time, np.cos(2 * np.pi * 125e3 * time), **config["fft"])
        analyze(processor, config)
        peak = max(processor.peaks, key=lambda peak: peak["height"])
        self.assertAlmostEqual(processor.mz_axis[peak["index"]], 100)
        self.assertTrue(all(len(f["harmonics"]) == 1 for f in processor.harmonic_families))
        config["peaks"]["height_sigma"] = 1e20
        analyze(processor, config)
        self.assertEqual(processor.peaks, [])
        config["calibration"]["enabled"] = False
        analyze(processor, config)
        self.assertIsNone(processor.estimated_mz)

    def test_main_uses_config_input_plot_and_disabled_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            time = np.arange(128) / 1e6
            np.savetxt(base / "wave.csv", np.column_stack((time, np.sin(np.arange(128)))),
                       delimiter=",", header="time_s,voltage_v", comments="")
            path = base / "config.yaml"
            path.write_text("input:\n  file: wave.csv\ncalibration:\n  enabled: false\n"
                            "plot:\n  show: false\n  save: plots/test.png\n  show_harmonics: false\n")
            with (patch.object(DataProcessor, "frequency_to_mz") as mapping,
                  patch.object(workflow, "CONFIG_PATH", path)):
                workflow.main()
            mapping.assert_not_called()
            self.assertTrue((base / "plots/test.png").is_file())

    def test_harmonic_annotations_can_be_hidden(self):
        time = np.arange(2048) / 1e6
        processor = DataProcessor(time, np.cos(2 * np.pi * 62500 * time)
                                   + 0.5 * np.cos(2 * np.pi * 125000 * time))
        processor.frequency_to_mz()
        figure, axes = processor.plot(show=False, show_harmonics=False)
        self.assertFalse(any(text.get_text().startswith("H") for text in axes[-1].texts))
        import matplotlib.pyplot as plt
        plt.close(figure)

    def test_decay_config_controls_fit_and_saved_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.yaml"
            path.write_text("decay:\n  fit_start_s: 0.002\n  fit_end_s: 0.012\n"
                            "plot:\n  show_envelope: true\n  show: false\n  save: decay.png\n"
                            "calibration:\n  enabled: false\n")
            config = load_config(path)
            self.assertEqual(config["decay"]["expected_frequency_hz"], 100000)
            time = np.arange(20000) / 1e6
            voltage = np.exp(-time / 0.006) * np.cos(2 * np.pi * 100000 * time)
            processor = analyze(DataProcessor(time, voltage), config)
            self.assertAlmostEqual(processor.decay_fit["tau_s"], 0.006, delta=0.00002)
            self.assertGreaterEqual(processor.decay_fit["reference_time_s"], 0.002)
            figure, axes = workflow.display(processor, config)
            self.assertEqual(len(axes[0].lines), 2)
            self.assertEqual(len(axes), 2)
            self.assertTrue((Path(directory) / "decay.png").read_bytes().startswith(b"\x89PNG"))
