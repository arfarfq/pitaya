import ast
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from scipy.signal.windows import blackmanharris

from src.config import DEFAULTS
from src.data_processor import DataProcessor


class DataProcessorTests(unittest.TestCase):
    def test_fft_matches_november_function_in_equivalent_units(self):
        # Extract only the reference function: importing the November module
        # would load its configured input and generate files and figures.
        path = Path(__file__).resolve().parents[1] / "src/FFT_decay_analysis_Nov_2026.py"
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "compute_fft")
        namespace = {"np": np, "fft": fft, "blackmanharris": blackmanharris}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)

        for n in (1023, 1024):
            with self.subTest(samples=n):
                time_s = np.arange(n) / 2e6
                volts = 0.1 + 0.02 * np.sin(2 * np.pi * 125e3 * time_s)
                processor = DataProcessor(time_s, volts)
                with contextlib.redirect_stdout(io.StringIO()):
                    khz, magnitude_v = namespace["compute_fft"](
                        volts - volts.mean(), processor.dt_seconds)
                frequency_khz, magnitude = processor.dft()
                np.testing.assert_allclose(frequency_khz, khz)
                np.testing.assert_allclose(magnitude, magnitude_v, atol=1e-12)
                self.assertAlmostEqual(frequency_khz[1] - frequency_khz[0], 2e3 / (8 * n))
                self.assertLessEqual(frequency_khz[-1], 500)
                self.assertIs(processor.dft()[1], magnitude)

    def test_sinusoid_amplitude_and_optional_full_fft(self):
        time_s = np.arange(1024) / 2e6
        y = 12 + 7 * np.cos(2 * np.pi * 125e3 * time_s)
        processor = DataProcessor(time_s, y, padding_factor=1, cutoff_khz=None)
        frequency, amplitude = processor.dft()
        self.assertEqual(frequency[-1], 1e3)
        self.assertEqual(frequency[np.argmax(amplitude)], 125)
        self.assertAlmostEqual(amplitude.max(), 7, places=5)
        np.testing.assert_allclose(processor.y, y)
        self.assertAlmostEqual(processor.y_ac.mean(), 0)

    def test_small_time_jitter_preserves_original_samples(self):
        timestamps = [4, 5.001, 5.999, 7]
        processor = DataProcessor(timestamps, [1, 3, 2, 4])
        np.testing.assert_array_equal(processor.x, timestamps)
        np.testing.assert_array_equal(processor.y, [1, 3, 2, 4])
        self.assertAlmostEqual(processor.dt_seconds, 1)
        self.assertFalse(processor.x.flags.writeable)
        self.assertFalse(processor.y_ac.flags.writeable)

    def test_fft_configuration_is_read_only_after_mapping(self):
        processor = self.harmonic_signal()
        processor.frequency_to_mz()
        frequencies, magnitude = processor.dft()
        for name, value in (("padding_factor", 2), ("cutoff_khz", 50), ("dt_seconds", 1)):
            with self.assertRaises(AttributeError):
                setattr(processor, name, value)
        self.assertIs(processor.dft()[0], frequencies)
        self.assertIs(processor.dft()[1], magnitude)

    def test_nyquist_amplitude_is_not_doubled(self):
        for padding in (1, 8):
            with self.subTest(padding=padding):
                processor = DataProcessor(np.arange(1024) / 2e6, (-1.) ** np.arange(1024),
                                          padding_factor=padding, cutoff_khz=None)
                frequency, amplitude = processor.dft()
                self.assertEqual(frequency[-1], 1000)
                self.assertAlmostEqual(amplitude[-1], 1)

    def test_odd_fft_last_bin_is_doubled(self):
        n = 1023
        samples = np.arange(n)
        voltage = np.cos(2 * np.pi * ((n - 1) // 2) * samples / n)
        processor = DataProcessor(samples / 2e6, voltage, padding_factor=1, cutoff_khz=None)
        _, actual = processor.dft()
        window = blackmanharris(n, sym=True)
        expected = 2 * abs(np.sum(processor.y_ac * window
                                 * np.exp(-2j * np.pi * ((n - 1) // 2) * samples / n))) / window.sum()
        self.assertAlmostEqual(actual[-1], expected)

    def test_plot_can_return_figure_without_display(self):
        processor = self.harmonic_signal()
        processor.frequency_to_mz()
        with patch.object(plt, "show") as show:
            figure, axes = processor.plot(show=False)
        show.assert_not_called()
        self.assertEqual(len(axes), 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spectrum.png"
            figure.savefig(path)
            self.assertGreater(path.stat().st_size, 0)
        plt.close(figure)

    def test_mapping_calibration_range_and_plot_amplitudes(self):
        processor = DataProcessor(np.arange(1024) * 1e-6, np.sin(np.arange(1024)))
        mz = processor.frequency_to_mz()
        frequencies = processor.dft_freq[processor.mz_mask]
        reference = DEFAULTS["calibration"]
        np.testing.assert_allclose(mz, reference["ref_mz"] * (reference["ref_khz"] / frequencies) ** 2)
        self.assertTrue(np.all((mz >= 1) & (mz <= 200)))
        with patch.object(plt, "show"):
            figure, axes = processor.plot(frequency_limits_khz=(10, 250))
        self.assertEqual(axes[0].get_xlabel(), "Time (s)")
        self.assertEqual(axes[0].get_ylabel(), "Voltage (V)")
        self.assertEqual(axes[1].get_xlabel(), "Frequency (kHz)")
        self.assertEqual(axes[1].get_ylabel(), "Peak amplitude (V)")
        self.assertEqual(axes[1].get_xlim(), (10, 250))
        order = np.argsort(mz)
        np.testing.assert_allclose(axes[-1].lines[0].get_xdata(), mz[order])
        np.testing.assert_allclose(axes[-1].lines[0].get_ydata(),
                                   processor.cleaned_mag[processor.mz_mask][order])
        plt.close(figure)

    def harmonic_signal(self):
        time_s = np.arange(8192) / 1e6
        rng = np.random.default_rng(42)
        voltage = rng.normal(0, 0.01, time_s.size)
        for frequency, amplitude in ((62.5e3, 1), (125e3, 0.4), (187.5e3, 0.2),
                                     (93.75e3, 0.6)):
            voltage += amplitude * np.sin(2 * np.pi * frequency * time_s)
        return DataProcessor(time_s, voltage)

    def test_peak_measurements_and_harmonic_assignment(self):
        processor = self.harmonic_signal()
        peaks = processor.detect_peaks()
        family = min(processor.detect_harmonics(), key=lambda f: abs(f["freq_khz"] - 62.5))
        self.assertAlmostEqual(family["freq_khz"], 62.5)
        self.assertEqual([member["order"] for member in family["harmonics"]], [1, 2, 3])
        np.testing.assert_allclose([member["freq_khz"] for member in family["harmonics"]],
                                   [62.5, 125, 187.5])
        self.assertGreater(family["total_power"], family["fundamental_power"])
        assigned = [member["index"] for f in processor.harmonic_families for member in f["harmonics"]]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(len(assigned), len(peaks))
        fundamental = family["harmonics"][0]
        self.assertAlmostEqual(fundamental["height"], 1, delta=0.002)
        self.assertGreater(fundamental["width_khz"], 0)
        self.assertGreater(fundamental["power"], 0)
        self.assertGreater(fundamental["area"], 0)

    def test_cleaning_preserves_peaks_and_flattens_background(self):
        processor = self.harmonic_signal()
        _, raw = processor.dft()
        original = raw.copy()
        processor.frequency_to_mz()
        cleaned = processor.cleaned_mag
        self.assertTrue(processor.peak_mask.any())
        self.assertTrue((~processor.peak_mask).any())
        np.testing.assert_array_equal(cleaned[~processor.peak_mask], 0)
        np.testing.assert_array_equal(cleaned[processor.peak_mask], raw[processor.peak_mask])
        np.testing.assert_array_equal(raw, original)
        processor.detect_peaks(min_distance_khz=1)
        self.assertIsNone(processor.harmonic_families)
        self.assertIsNone(processor.cleaned_mag)
        with patch.object(plt, "show"):
            figure, axes = processor.plot(clean_mz=False)
        order = np.argsort(processor.estimated_mz)
        np.testing.assert_array_equal(axes[-1].lines[0].get_ydata(), raw[processor.mz_mask][order])
        plt.close(figure)

    def test_no_peaks_and_tiny_spectra_are_safe(self):
        for cutoff in (500, 0.001):
            processor = DataProcessor(np.arange(32) * 1e-6, np.zeros(32), cutoff_khz=cutoff)
            self.assertEqual(processor.detect_peaks(), [])
            self.assertEqual(processor.detect_harmonics(), [])
            np.testing.assert_array_equal(processor.clean_spectrum(), 0)
            processor.frequency_to_mz()
            with patch.object(plt, "show"):
                figure, _ = processor.plot()
            plt.close(figure)

    def test_custom_reference_offset_and_harmonic_orders(self):
        processor = DataProcessor(np.arange(1024) * 1e-6, np.zeros(1024))
        mz = processor.frequency_to_mz(ref_khz=200, ref_mz=40,
                                       ref_harmonic_order=2, harmonic_order=2,
                                       offset_khz=1, mz_range=None)
        frequency = processor.dft_freq[processor.mz_mask]
        np.testing.assert_allclose(mz, 40 * (99 / (frequency / 2 - 1)) ** 2)
        # The reference need not land on a bin; check the inverse mapping too.
        np.testing.assert_allclose(frequency, 2 * (1 + 99 * np.sqrt(40 / mz)))

    def test_legacy_loader_skips_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scope.csv"
            path.write_text("Scope export\nA,B,C,time,voltage\n"
                            "0,0,0,0,0.1\n0,0,0,0.000001,0.2\n")
            processor = DataProcessor.load_data(path, padding_factor=2)
        np.testing.assert_allclose(processor.x, [0, 1e-6])
        np.testing.assert_allclose(processor.y, [0.1, 0.2])
        self.assertEqual(processor.padding_factor, 2)

    def test_custom_cutoff_is_in_khz(self):
        processor = DataProcessor(np.arange(1024) * 1e-6, np.zeros(1024),
                                  cutoff_khz=125)
        frequency, _ = processor.dft()
        self.assertAlmostEqual(frequency[-1], 125)





if __name__ == "__main__":
    unittest.main()
