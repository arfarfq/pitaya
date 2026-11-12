"""Analyze oscilloscope transients with NumPy, pandas, and Matplotlib."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_WAVEFORM = Path("data") / "k_20ms_avg128_26.5degC.csv"


@dataclass(frozen=True)
class TransientAnalysis:
    """A baseline-corrected segment and its one-sided amplitude spectrum."""

    time_s: np.ndarray
    voltage_v: np.ndarray
    spectrum: pd.DataFrame

    def plot_data(self):
        """Plot this segment and its spectrum without recomputing the FFT."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
        axes[0].plot(self.time_s * 1e6, self.voltage_v * 1e3, linewidth=0.7)
        axes[0].set(title="Prepared detector transient", xlabel="Time (µs)",
                    ylabel="Voltage (mV)")
        axes[1].plot(self.spectrum.frequency_hz / 1e3,
                     self.spectrum.amplitude_v * 1e3, linewidth=0.7)
        window = self.spectrum.attrs["window"]
        axes[1].set(title=f"{window} amplitude spectrum", xlabel="Frequency (kHz)",
                    ylabel="Peak amplitude (mV)")
        for axis in axes:
            axis.grid(True, alpha=0.3)
        return fig, axes


class DataProcessor:
    """Process a uniformly sampled detector transient (seconds, volts).

    CSV defaults match the oscilloscope export used in test.ipynb: no header,
    time in column 3, voltage in column 4 (zero-based). Spectrum-only CSVs
    cannot be loaded as time traces.
    """

    def __init__(self, x, y):
        self.x = np.array(x, dtype=float, copy=True)
        self.y = np.array(y, dtype=float, copy=True)

        if (self.x.ndim != 1 or self.y.ndim != 1
                or self.x.size != self.y.size or self.x.size < 4):
            raise ValueError("Provide matching 1-D time/voltage arrays with >= 4 samples.")

        if not (np.isfinite(self.x).all() and np.isfinite(self.y).all()):
            raise ValueError("Time and voltage must contain only finite numbers.")

        if np.any(np.diff(self.x) <= 0):
            raise ValueError("Time must be strictly increasing.")

        self.sample_interval_s = (self.x[-1] - self.x[0]) / (self.x.size - 1)
        grid = self.x[0] + np.arange(self.x.size) * self.sample_interval_s

        # Allow small oscilloscope timestamp rounding, not irregular sampling.
        if np.max(np.abs(self.x - grid)) > 0.02 * self.sample_interval_s:
            raise ValueError("FFT requires uniform sampling (within 0.02 sample of a grid).")

        self.sample_rate_hz = 1 / self.sample_interval_s

        self.x.setflags(write=False)
        self.y.setflags(write=False)

        self.calibration = None

    @classmethod
    def load_data(cls, file_path, *, time_column=3, voltage_column=4,
                  header=None, time_scale=1.0, voltage_scale=1.0):

        """Load a waveform CSV; scale source units to seconds and volts."""
        if (not np.isfinite([time_scale, voltage_scale]).all()
                or time_scale <= 0 or voltage_scale <= 0):
            raise ValueError("Unit scale factors must be finite and positive.")

        data = pd.read_csv(file_path, header=header)

        if not (0 <= time_column < data.shape[1]
                and 0 <= voltage_column < data.shape[1]
                and time_column != voltage_column):
            raise ValueError("Select distinct time and voltage column indices present in the CSV.")

        x = pd.to_numeric(data.iloc[:, time_column], errors="raise").to_numpy(float)
        y = pd.to_numeric(data.iloc[:, voltage_column], errors="raise").to_numpy(float)
        return cls(x * time_scale, y * voltage_scale)

    def prepare(self, *, start_s=None, stop_s=None, baseline_interval_s=None):
        """Return a new baseline-corrected segment; intervals are [start, stop).

        Set start_s after the measured switching/recovery interval. If no
        baseline interval is given, subtract the selected segment's mean.
        """
        start = self.x[0] if start_s is None else float(start_s)
        stop = self.x[-1] + self.sample_interval_s if stop_s is None else float(stop_s)

        if not np.isfinite([start, stop]).all() or start >= stop:
            raise ValueError("Analysis interval must have finite start < stop.")


        mask = (self.x >= start) & (self.x < stop)
        if mask.sum() < 4:
            raise ValueError("Analysis interval must contain at least four samples.")
        if baseline_interval_s is None:
            baseline_mask = mask
        else:
            left, right = baseline_interval_s
            if not np.isfinite([left, right]).all() or left >= right:
                raise ValueError("Baseline interval must have finite start < stop.")
            baseline_mask = (self.x >= left) & (self.x < right)
            if not baseline_mask.any():
                raise ValueError("Baseline interval contains no samples.")

        baseline = float(self.y[baseline_mask].mean())
        return self.x[mask].copy(), self.y[mask] - baseline, baseline

    @staticmethod
    def _window(size, name):
        if name == "rectangular":
            return np.ones(size)
        if name != "blackmanharris":
            raise ValueError("Window must be 'blackmanharris' or 'rectangular'.")
        # Periodic four-term Blackman–Harris, equivalent to sym=False.
        phase = 2 * np.pi * np.arange(size) / size
        return (0.35875 - 0.48829 * np.cos(phase)
                + 0.14128 * np.cos(2 * phase) - 0.01168 * np.cos(3 * phase))

    def dft(self, *, window="blackmanharris", **selection):
        """Return complex, unnormalized real FFT of the prepared segment."""
        _, voltage, _ = self.prepare(**selection)
        return np.fft.rfft(voltage * self._window(voltage.size, window))

    def spectrum(self, *, window="blackmanharris", **selection):
        """One-sided peak amplitudes in V, normalized by sum(window).

        This recovers the amplitude of a bin-centered steady tone. Decaying
        or overlapping components do not directly give the initial amplitude.
        """
        return self.analyze(window=window, **selection).spectrum

    def analyze(self, *, window="blackmanharris", **selection):
        """Prepare a segment and compute its spectrum in a single pass."""
        time, voltage, baseline = self.prepare(**selection)
        weights = self._window(voltage.size, window)
        transform = np.fft.rfft(voltage * weights)
        amplitude = 2 * np.abs(transform) / weights.sum()
        amplitude[0] /= 2
        if voltage.size % 2 == 0:
            amplitude[-1] /= 2
        result = pd.DataFrame({
            "frequency_hz": np.fft.rfftfreq(voltage.size, d=self.sample_interval_s),
            "amplitude_v": amplitude,
        })
        result.attrs.update(window=window, baseline_v=baseline,
                            bin_spacing_hz=self.sample_rate_hz / voltage.size,
                            duration_s=voltage.size * self.sample_interval_s,
                            start_s=float(time[0]), sample_count=voltage.size)
        return TransientAnalysis(time_s=time, voltage_v=voltage, spectrum=result)

    def find_peaks(self, *, min_amplitude_v, min_frequency_hz=0.0,
                   max_frequency_hz=None, min_separation_hz=0.0, **spectrum_options):
        """Find thresholded local maxima, strongest first.

        Choose the amplitude threshold from a matched background. SNR is
        amplitude / median spectrum amplitude within the search band, an
        exploratory floor estimate rather than statistical significance.
        Width is the discrete full width at half amplitude, without a
        background correction. Endpoints (DC/Nyquist) are not peak candidates.
        """
        upper = self.sample_rate_hz / 2 if max_frequency_hz is None else max_frequency_hz
        if (not np.isfinite([min_amplitude_v, min_frequency_hz, upper,
                             min_separation_hz]).all()
                or min_amplitude_v < 0 or min_frequency_hz < 0
                or upper <= min_frequency_hz or upper > self.sample_rate_hz / 2
                or min_separation_hz < 0):
            raise ValueError("Use nonnegative thresholds/separation and a band within Nyquist.")
        spectrum = self.spectrum(**spectrum_options)
        f = spectrum.frequency_hz.to_numpy()
        a = spectrum.amplitude_v.to_numpy()
        band = (f >= min_frequency_hz) & (f <= upper)
        if not band.any():
            raise ValueError("Search band contains no FFT bins.")
        floor = float(np.median(a[band]))
        candidates = np.flatnonzero((a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:])) + 1
        candidates = [i for i in candidates if band[i] and a[i] >= min_amplitude_v]
        selected = []
        rows = []
        for i in sorted(candidates, key=lambda j: a[j], reverse=True):
            if any(abs(f[i] - f[j]) < min_separation_hz for j in selected):
                continue
            selected.append(i)
            left = right = i
            while left > 0 and a[left] > a[i] / 2:
                left -= 1
            while right < a.size - 1 and a[right] > a[i] / 2:
                right += 1
            flags = "review pickup / harmonics / overlap"
            if left == 0 or right == a.size - 1:
                flags += "; width reaches spectrum boundary"
            rows.append((f[i], a[i], f[right] - f[left],
                         a[i] / floor if floor > 0 else np.nan, flags))
        return pd.DataFrame(rows, columns=["frequency_hz", "amplitude_v",
                                          "width_hz", "snr_amplitude", "flags"])

    def fit_decay(self, frequency_hz, *, block_cycles=20, min_amplitude_v=0.0,
                  **selection):
        """Demodulate an isolated tone in nonoverlapping blocks; fit log amplitude.

        The approximate bandwidth is fs / block_samples. Choose blocks long
        enough to reject nearby components but short compared with tau. Select
        a recovered, above-noise interval; no automatic noise/edge rejection is
        inferred. The returned tau uncertainty is an OLS standard error only.
        """
        if (not np.isfinite([frequency_hz, block_cycles, min_amplitude_v]).all()
                or not 0 < frequency_hz < self.sample_rate_hz / 2
                or block_cycles < 1 or min_amplitude_v < 0):
            raise ValueError("Use a frequency within Nyquist, >= 1 cycle, and a nonnegative floor.")
        time, voltage, baseline = self.prepare(**selection)
        block_size = max(4, int(round(block_cycles * self.sample_rate_hz / frequency_hz)))
        count = voltage.size // block_size
        if count < 4:
            raise ValueError("Select at least four complete demodulation blocks.")
        time_blocks = time[:count * block_size].reshape(count, block_size)
        signal_blocks = voltage[:count * block_size].reshape(count, block_size)
        relative_time = time_blocks - time[0]
        amplitude = 2 * np.abs(np.mean(
            signal_blocks * np.exp(-2j * np.pi * frequency_hz * relative_time), axis=1))
        centers = time_blocks.mean(axis=1)
        keep = amplitude > min_amplitude_v
        if keep.sum() < 4:
            raise ValueError("Fewer than four blocks remain above the amplitude floor.")
        elapsed = centers[keep] - time[0]
        coefficients, covariance = np.polyfit(elapsed, np.log(amplitude[keep]), 1, cov=True)
        slope, intercept = coefficients
        if slope >= 0:
            raise ValueError("Selected envelope does not have a negative decay slope.")
        residuals = np.log(amplitude[keep]) - (intercept + slope * elapsed)
        envelope = pd.DataFrame({"time_s": centers, "amplitude_v": amplitude, "used": keep})
        envelope["fit_amplitude_v"] = np.exp(intercept + slope * (centers - time[0]))
        return {"tau_s": float(-1 / slope), "decay_rate_per_s": float(-slope),
                "tau_standard_error_s": float(np.sqrt(covariance[0, 0]) / slope**2),
                "fit_start_s": float(centers[keep][0]),
                "fit_stop_s": float(centers[keep][-1]), "baseline_v": baseline,
                "block_samples": block_size, "envelope": envelope,
                "log_residuals": residuals}

    def calibrate_mass(self, frequencies_hz, mass_to_charge, *, calibration_id):
        """Fit m/z = C/f² to confirmed references; caller verifies settings/assignments.

        C minimizes unweighted squared m/z residuals. A single reference
        initializes C but cannot establish a validated range or uncertainty.
        """
        f = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
        mz = np.atleast_1d(np.asarray(mass_to_charge, dtype=float))
        if (f.ndim != 1 or mz.shape != f.shape or not f.size
                or not np.isfinite(f).all() or not np.isfinite(mz).all()
                or np.any(f <= 0) or np.any(mz <= 0) or not calibration_id):
            raise ValueError("Provide matching positive finite references and a calibration ID.")
        inverse_square = 1 / f**2
        coefficient = float(np.dot(inverse_square, mz) / np.dot(inverse_square, inverse_square))
        references = pd.DataFrame({"frequency_hz": f, "reference_mz": mz,
                                   "predicted_mz": coefficient / f**2})
        references["residual_mz"] = references.predicted_mz - mz
        self.calibration = {"id": str(calibration_id), "coefficient": coefficient,
                            "frequency_range_hz": (float(f.min()), float(f.max())),
                            "references": references}
        return references.copy()

    def mass_spectrum(self, *, spectrum=None, **spectrum_options):
        """Map positive frequencies to m/z and flag bins within the reference range.

        Pass a spectrum from this processor to reuse its FFT, or pass the
        selection and window options to compute a spectrum here.
        """
        if self.calibration is None:
            raise ValueError("Calibrate with confirmed reference ions first.")
        if spectrum is None:
            spectrum = self.spectrum(**spectrum_options)
        elif spectrum_options:
            raise ValueError("Use either a computed spectrum or spectrum options.")
        result = spectrum.loc[spectrum.frequency_hz > 0].copy()
        result["mass_to_charge"] = self.calibration["coefficient"] / result.frequency_hz**2
        low, high = self.calibration["frequency_range_hz"]
        result["within_reference_range"] = result.frequency_hz.between(low, high)
        result.attrs["calibration_id"] = self.calibration["id"]
        return result.sort_values("mass_to_charge").reset_index(drop=True)

    def plot_data(self, *, window="blackmanharris", **selection):
        """Plot the prepared trace and spectrum; return figure/axes for customization."""
        return self.analyze(window=window, **selection).plot_data()

    def plot_mass_spectrum(self, **spectrum_options):
        """Plot the mass spectrum and mark the calibration reference or range."""
        spectrum = self.mass_spectrum(**spectrum_options)
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        ax.plot(spectrum.mass_to_charge, spectrum.amplitude_v * 1e3, linewidth=0.7)
        ax.set(title=f"Mass spectrum (calibration ID: {self.calibration['id']})",
               xlabel="Mass-to-charge ratio (m/z)", ylabel="Peak amplitude (mV)")
        low, high = self.calibration["frequency_range_hz"]
        coefficient = self.calibration["coefficient"]
        if low == high:
            ax.axvline(coefficient / low**2, color="orange", label="Reference m/z")
        else:
            ax.axvspan(coefficient / high**2, coefficient / low**2,
                       color="orange", alpha=0.2, label="Reference range")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig, ax


def find_waveform(path=None):
    """Resolve an explicit CSV path, or find the example CSV above the cwd."""
    if path is not None:
        waveform = Path(path).expanduser().resolve()
        if waveform.is_file():
            return waveform
        raise FileNotFoundError(f"Waveform CSV not found at {waveform}")

    current = Path.cwd()
    for directory in (current, *current.parents):
        waveform = directory / DEFAULT_WAVEFORM
        if waveform.is_file():
            return waveform
    raise FileNotFoundError(
        f"Could not find {DEFAULT_WAVEFORM} from {current} or its parent directories. "
        "Pass the CSV path explicitly."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze a detector waveform CSV.")
    parser.add_argument("csv_path", nargs="?", help=f"CSV path (default: {DEFAULT_WAVEFORM})")
    parser.add_argument("--start-s", type=float, default=0.0,
                        help="Analysis start in seconds (default: 0)")
    parser.add_argument("--stop-s", type=float, help="Exclusive analysis stop in seconds")
    parser.add_argument("--reference", action="append", nargs=2, type=float,
                        metavar=("FREQUENCY_HZ", "MZ"),
                        help="Confirmed calibration pair. Repeat for more references.")
    parser.add_argument("--calibration-id", help="Identifier for the calibration references")
    args = parser.parse_args(argv)
    if bool(args.reference) != bool(args.calibration_id):
        parser.error("Use --reference and --calibration-id together.")

    try:
        processor = DataProcessor.load_data(find_waveform(args.csv_path))
        baseline_interval = (processor.x[0], 0.0) if processor.x[0] < 0 else None
        analysis = processor.analyze(start_s=args.start_s, stop_s=args.stop_s,
                                     baseline_interval_s=baseline_interval)
        if args.reference:
            frequencies, masses = zip(*args.reference)
            processor.calibrate_mass(frequencies, masses, calibration_id=args.calibration_id)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    spectrum = analysis.spectrum
    stats = spectrum.attrs
    baseline_source = "pre-trigger samples" if baseline_interval is not None else "selected samples"
    print(f"Samples used: {stats['sample_count']:,} of {processor.y.size:,} | "
          f"Sample rate: {processor.sample_rate_hz / 1e6:.3f} MS/s")
    print(f"FFT duration: {stats['duration_s'] * 1e3:.3f} ms | "
          f"Bin spacing: {stats['bin_spacing_hz']:.2f} Hz")
    print(f"Baseline: {stats['baseline_v'] * 1e3:.3f} mV ({baseline_source})")

    analysis.plot_data()
    if args.reference:
        processor.plot_mass_spectrum(spectrum=spectrum)
    plt.show()


if __name__ == "__main__":
    main()
