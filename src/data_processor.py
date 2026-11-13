
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal.windows import blackmanharris
import yaml
from scipy.integrate import trapezoid
from scipy.signal import find_peaks


class DataProcessor:
    """Input units match the original notebook: x in microseconds, y in millivolts."""

    def __init__(self, x, y, *, config_path=None):
        self.x = np.array(x, dtype=float, copy=True)
        self.y = np.array(y, dtype=float, copy=True)

        self.config = self.load_yaml(config_path)
        
        self.dt_seconds = float((self.x[-1] - self.x[0]) * 1e-6 / (self.x.size - 1))

        self.x.setflags(write=False)
        self.y.setflags(write=False)
        self.dft_freq = self.dft_mag = self.estimated_mz = None
        self.calibration = None
        self.peaks = None

    @classmethod
    def load_data(cls, file_path, *, config_path=None):
        # Native captures carry explicit SI units in their header. Older scope
        # exports have no header and store seconds/volts in columns 4 and 5.
        capture = pd.read_csv(file_path, nrows=0)
        if list(capture.columns) == ["time_s", "voltage_v"]:
            data = pd.read_csv(file_path)
            x = pd.to_numeric(data["time_s"], errors="raise").to_numpy() * 1e6
            y = pd.to_numeric(data["voltage_v"], errors="raise").to_numpy() * 1e3
            return cls(x, y, config_path=config_path)
        data = pd.read_csv(file_path, header=None)
        x = pd.to_numeric(data.iloc[:, 3], errors="raise").to_numpy() * 1e6
        y = pd.to_numeric(data.iloc[:, 4], errors="raise").to_numpy() * 1e3
        return cls(x, y, config_path=config_path)

    @staticmethod
    def load_yaml(file_path=None):
        if file_path is None:
            file_path = Path(__file__).resolve().parents[1] / "config.yaml"
        with open(file_path, encoding="utf-8") as file:
            config = yaml.safe_load(file)
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a YAML mapping.")
        return config

    def dft(self):
        if self.dft_freq is None:
            n = self.y.size
            window = blackmanharris(n, sym=False)
            # Remove baseline BEFORE windowing to reduce DC leakage.
            spectrum = np.fft.rfft((self.y - self.y.mean()) * window)
            amplitude = np.abs(spectrum) / window.sum()
            # One-sided peak amplitude: do not double DC or the even-N Nyquist bin.
            amplitude[1:-1 if n % 2 == 0 else None] *= 2
            self.dft_freq = np.fft.rfftfreq(n, d=self.dt_seconds)
            self.dft_mag = amplitude  # mV, rather than an unnormalized FFT sum
        return self.dft_freq, self.dft_mag

    def _mass_calibration(self):
        calibration = self.config["calibration"]

        def positive_number(key, default=None):
            raw = calibration[key] if default is None else calibration.get(key, default)
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"calibration.{key} must be a number without units. Got {raw!r}."
                ) from exc
            if isinstance(raw, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"calibration.{key} must be finite and positive.")
            return value

        m1 = positive_number("m_ref_1")
        m2 = positive_number("m_ref_2")
        ref_order = positive_number("ref_harmonic_order", 1)
        harmonic_order = positive_number("harmonic_order", 1)
        f1 = positive_number("f_ref_1_khz") / ref_order
        f2 = positive_number("f_ref_2_khz") / ref_order

        use_offset = calibration.get("use_calculated_offset", False)
        if not isinstance(use_offset, bool):
            raise ValueError("calibration.use_calculated_offset must be true or false.")

        sqrt_m1, sqrt_m2 = np.sqrt([m1, m2])
        if sqrt_m1 == sqrt_m2:
            raise ValueError("Calibration requires two distinct reference masses.")

        calculated_f0 = (f1 * sqrt_m1 - f2 * sqrt_m2) / (sqrt_m1 - sqrt_m2)
        f0 = calculated_f0 if use_offset else 0.0
        K = ((f1 - f0) * sqrt_m1) ** 2
        if not np.isfinite(K) or K <= 0:
            raise ValueError("Calibration must yield a finite positive K.")

        return K, f0, harmonic_order, calculated_f0

    @staticmethod
    def _mz_from_khz(observed_khz, K, f0, harmonic_order):
        shifted_khz = np.asarray(observed_khz, dtype=float) / harmonic_order - f0
        denominator = shifted_khz**2
        return np.divide(
            K,
            denominator,
            out=np.full(shifted_khz.shape, np.nan),
            where=denominator > 0,
        )

    def frequency_to_mz(self):
        """Map positive FFT frequencies to calibrated m/z."""
        K, f0, harmonic_order, calculated_f0 = self._mass_calibration()
        calibration = self.config["calibration"]

        frequencies_hz, _ = self.dft()
        observed_khz = frequencies_hz[frequencies_hz > 0] / 1000.0
        self.estimated_mz = self._mz_from_khz(observed_khz, K, f0, harmonic_order)
        self.calibration = str(calibration.get("calibration_label", "Mass calibration"))

        print("\nMass calibration")
        print(f"Calculated two-reference offset = {calculated_f0:.6f} kHz")
        print(f"Applied frequency offset f0 = {f0:.6f} kHz")
        print(f"Calibration constant K = {K:.6e} kHz²·Da")

        return self.estimated_mz

    def detect_peaks(self):
        """Store and return peak measurements in ascending observed frequency.

        Frequencies are in kHz, FFT bin heights are in mV, and calibrated
        m/z values are in Th. Widths are measured at half prominence.
        The detect_peaks config section controls sensitivity: increasing either
        sigma multiplier rejects weaker peaks; increasing the minimum distance
        suppresses nearby maxima in favor of the taller peak. Defaults match
        FFT_decay_analysis_Nov_2026.py (0.50 kHz, 5 sigma, 2 sigma).
        """
        settings = self.config.get("detect_peaks", {})
        if not isinstance(settings, dict):
            raise ValueError("detect_peaks must be a YAML mapping.")

        def nonnegative_number(name, default):
            raw = settings.get(name, default)
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"detect_peaks.{name} must be a number without units. Got {raw!r}."
                ) from exc
            if isinstance(raw, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"detect_peaks.{name} must be finite and nonnegative.")
            return value

        min_distance_khz = nonnegative_number("min_peak_distance_khz", 0.50)
        height_sigma = nonnegative_number("height_sigma", 5.0)
        prominence_sigma = nonnegative_number("prominence_sigma", 2.0)

        frequencies_hz, magnitudes = self.dft()
        if not np.all(np.isfinite(magnitudes)):
            raise ValueError("Peak detection requires a finite FFT spectrum.")
        if frequencies_hz.size < 3:
            self.peaks = []
            return self.peaks

        frequencies_khz = frequencies_hz / 1e3
        bin_spacing_khz = frequencies_khz[1] - frequencies_khz[0]
        distance_bins = max(1, int(np.ceil(min_distance_khz / bin_spacing_khz)))

        # Exclude DC and, for even N, Nyquist from the noise estimate.
        noise_samples = magnitudes[1:-1] if self.y.size % 2 == 0 else magnitudes[1:]
        noise_median = np.median(noise_samples)
        noise_sigma = 1.4826 * np.median(np.abs(noise_samples - noise_median))

        peaks, properties = find_peaks(
            magnitudes,
            height=noise_median + height_sigma * noise_sigma,
            prominence=prominence_sigma * noise_sigma,
            distance=distance_bins,
            width=(None, None),
            rel_height=0.5,
        )
        if peaks.size == 0:
            self.peaks = []
            return self.peaks

        K, f0, harmonic_order, _ = self._mass_calibration()
        pole_khz = f0 * harmonic_order
        bin_indices = np.arange(frequencies_khz.size)
        results = []

        for i, peak in enumerate(peaks):
            left_ip = properties["left_ips"][i]
            right_ip = properties["right_ips"][i]
            f_left, f_right = np.interp(
                [left_ip, right_ip], bin_indices, frequencies_khz
            )
            width_khz = f_right - f_left

            # Fit a parabola through a strict local maximum and its neighbors.
            left, center, right = magnitudes[peak - 1:peak + 2]
            peak_khz = frequencies_khz[peak]
            if center > max(left, right):
                offset_bins = 0.5 * (left - right) / (left - 2 * center + right)
                peak_khz += offset_bins * bin_spacing_khz

            mz, mz_left, mz_right = self._mz_from_khz(
                [peak_khz, f_left, f_right], K, f0, harmonic_order
            )
            width_mz = abs(mz_left - mz_right)
            if f_left <= pole_khz <= f_right:
                width_mz = np.nan

            # Include fractional endpoints at the half-prominence level.
            first_inside = int(np.floor(left_ip)) + 1
            stop_inside = int(np.ceil(right_ip))
            width_height = properties["width_heights"][i]
            integration_freqs = np.r_[
                f_left, frequencies_khz[first_inside:stop_inside], f_right
            ]
            integration_mags = np.r_[
                width_height, magnitudes[first_inside:stop_inside], width_height
            ]

            results.append({
                "freq": peak_khz,
                "height": center,
                "width_kHz": width_khz,
                "width_u": width_mz,
                "left_freq_kHz": f_left,
                "right_freq_kHz": f_right,
                "mass": mz,
                "area": trapezoid(integration_mags, integration_freqs),
                "power": trapezoid(integration_mags**2, integration_freqs),
                "resolution_frequency": peak_khz / width_khz if width_khz > 0 else np.nan,
                "resolution_mass": mz / width_mz if width_mz > 0 else np.nan,
            })

        self.peaks = results
        return self.peaks

    def plot(self, *, mz_limits=(1, 100), frequency_limits_hz=None, peaks=None):
        """Plot computed spectra with stored peaks, or explicitly supplied peaks."""
        if peaks is None:
            peaks = self.peaks
        has_fft = self.dft_freq is not None and self.dft_mag is not None
        has_mz = has_fft and self.estimated_mz is not None

        n_panels = 1 + int(has_fft) + int(has_mz)
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=(6 * n_panels, 4.5),
            squeeze=False,
            layout="constrained",
        )
        axes = axes[0]

        axes[0].plot(self.x, self.y, linewidth=0.7)
        axes[0].set(
            title="Time Domain Signal",
            xlabel="Time (µs)",
            ylabel="Voltage (mV)",
        )

        if has_fft:
            fft_ax = axes[1]
            fft_ax.plot(self.dft_freq, self.dft_mag, linewidth=0.7)
            fft_ax.set(
                title="Frequency Spectrum",
                xlabel="Frequency (Hz)",
                ylabel="Peak amplitude (mV)",
            )
            if frequency_limits_hz is not None:
                fft_ax.set_xlim(*frequency_limits_hz)

        if has_mz:
            mz_ax = axes[-1]
            mz = self.estimated_mz
            magnitude = self.dft_mag[self.dft_freq > 0]
            finite = np.isfinite(mz)
            order = np.argsort(mz[finite])

            mz_ax.plot(mz[finite][order], magnitude[finite][order], linewidth=0.7)
            mz_ax.set(
                title=f"Mass-to-Charge Spectrum\n{self.calibration}",
                xlabel="m/z (Th)",
                ylabel="Peak amplitude (mV)",
            )
            if mz_limits is not None:
                mz_ax.set_xlim(*mz_limits)

            visible = finite & (mz >= min(mz_ax.get_xlim())) & (mz <= max(mz_ax.get_xlim()))
            if visible.any():
                top = max(1.05 * magnitude[visible].max(), np.finfo(float).eps)
                mz_ax.set_ylim(0, top)

        if peaks:
            # detect_peaks() reports frequency in kHz. The FFT axis uses Hz.
            peak_freq_hz = np.asarray([peak["freq"] for peak in peaks]) * 1_000.0
            peak_height_mv = np.asarray([peak["height"] for peak in peaks])
            peak_mz = np.asarray([peak["mass"] for peak in peaks])

            marker_style = {
                "color": "tab:red",
                "marker": "x",
                "s": 36,
                "zorder": 3,
                "label": "Detected peaks",
            }

            if has_fft:
                left, right = sorted(fft_ax.get_xlim())
                visible = (peak_freq_hz >= left) & (peak_freq_hz <= right)
                if visible.any():
                    fft_ax.scatter(
                        peak_freq_hz[visible], peak_height_mv[visible], **marker_style
                    )
                    fft_ax.legend()

            if has_mz:
                left, right = sorted(mz_ax.get_xlim())
                visible = np.isfinite(peak_mz) & (peak_mz >= left) & (peak_mz <= right)
                if visible.any():
                    mz_ax.scatter(peak_mz[visible], peak_height_mv[visible], **marker_style)
                    top = max(mz_ax.get_ylim()[1], 1.05 * peak_height_mv[visible].max())
                    mz_ax.set_ylim(0, top)
                    mz_ax.legend()

        plt.show()
        
        return fig, axes


