import csv
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.signal import detrend as detrend_signal, find_peaks, peak_widths
from scipy.signal.windows import get_window

from .config import DEFAULTS


class DataProcessor:
    """November FFT workflow with x in seconds, y in volts, frequencies in kHz.

    Assume valid, uniformly sampled input. Preserve timestamps and use the
    mean interval for FFT without resampling voltage.
    Keep raw voltage via ``y`` and DC-corrected voltage via ``y_ac``.
    sampling_rtol is accepted for compatibility and is unused.
    """

    def __init__(self, x, y, *, padding_factor=DEFAULTS["fft"]["padding_factor"],
                 cutoff_khz=DEFAULTS["fft"]["cutoff_khz"], sampling_rtol=0.01,
                 window="blackmanharris", window_parameter=None, window_symmetric=True,
                 detrend="constant", fft_length=None):
        self.x = np.array(x, dtype=float, copy=True)
        self.y = np.array(y, dtype=float, copy=True)

        intervals = np.diff(self.x)
        self._dt_seconds = float(intervals.mean())
        self.y_ac = self.y - self.y.mean()
        self._padding_factor = int(padding_factor)
        self._cutoff_khz = cutoff_khz
        self._fft_length = fft_length
        self._detrend = detrend
        # Keep the raw capture separate from the signal used for analysis.
        self.y_analysis = self.y if detrend == "none" else (
            self.y_ac if detrend == "constant" else detrend_signal(self.y, type="linear"))
        self.y_analysis.setflags(write=False)
        specification = window if window_parameter is None else (window, window_parameter)
        self._window = get_window(specification, self.y.size, fftbins=not window_symmetric)
        self._window.setflags(write=False)

        # Freeze input arrays so cached results stay consistent with the capture.
        self.x.setflags(write=False)
        self.y.setflags(write=False)
        self.y_ac.setflags(write=False)
        self.dft_freq = self.dft_mag = self.estimated_mz = None
        self.mz_mask = None
        self.calibration = None
        self.peaks = self.harmonic_families = None
        self.peak_mask = self.cleaned_mag = self.mz_axis = None

    @property
    def dt_seconds(self):
        return self._dt_seconds

    @property
    def padding_factor(self):
        """FFT configuration is fixed; construct a new processor to change it."""
        return self._padding_factor

    @property
    def cutoff_khz(self):
        return self._cutoff_khz

    @classmethod
    def load_data(cls, file_path, **kwargs):
        """Load seconds/volts from native captures or legacy scope CSVs."""
        # Native captures carry explicit SI units in their header. Older scope
        # exports have no header and store seconds/volts in columns 4 and 5.
        with open(file_path, encoding="utf-8-sig") as file:
            header = file.readline().strip().split(",")
        if header == ["time_s", "voltage_v"]:
            data = pd.read_csv(file_path)
            x = pd.to_numeric(data["time_s"], errors="raise").to_numpy()
            y = pd.to_numeric(data["voltage_v"], errors="raise").to_numpy()
            return cls(x, y, **kwargs)

        # Some scope exports include metadata before the numerical rows.
        samples = []
        number = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
        with open(file_path, encoding="utf-8-sig") as file:
            for columns in csv.reader(file):
                if not samples and (len(columns) < 5 or not number.fullmatch(columns[3].strip())):
                    continue
                samples.append((float(columns[3]), float(columns[4])))
        data = np.asarray(samples)
        return cls(data[:, 0], data[:, 1], **kwargs)

    def dft(self):
        """Return cached kHz/V FFT; default Blackman-Harris window and 8N FFT.

        Normalize by window sum; double bins except DC and even-length Nyquist.
        Padding refines bin spacing; it does not improve physical resolution.
        Set cutoff_khz=None at construction to retain the full frequency range.
        Configure window, detrending, and an optional explicit fft_length there.
        """
        if self.dft_freq is None:
            n = self.y.size
            window = self._window
            n_fft = self._fft_length or self.padding_factor * n
            spectrum = np.fft.rfft(self.y_analysis * window, n=n_fft)
            # Recover one-sided peak amplitudes, accounting for the window gain.
            amplitude = 2 * np.abs(spectrum) / window.sum()
            amplitude[0] /= 2
            if n_fft % 2 == 0:
                amplitude[-1] /= 2
            # Convert Hz to kHz before applying the configured frequency cutoff.
            frequencies = np.fft.rfftfreq(n_fft, d=self.dt_seconds) / 1e3
            keep = np.ones(frequencies.size, dtype=bool)
            if self.cutoff_khz is not None:
                keep = frequencies <= self.cutoff_khz
            self.dft_freq = frequencies[keep]
            self.dft_mag = amplitude[keep]
        return self.dft_freq, self.dft_mag

    def detect_peaks(self, *, min_distance_khz=0.50, prominence_sigma=2, height_sigma=5,
                     width_rel_height=0.5, integration_width_factor=2):
        """Measure FFT peaks using November's median/MAD noise thresholds.

        Default widths are at half prominence. Areas (V kHz) and powers (V² kHz)
        default to integration over ±2 rounded-up widths, using the original FFT.
        Calling again replaces detections and invalidates families/cleaning.
        """
        frequencies, magnitude = self.dft()
        self.peaks = []
        self.harmonic_families = self.peak_mask = self.cleaned_mag = None
        # Median absolute deviation estimates noise without being dominated by peaks.
        self.noise_median = float(np.median(magnitude))
        self.noise_sigma = float(1.4826 * np.median(np.abs(magnitude - self.noise_median)))
        if frequencies.size < 3:
            return self.peaks

        spacing = frequencies[1] - frequencies[0]
        # SciPy measures minimum peak separation in bins, rather than kHz.
        distance = max(1, int(np.ceil(min_distance_khz / spacing)))
        indices, _ = find_peaks(
            magnitude, distance=distance,
            prominence=prominence_sigma * self.noise_sigma,
            height=self.noise_median + height_sigma * self.noise_sigma)
        if indices.size == 0:
            return self.peaks
        widths, _, left_ips, right_ips = peak_widths(magnitude, indices, rel_height=width_rel_height)
        bins = np.arange(frequencies.size)
        for index, width, left_ip, right_ip in zip(indices, widths, left_ips, right_ips):
            left_khz, right_khz = np.interp([left_ip, right_ip], bins, frequencies)
            # Clip each integration region to the available FFT bins.
            radius = int(np.ceil(integration_width_factor * np.ceil(width)))
            left = max(0, int(index) - radius)
            right = min(magnitude.size - 1, int(index) + radius)
            region = slice(left, right + 1)
            self.peaks.append({
                "index": int(index), "freq_khz": float(frequencies[index]),
                "height": float(magnitude[index]), "width_khz": float(right_khz - left_khz),
                "left_freq_khz": float(left_khz), "right_freq_khz": float(right_khz),
                "left_index": left, "right_index": right,
                "area": float(trapezoid(magnitude[region], frequencies[region])),
                "power": float(trapezoid(magnitude[region] ** 2, frequencies[region])),
            })
        return self.peaks

    def detect_harmonics(self, *, max_harmonic=10, tolerance=0.01):
        """Group peaks lowest-frequency first, matching integer multiples.

        Each peak belongs to one family at most. The closest unused peak within
        the relative tolerance wins. An unmatched peak starts a singleton family.
        These are candidate families, not confirmed chemical identifications.
        """
        if self.peaks is None:
            self.detect_peaks()
        peaks = sorted(self.peaks, key=lambda peak: peak["freq_khz"])
        used = set()
        self.harmonic_families = []
        for peak in peaks:
            # Each unused peak starts a family; matched harmonics cannot be reused.
            if peak["index"] in used:
                continue
            used.add(peak["index"])
            members = [{**peak, "order": 1}]
            for order in range(2, max_harmonic + 1):
                target = order * peak["freq_khz"]
                candidates = [candidate for candidate in peaks
                              if candidate["index"] not in used
                              and abs(candidate["freq_khz"] - target) / target < tolerance]
                if candidates:
                    match = min(candidates, key=lambda candidate: abs(candidate["freq_khz"] - target))
                    used.add(match["index"])
                    members.append({**match, "order": order})
            self.harmonic_families.append({
                "index": peak["index"], "freq_khz": peak["freq_khz"],
                "harmonics": members, "fundamental_power": peak["power"],
                "total_power": sum(member["power"] for member in members),
            })
        return self.harmonic_families

    def clean_spectrum(self):
        """Zero bins outside detected peak neighborhoods; retain peak shapes.

        Keep the same configured regions used for integration, including harmonics.
        This prepares display amplitudes, without changing the waveform or FFT.
        No detected peaks produces an all-zero spectrum.
        """
        if self.peaks is None:
            self.detect_peaks()
        self.peak_mask = np.zeros(self.dft_mag.size, dtype=bool)
        # Combine peak regions into one mask, retaining their original amplitudes.
        for peak in self.peaks:
            self.peak_mask[peak["left_index"]:peak["right_index"] + 1] = True
        self.cleaned_mag = np.where(self.peak_mask, self.dft_mag, 0.0)
        return self.cleaned_mag

    def frequency_to_mz(self, *, ref_khz=DEFAULTS["calibration"]["ref_khz"],
                        ref_mz=DEFAULTS["calibration"]["ref_mz"], harmonic_order=1,
                        ref_harmonic_order=1, offset_khz=0.0, mz_range=(1, 200),
                        calibration_label="Unconfirmed reference"):
        """m/z = K / (f / order - offset_khz)**2, with K from a reference.

        Default reference: 38.9637065 at 105.912 kHz, with zero offset, matching
        November's applied calibration. Its second reference is not used because
        that script overrides the calculated two-reference offset with zero.
        ref_khz and offset_khz are in kHz. Set mz_range=None to retain all mappings.
        Use the reference harmonic separately from the spectrum's harmonic.
        One harmonic order is assumed for this entire mapped spectrum; harmonics
        of a single species can otherwise look like extra species.
        This maps detector amplitude to m/z; it is not abundance or a density.
        """

        if mz_range is not None:
            lower, upper = mz_range

        frequencies, _ = self.dft()
        corrected = frequencies / harmonic_order - offset_khz
        # Only positive corrected frequencies have a physical mass mapping.
        self.mz_mask = (frequencies > 0) & (corrected > 0)
        k = ref_mz * (ref_khz / ref_harmonic_order - offset_khz) ** 2
        mapped = np.full(frequencies.shape, np.nan)
        mapped[self.mz_mask] = k / corrected[self.mz_mask] ** 2
        if mz_range is not None:
            self.mz_mask &= (mapped >= lower) & (mapped <= upper)
        self.estimated_mz = mapped[self.mz_mask]
        self.mz_axis = mapped
        self.calibration = str(calibration_label)
        if self.harmonic_families is None:
            self.detect_harmonics()
        self.clean_spectrum()
        return self.estimated_mz

    def plot(self, *, mz_limits=(1, 200), frequency_limits_khz=None, clean_mz=True,
             show=True, show_harmonics=True):
        """One plot() interface: time only, time+FFT, or time+FFT+m/z by state.

        mz_limits is a display zoom, not a data filter. Set it to None for all
        stored masses. The mapping's mz_range controls the actual data filter.
        By default, m/z uses cleaned amplitudes; clean_mz=False shows the raw FFT.
        Set show=False to return the figure without opening a display.
        Red markers identify candidate fundamentals; orange markers show harmonics.
        """
        has_fft = self.dft_freq is not None and self.dft_mag is not None
        has_mz = self.estimated_mz is not None
        if has_fft and self.harmonic_families is None:
            self.detect_harmonics()

        # Add panels only for processing stages that have already been computed.
        fig, axes = plt.subplots(1, 1 + int(has_fft) + int(has_mz),
                                 figsize=(6 * (1 + int(has_fft) + int(has_mz)), 4.5),
                                 squeeze=False, layout="constrained")
        axes = axes[0]
        axes[0].plot(self.x, self.y_analysis, linewidth=0.7)
        signal_title = {"constant": "DC-Corrected Signal", "linear": "Detrended Signal",
                        "none": "Raw Signal"}[self._detrend]
        axes[0].set(title=signal_title, xlabel="Time (s)", ylabel="Voltage (V)")
        if has_fft:
            axes[1].plot(self.dft_freq, self.dft_mag, linewidth=0.7)
            axes[1].scatter([peak["freq_khz"] for peak in self.peaks],
                            [peak["height"] for peak in self.peaks], color="red", s=9)
            axes[1].set(title="Frequency Spectrum", xlabel="Frequency (kHz)",
                        ylabel="Peak amplitude (V)")
            if frequency_limits_khz is not None:
                axes[1].set_xlim(*frequency_limits_khz)
        if has_mz:
            mz = self.estimated_mz
            spectrum = self.clean_spectrum() if clean_mz else self.dft_mag
            magnitude = spectrum[self.mz_mask]
            order = np.argsort(mz)  # inverse-square mapping reverses the frequency order
            ax = axes[-1]
            ax.plot(mz[order], magnitude[order], linewidth=0.7)
            ax.set(title=f"Mass-to-Charge Spectrum\n{self.calibration}",
                   xlabel="m/z (Th)", ylabel="Peak amplitude (V)")
            for family in self.harmonic_families:
                for peak in family["harmonics"]:
                    if peak["order"] > 1 and not show_harmonics:
                        continue
                    index = peak["index"]
                    if not self.mz_mask[index]:
                        continue
                    mass = self.mz_axis[index]
                    if mz_limits is not None and not mz_limits[0] <= mass <= mz_limits[1]:
                        continue
                    color = "red" if peak["order"] == 1 else "darkorange"
                    ax.plot(mass, spectrum[index], "o", color=color, markersize=3)
                    label = f"{mass:.2f}" if peak["order"] == 1 else f"H{peak['order']}"
                    ax.annotate(label, (mass, spectrum[index]), xytext=(0, 6),
                                textcoords="offset points", ha="center", fontsize=7)
            if mz_limits is not None:
                ax.set_xlim(*mz_limits)
                visible = (mz >= mz_limits[0]) & (mz <= mz_limits[1])
                if visible.any():
                    ax.set_ylim(0, max(float(magnitude[visible].max()) * 1.05,
                                       np.finfo(float).eps))
        if show:
            plt.show()
        return fig, axes
