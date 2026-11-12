# ION DataProcessor implementation plan

Follow the steps in order. The snippets are proposed changes to the supplied script. Steps 1–4 build the imports and class, and steps 5–7 add module-level functions and the entry point. Red Pitaya trigger generation is deferred. The first goal is to produce a correctly scaled frequency spectrum from the saved potassium recording. Assign m/z once a reference peak has been verified for that setup.

## 1. Add the imports and use consistent units

- [ ] Add the missing Matplotlib import so the current plot can run. Add SciPy for the FFT taper and peak detection.
- [ ] Use seconds, volts, and hertz internally. The existing loader treats CSV columns 3 and 4 (zero-based) as seconds and volts before converting for display. Verify those source units against the scope export. Continue displaying µs and mV on the time-domain plot.

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert
from scipy.signal.windows import blackmanharris
```

## 2. Validate the CSV and calculate the sample rate

- [ ] Replace the constructor and `load_data` with the versions below. The example preserves the existing headerless CSV format and detector column. Confirm whether that column already contains the differential voltage.
- [ ] Require matching finite arrays, strictly increasing time, and approximately uniform intervals. Use the median of all time differences to estimate sample spacing. The tolerance below is a starting value. Check it against the precision of the exported timestamps.
- [ ] Search existing call sites for `x`, `y`, `dft_freq`, and `dft_mag` before migrating to the named SI attributes and explicit FFT return values.

```python
class DataProcessor:
    def __init__(self, time_s, voltage_v):
        self.time_s = np.asarray(time_s, dtype=float)
        self.voltage_v = np.asarray(voltage_v, dtype=float)

        if self.time_s.ndim != 1 or self.time_s.shape != self.voltage_v.shape:
            raise ValueError("Expected matching one-dimensional time and voltage arrays")
        if self.time_s.size < 2:
            raise ValueError("At least two samples are required")
        if not (np.isfinite(self.time_s).all() and np.isfinite(self.voltage_v).all()):
            raise ValueError("Time and voltage must contain only finite values")

        intervals_s = np.diff(self.time_s)
        if not np.isfinite(intervals_s).all() or np.any(intervals_s <= 0):
            raise ValueError("Time samples must be strictly increasing")
        self.dt_s = float(np.median(intervals_s))
        if not np.allclose(intervals_s, self.dt_s, rtol=1e-4, atol=0):
            raise ValueError("The FFT requires uniformly spaced samples")
        self.fs_hz = 1.0 / self.dt_s

    @classmethod
    def load_data(cls, file_path):
        data = pd.read_csv(file_path, header=None)
        if data.shape[1] < 5:
            raise ValueError("Expected time and voltage in CSV columns 3 and 4")

        time_s = pd.to_numeric(data.iloc[:, 3], errors="raise").to_numpy()
        voltage_v = pd.to_numeric(data.iloc[:, 4], errors="raise").to_numpy()
        return cls(time_s, voltage_v)
```

## 3. Select and record the FFT interval

- [ ] Plot the full trace first. Choose the first usable sample after switching artifacts and clipping. Check where the tail reaches the noise floor.
- [ ] Add this method inside `DataProcessor`. `start_s` is relative to the first CSV timestamp. The duration is converted to the nearest whole sample count. `None` uses the rest of the recording. Preserve the trigger timestamp separately if the export provides one.
- [ ] To reproduce the documented sodium run, choose **10 ms** and a **Blackman–Harris** taper. The original segment offset is unspecified. The sketch's **100 ms** describes the transient span. Try longer FFT segments when the recording supports them. Verify actual duration from the CSV, including for the file with `20ms` in its name.

```python
    def select_segment(self, *, start_s=0.0, duration_s=None):
        if not np.isfinite(start_s) or start_s < 0:
            raise ValueError("Start time must be finite and nonnegative")

        elapsed_s = self.time_s - self.time_s[0]
        first = int(np.searchsorted(elapsed_s, start_s))
        if duration_s is None:
            stop = self.time_s.size
        else:
            if not np.isfinite(duration_s) or duration_s <= 0:
                raise ValueError("Duration must be finite and positive")
            stop = first + round(duration_s * self.fs_hz)

        if stop > self.time_s.size or stop - first < 2:
            raise ValueError("The requested segment must contain at least two recorded samples")
        return slice(first, stop)
```

## 4. Compute and plot a one-sided amplitude spectrum

- [ ] Replace the original `dft()` and `plot()` with these methods. Compute the FFT from the selected interval each time so changes to the interval or taper take effect.
- [ ] Remove the segment mean by default. Apply the periodic Blackman–Harris taper (`sym=False`). Use the rectangular option to compare how the taper affects leakage and peak width.
- [ ] Divide by the sum of the taper weights, then double interior positive-frequency bins. Leave DC and, for even FFT lengths, Nyquist undoubled. The amplitude is in V and recovers the peak voltage of a bin-centered steady tone. A decaying signal gives an amplitude that depends on the selected interval and taper.
- [ ] Log both the native bin spacing `fs_hz / sample_count` and the FFT grid spacing `fs_hz / n_fft`. A 10 ms segment has 100 Hz native spacing. A 100 ms segment has 10 Hz native spacing. Zero-padding densifies the grid. Measured peak widths also depend on decay and taper.

```python
    def dft(self, segment, *, window="blackmanharris", remove_mean=True, n_fft=None):
        voltage_v = self.voltage_v[segment]
        sample_count = voltage_v.size
        if sample_count < 2:
            raise ValueError("The FFT segment must contain at least two samples")
        if n_fft is None:
            n_fft = sample_count
        if not isinstance(n_fft, (int, np.integer)) or n_fft < sample_count:
            raise ValueError("n_fft must be an integer at least as large as the segment")

        if window == "blackmanharris":
            weights = blackmanharris(sample_count, sym=False)
        elif window == "rectangular":
            weights = np.ones(sample_count)
        else:
            raise ValueError(f"Unknown FFT window {window!r}")

        if remove_mean:
            voltage_v = voltage_v - voltage_v.mean()

        transform = np.fft.rfft(voltage_v * weights, n=n_fft)
        freq_hz = np.fft.rfftfreq(n_fft, d=self.dt_s)
        amplitude_v = np.abs(transform) / weights.sum()
        if n_fft % 2 == 0:
            amplitude_v[1:-1] *= 2
        else:
            amplitude_v[1:] *= 2
        return freq_hz, amplitude_v

    def plot(self, spectrum=None, *, segment=None):
        fig, axes = plt.subplots(
            1, 2 if spectrum is not None else 1, figsize=(12, 5), squeeze=False
        )
        time_axis = axes[0, 0]
        time_axis.plot(self.time_s * 1e6, self.voltage_v * 1e3)
        if segment is not None:
            time_axis.axvspan(
                self.time_s[segment.start] * 1e6,
                (self.time_s[segment.stop - 1] + self.dt_s) * 1e6,
                alpha=0.2,
            )
        time_axis.set(xlabel="Time (µs)", ylabel="Detector voltage (mV)")

        if spectrum is not None:
            freq_hz, amplitude_v = spectrum
            spectrum_axis = axes[0, 1]
            spectrum_axis.plot(freq_hz / 1e3, amplitude_v * 1e3)
            spectrum_axis.set(xlabel="Frequency (kHz)", ylabel="Peak amplitude (mV)")

        fig.tight_layout()
        plt.show()
```

## 5. Detect candidate peaks in frequency space

- [ ] Add this helper outside the class. Choose the frequency bounds from the expected fundamentals and harmonics for the run. Choose minimum prominence in volts from the observed noise and peak sizes. The helper reports candidate frequencies at FFT bin centers.
- [ ] Inspect the candidate list and plot. Compare with a background acquisition if available. Label known harmonics and noise before assigning ion identities. Add a sub-bin estimate later if calibration accuracy requires it and verify the estimate on synthetic off-bin tones.

```python
def detect_peaks(freq_hz, amplitude_v, *, min_hz, max_hz, prominence_v):
    if not 0 <= min_hz < max_hz <= freq_hz[-1]:
        raise ValueError("Choose an increasing search band within the FFT grid")
    if not np.isfinite(prominence_v) or prominence_v <= 0:
        raise ValueError("Peak prominence must be finite and positive")

    indices, properties = find_peaks(amplitude_v, prominence=prominence_v)
    in_band = (freq_hz[indices] >= min_hz) & (freq_hz[indices] <= max_hz)
    selected = indices[in_band]
    return pd.DataFrame({
        "frequency_hz": freq_hz[selected],
        "amplitude_v": amplitude_v[selected],
        "prominence_v": properties["prominences"][in_band],
    })
```

## 6. Calibrate using a verified fundamental

- [ ] Supply the reference m/z and its measured fundamental in Hz from the same setup. For a sodium check, the project notes give a nominal reference of **23 at 116 kHz**. For the potassium recording, confirm the K-39 fundamental in that setup. The sketch appears to indicate **120 kHz**. The nominal sodium calibration would predict **89.08 kHz** for K-39 under its own conditions.
- [ ] Convert only positive frequencies. For confirmed harmonic order `h`, divide the peak frequency by `h` before converting to its parent ion's m/z. Unclassified peaks can be plotted on a fundamental-equivalent m/z axis pending assignment. Keep the raw frequency and amplitude.
- [ ] Check calibration error using another identified fundamental under the same conditions when available. A reference point will reproduce its assigned m/z by construction.

```python
def frequency_to_mz(freq_hz, *, ref_hz, ref_mz, harmonic_order=1):
    freq_hz = np.asarray(freq_hz, dtype=float)
    if not np.isfinite(freq_hz).all() or np.any(freq_hz <= 0):
        raise ValueError("Calibration requires finite, positive frequencies")
    if not (np.isfinite(ref_hz) and ref_hz > 0 and np.isfinite(ref_mz) and ref_mz > 0):
        raise ValueError("Reference frequency and m/z must be finite and positive")
    if not isinstance(harmonic_order, (int, np.integer)) or harmonic_order < 1:
        raise ValueError("Harmonic order must be a positive integer")

    fundamental_hz = freq_hz / harmonic_order
    return ref_mz * (ref_hz / fundamental_hz) ** 2


def plot_mz_spectrum(freq_hz, amplitude_v, *, ref_hz, ref_mz):
    positive = freq_hz > 0
    mz = frequency_to_mz(freq_hz[positive], ref_hz=ref_hz, ref_mz=ref_mz)
    order = np.argsort(mz)
    fig, axis = plt.subplots()
    axis.plot(mz[order], amplitude_v[positive][order] * 1e3)
    axis.set(xlabel="Fundamental-equivalent m/z", ylabel="Peak amplitude (mV)")
    fig.tight_layout()
    plt.show()
```

## 7. Run the recording and save the settings used

- [ ] Replace the original entry point with a configurable run function. Inspect the whole trace before choosing `start_s`. For an initial frequency-only run, omit `calibration`. The calibration tuple is ordered as `(measured_reference_hz, reference_mz)`.
- [ ] Record the CSV path and channel, sample rate, interval bounds, taper, mean removal, FFT length, and reference values with the peak table. If exporting results, save the settings alongside the peak CSV. Add the trigger time and ion-source conditions if available.

```python
def analyze_recording(
    file_path, *, start_s, duration_s, min_hz, max_hz, prominence_v, calibration=None
):
    processor = DataProcessor.load_data(file_path)
    segment = processor.select_segment(start_s=start_s, duration_s=duration_s)
    window = "blackmanharris"
    sample_count = segment.stop - segment.start
    freq_hz, amplitude_v = processor.dft(segment, window=window, remove_mean=True)
    peaks = detect_peaks(
        freq_hz, amplitude_v, min_hz=min_hz, max_hz=max_hz, prominence_v=prominence_v
    )

    settings = {
        "file": str(file_path),
        "csv_time_column": 3,
        "csv_voltage_column": 4,
        "fs_hz": processor.fs_hz,
        "segment_start_csv_s": float(processor.time_s[segment.start]),
        "segment_duration_s": sample_count / processor.fs_hz,
        "window": window,
        "remove_mean": True,
        "n_fft": sample_count,
        "native_bin_spacing_hz": processor.fs_hz / sample_count,
        "amplitude_unit": "V peak",
        "peak_band_hz": (min_hz, max_hz),
        "peak_prominence_v": prominence_v,
    }
    if calibration is not None:
        ref_hz, ref_mz = calibration
        settings.update(reference_hz=ref_hz, reference_mz=ref_mz)
        peaks["fundamental_equivalent_mz"] = frequency_to_mz(
            peaks["frequency_hz"], ref_hz=ref_hz, ref_mz=ref_mz
        )

    print(settings)
    print(peaks.to_string(index=False))
    processor.plot((freq_hz, amplitude_v), segment=segment)
    if calibration is not None:
        plot_mz_spectrum(freq_hz, amplitude_v, ref_hz=ref_hz, ref_mz=ref_mz)
    return peaks, settings


def save_run(output_dir, peaks, settings):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    peaks.to_csv(output_dir / "peaks.csv", index=False)
    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    file_path = Path(__file__).resolve().parent / "../data/k_20ms_avg128_26.5degC.csv"
    DataProcessor.load_data(file_path).plot()
```

The entry point first displays the raw trace. After inspecting it, call `analyze_recording` with the interval, search band, and prominence chosen from that trace. Pass its two return values to `save_run` with an output directory for that analysis. Check the data path if the script sits at a different depth from the original layout.

## 8. Verify before interpreting the bench result

- [ ] Run this synthetic check after adding the class and helpers. The synthetic signal includes a DC offset, a known fundamental, and its second harmonic. Confirm the peak list and normalized amplitude, then verify the second harmonic maps to the parent mass when its order is supplied.
- [ ] Repeat with an off-bin tone, a damped tone, odd and even FFT lengths, and malformed timestamps. Verify that a too-short requested segment gives a clear error. Use an independent synthetic frequency-to-m/z pair as well as the reference self-check.
- [ ] When the matching sodium raw trace is available, compare the 10 ms Blackman–Harris result with the meeting deck's approximately **0.08 m/z FWHM**. If measuring FWHM, define the local baseline and interpolate the half-height crossings. The approximately **50 Hz** FWHM near 116 kHz belongs to a separate scope FFT example.

```python
fs_hz = 1_000_000
time_s = np.arange(10_000) / fs_hz
voltage_v = (
    0.020
    + 0.004 * np.sin(2 * np.pi * 116_000 * time_s)
    + 0.001 * np.sin(2 * np.pi * 232_000 * time_s)
)
processor = DataProcessor(time_s, voltage_v)
freq_hz, amplitude_v = processor.dft(processor.select_segment(duration_s=0.010))
peaks = detect_peaks(freq_hz, amplitude_v, min_hz=100_000, max_hz=250_000,
                     prominence_v=0.0005)
np.testing.assert_allclose(peaks["frequency_hz"], [116_000, 232_000], atol=0.1)
np.testing.assert_allclose(peaks["amplitude_v"], [0.004, 0.001], rtol=1e-6)
assert np.isclose(frequency_to_mz(232_000, ref_hz=116_000, ref_mz=23,
                                   harmonic_order=2), 23)
```

## 9. Add the decay estimate if needed for the sketch workflow

- [ ] Fit the envelope separately from the windowed FFT. Use a single dominant component, or isolate the component before fitting when several tones cause beating. Select a fit interval after the switching and clipping region, above the noise floor, and away from trace edges. Check the fit residuals and record the interval and `tau_s`.

```python
def estimate_decay_tau_s(processor, fit_segment):
    voltage_v = processor.voltage_v - np.median(processor.voltage_v)
    envelope_v = np.abs(hilbert(voltage_v))
    time_s = processor.time_s[fit_segment]
    amplitude_v = envelope_v[fit_segment]
    if time_s.size < 3 or np.any(amplitude_v <= 0):
        raise ValueError("Select at least three samples with positive envelope amplitude")

    slope, _ = np.polyfit(time_s - time_s[0], np.log(amplitude_v), deg=1)
    if slope >= 0:
        raise ValueError("The selected envelope does not decay")
    return -1.0 / slope
```

Project settings and source references are recorded in [AGENTS.md](https://chatgpt.com/library?file_id=libfile_b0f3d327412c81919a087d0fa10ace5a&account_id=personal).
