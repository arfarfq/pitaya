# Pitaya / Racetrap signal processing

See the [environment setup guide](docs/pitaya-environment-guide.md) for installation
and notebook setup. Run the workflow with the project's locked environment:

```sh
uv sync --locked
uv run --locked python main.py
```

`main()` lives in `main.py`. Its `CONFIG_PATH` explicitly points to the
`config.yaml` beside that file. Change this constant to select a different
configuration. There are no command-line settings or argument parsers.

All runtime settings are in YAML: input, FFT, peak detection, harmonic grouping,
calibration, plotting, and acquisition. Set `input.source` to `file` to analyze
`input.file`, or `acquisition` to capture using the `acquisition` section.
To save a figure without opening a window, set `plot.show: false` and
`plot.save: spectrum.png`.

Partial YAML files are supported: omitted settings keep their built-in defaults.
Configuration and waveform data are assumed to be correct; explicit input
validation is omitted. Paths in YAML are relative to that YAML's directory,
independently of the working directory.
The dependency minimums support the tested NumPy 1.26.4 environment;
`uv.lock` selects the versions used by the project command above.

The sections are:

- `input`: source (`file` or `acquisition`) and waveform CSV path.
- `fft`: window name and optional parameter (for example `kaiser` with beta
  `8.6`), symmetric or periodic window, detrending (`constant`, `linear`, or
  `none`), padding factor, explicit FFT length, and cutoff in kHz.
- `peaks`: frequency separation, noise thresholds, width measurement height,
  and integration radius multiplier.
- `harmonics`: maximum order and relative frequency tolerance.
- `calibration`: enable m/z mapping, reference frequency and m/z, reference and
  spectrum harmonic orders, offset, mapped range, and calibration label.
- `plot`: limits, cleaning, harmonic annotations, display, saved path, and DPI.
- `acquisition`: host, port, ADC rate, decimation, capture duration, trigger
  level, timeouts, block size, pulse shape/frequency/amplitude/offset, and raw CSV path.

`fft.fft_length` overrides `padding_factor` and must be at least the number of
input samples to avoid truncating the signal. Padding changes bin spacing, not
physical resolution. Amplitude
normalization remains window-sum corrected, one-sided peak amplitude in volts.
The time plot shows the selected detrended signal before windowing; raw voltage
remains available as `processor.y`.

Set `plot.show_harmonics: false` to annotate only family fundamentals on the m/z
plot. Set `plot.clean_mz: false` as well to display the full converted spectrum,
matching November's presentation. Harmonic classification remains a frequency
match rather than a confirmed identification.

The same config can be used from Python:

```python
from main import analyze
from src.config import load_config
from src.data_processor import DataProcessor

config = load_config("config.yaml")
processor = DataProcessor.load_data(config["input"]["file"], **config["fft"])
analyze(processor, config)
figure, axes = processor.plot(show=False)
```

To capture from the board, set `input.source: acquisition`, set
`acquisition.host` to your board's address, and optionally set `acquisition.csv`
to archive the raw capture. Run the same `python main.py` workflow. FFT,
detection, calibration, and plot settings all come from the same YAML file.
Set `calibration.enabled: false` for a time/FFT-only plot.

The SCPI server must be running. Hardware routing remains OUT1 excitation,
IN1 acquisition, and a rising trigger on IN2. The default ADC rate assumes a
125 MHz board, with a 100 ms capture; download time is additional. Permitted
acquisition settings depend on the connected board.

The capture returns a processor containing the raw waveform. In `main.py`,
`analyze()` computes the FFT before peak detection, harmonic grouping, and calibration.
For use directly in Python, compute the FFT explicitly:

```python
from src.start_signal import record_transient

processor = record_transient("rp-xxxxxx.local")
frequencies, amplitudes = processor.dft()
processor.plot()
```

Call `processor.frequency_to_mz(...)` with a confirmed calibration if you also
want the mass-to-charge spectrum. Saved captures can be reopened using
`DataProcessor.load_data("data/capture.csv")`.

`DataProcessor(x, y)` expects time in **seconds** and voltage in **volts**.
CSV loading uses these same units. `dft()` returns frequencies in **kHz** and
peak amplitudes in **volts**. Frequency options also use kHz: `cutoff_khz`
(default 500), `ref_khz` (default 105.912), `offset_khz`, and
`frequency_limits_khz`. Mass-to-charge values remain in Th.

Original timestamps are preserved. The FFT uses their mean sample interval and
assumes uniformly sampled data. Substantial gaps require correcting or resampling
the data first. `sampling_rtol` is retained for compatibility and has no effect.
Legacy scope metadata before waveform samples is skipped; waveform rows are
converted directly to numbers.

`padding_factor`, `cutoff_khz`, and `dt_seconds` are read-only. Construct a new
processor to change FFT settings. One-sided amplitudes leave DC and the Nyquist
bin (for even FFT lengths) undoubled.

For saving or embedding figures in Python, `figure, axes = processor.plot(show=False)`
returns the plot without displaying it. The default remains `show=True`.

Mapping to m/z automatically detects FFT peaks, groups candidate harmonic
families, and prepares a cleaned spectrum. The m/z plot is flat at zero outside
peak neighborhoods; peak shapes and detected harmonics are retained. The FFT
plot and `dft_mag` retain the original spectrum. Red m/z markers label candidate
fundamentals with their m/z; orange markers label harmonics as H2, H3, etc.

For explicit control before mapping or plotting:

```python
peaks = processor.detect_peaks(min_distance_khz=0.50, prominence_sigma=2, height_sigma=5)
families = processor.detect_harmonics(max_harmonic=10, tolerance=0.01)
cleaned_amplitudes = processor.clean_spectrum()  # V, same bins as dft_freq
processor.frequency_to_mz()
processor.plot()                 # cleaned m/z spectrum
processor.plot(clean_mz=False)   # original m/z amplitudes
```

Peak thresholds use the FFT median and a robust noise estimate (1.4826 × median
absolute deviation), matching the November script. Peak dictionaries include
frequency and half-prominence width in kHz, height in V, integrated area in
V kHz, and integrated squared amplitude in V² kHz. Families contain their
members' harmonic orders and summed powers. Grouping starts with the lowest
unused frequency and assigns each peak once, so families are candidate matches.
Cleaning keeps ±2 rounded-up peak widths around each peak and sets all other
bins to zero. It does not filter the time-domain signal or remove harmonics.

Run the regression tests with:

```sh
uv run --locked python -m unittest discover -s tests
```
