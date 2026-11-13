# Pitaya / Racetrap signal processing

Capture IN1, triggered by a rising pulse on IN2, and immediately calculate and
plot the FFT after download (run from the repository root):

```sh
uv run python -m src.start_signal rp-xxxxxx.local --csv data/capture.csv
```

Replace the hostname with your board's address. The SCPI server must be running;
OUT1 must drive your excitation circuit and the trigger pulse must reach IN2.
The default ADC rate assumes a 125 MHz board. The capture lasts 100 ms; download
time is additional. Omit `--csv` to skip saving.

For use in Python, the returned processor already contains the FFT:

```python
from src.start_signal import record_transient

processor = record_transient("rp-xxxxxx.local")
frequencies, amplitudes = processor.dft()
processor.plot()
```

Call `processor.frequency_to_mz(...)` with a confirmed calibration if you also
want the mass-to-charge spectrum. Saved captures can be reopened using
`DataProcessor.load_data("data/capture.csv")`.

FFT peak detection is configured in the repository's `config.yaml`:

```yaml
detect_peaks:
  min_peak_distance_khz: 0.50
  height_sigma: 5.0
  prominence_sigma: 2.0
```

Increase `height_sigma` to reject weaker peaks, or `prominence_sigma` to reject
maxima that stand out less from their surroundings. Both use the robust noise
estimate (1.4826 × median absolute deviation) of the FFT magnitudes. Increase
`min_peak_distance_khz` to keep only the taller of closely spaced peaks; this
can also suppress real neighboring signals. The defaults match the FFT peak
detection settings in `FFT_decay_analysis_Nov_2026.py`.

Settings are loaded when a processor is created, so recreate it after editing
the YAML file. The default config path is independent of the working directory.
Use `DataProcessor.load_data("data/capture.csv", config_path="custom.yaml")`
or `DataProcessor(x, y, config_path="custom.yaml")` to choose another config.
Call `processor.detect_peaks()` to calculate peaks and `processor.plot()` to
display them.
