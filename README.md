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
