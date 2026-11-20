# Pitaya / Racetrap signal processing

Generate the protocol's noiseless, exponentially decaying 105 kHz triangular
transient on OUT1:

```sh
uv run python -m src.start_test
```

The SCPI server must be running on `rp-f0f7aa.local`. Adjust `HOST`, `PORT`,
`DURATION_S`, `DECAY_TIME_S`, `AMPLITUDE_V`, and `PHASE_RAD` at the top of
`src/start_test.py`. Duration supports 1 to 10 ms and the decay time must be
positive. The defaults use zero initial phase, a 10 ms duration, a chosen 3 ms
decay time, and 0.8 V initial peak amplitude with zero offset. The 16,384-sample
buffer plays once at `1 / duration`, then OUT1 is disabled. This step implements
signal generation only.

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
