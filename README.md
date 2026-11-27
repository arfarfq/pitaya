# Pitaya / Racetrap signal processing

Generate a triangular transient and acquire it through one entry point:

```sh
uv run python main.py
```

Configure `DUMMY_MODE`, `HOST`, `PORT`, `DURATION_S`, and `AMPLITUDE_V` at the top of `main.py`.
`DUMMY_MODE = True` runs locally; set it to `False` to use the board.
Dummy mode prepares the waveform and its configured noise locally and returns
the generated time axis and voltage samples. It skips both upload and
`receive_signal()` completely: no sockets, simulated acquisition, or capture margins.

Hardware mode requires OUT1 connected to IN1 and the SCPI server running. It
uploads without playback, arms IN1 on `AWG_PE`, then triggers exactly one burst
and downloads the samples. Both modes return `(time_s, voltage_v)` from
`main.run()`. `main.py` saves the returned samples as a timestamped CSV under
`outputs/`, with columns `time_s` and `voltage_v`. Hardware files start with
`capture_`; dummy files start with `generated_`. Configure `OUTPUT_DIR` or set
`SAVE_DATA = False` in `main.py` to disable saving. The waveform is then plotted
with time in milliseconds and voltage in volts; set `PLOT_DATA = False` to
disable the plot. The processing helper in `receive_signal.py` remains a placeholder.

Both modes use the waveform/noise settings in `src/start_test.py`, including
`ONE_SECOND`. The current `ONE_SECOND = False` selects 10 ms at 105 kHz;
set it to `True` for a 1 s, 1,050 Hz debug transient. `main.DURATION_S` overrides the
duration within the selected mode's allowed range. `SEND_SIGNAL` controls only
the standalone `start_test.py` preview; `main.py` uses `DUMMY_MODE`.
For standalone reception, edit the globals in `src/receive_signal.py` and run
`uv run python -m src.receive_signal`. Its default is hardware capture of an
already uploaded waveform. Dummy mode is controlled only by `main.py`.

For the current hardware settings, confirm `HOST` in `src/start_test.py` matches
your board (currently `rp-f0f7aa.local`), connect OUT1 to IN1, and start the SCPI
server on the board. `main.DUMMY_MODE = False` uploads and captures one transient.
Leave `SEND_SIGNAL = False`: it only controls the standalone preview script.

Preview the exponentially decaying 105 kHz triangular transient:

```sh
uv run python -m src.start_test
```

`SEND_SIGNAL = False` previews the waveform without connecting to hardware.
Set it to `True` to send on OUT1; the SCPI server must be running at `HOST:PORT`
(default `rp-f0f7aa.local:5000`). Adjust `FUNDAMENTAL_HZ`, `DURATION_S`,
`DECAY_TIME_S`, `AMPLITUDE_V`, and `PHASE_RAD` in `src/start_test.py`.
The defaults use 105 kHz, a 10 ms duration, a chosen 3 ms decay time, zero phase,
and a 0.8 V DAC peak setting with zero offset. Protocol durations are 1–10 ms;
decay time must be finite and positive, and phase must be finite.
The 16,384-sample buffer plays once at `1 / duration`, then OUT1 is disabled.

`ONE_SECOND = False` is the protocol default. Set it to `True` for debugging:
the defaults stretch duration and decay time by 100 and reduce frequency by
100, giving a 1 s transient at 1,050 Hz. Allowed durations become 0.1–1 s.
The second plot scales from the first 50 µs to the first 5 ms in this mode.
For a square pulse, replace the `generate_triangle(...)` call in the main block
with `generate_square()`; configure `SQUARE_START_S`, `SQUARE_WIDTH_S`,
`SQUARE_HIGH`, and `SQUARE_LOW`. Pulse levels are normalized values in [-1, 1]
and are multiplied by `AMPLITUDE_V` for output.

Gaussian white noise is enabled by default through
`WHITE_NOISE_ENABLED = True` in `src/start_test.py`. Choose
`WHITE_NOISE_SNR_DB` (20, 10, or 0 dB are the protocol's starting values).
Noise RMS is calculated from the noiseless RMS over the full transient.
The main block adds noise once before sending and plotting, even in preview
mode. `send_signal(samples, ...)` uploads the supplied samples unchanged.
Each call to `add_white_noise` draws a fresh sequence; set `WHITE_NOISE_SEED` to an integer
before starting the script to reproduce a sequence of distinct uploads.
With noise enabled, both signal and noise are divided by the fixed factor
`1 + 6 * noise_rms` in normalized units to provide DAC headroom. This reduces
the physical signal amplitude from `AMPLITUDE_V` while preserving SNR; the
factor stays constant across repetitions with the same waveform and SNR.
Exceptional samples outside the DAC range raise an error before connecting.
The plot uses the exact waveform prepared for the board. Set
`WHITE_NOISE_ENABLED = False` for the noiseless reference at the original amplitude.

Capture IN1, triggered by a rising pulse on IN2, and immediately calculate and
plot the FFT after download (run from the repository root):

```sh
uv run python -m src.start_signal rp-xxxxxx.local --csv data/capture.csv
```

Replace the hostname with your board's address. The SCPI server must be running;
This standalone acquisition script generates its own 8 µs pulse on OUT1;
the trigger pulse must reach IN2 and the response must reach IN1. It does not
yet acquire the triangular transient generated by `start_test.py` in one
integrated workflow. The default ADC rate assumes a 125 MHz board, decimated
by 64.

The capture duration follows `start_test.DURATION_S`, with 1 ms before the IN2
trigger and 1 ms after the expected transient (about 12 ms total by default).
`PRE_TRIGGER_S` and `POST_TRIGGER_S` configure these margins. Sample counts
round up to the ADC grid; the DMA buffer is aligned to eight samples.
Time zero marks the trigger, so pre-trigger times are negative. Download time
is additional. Use `--duration 0.005` to specify a 5 ms transient independently
of the generator defaults, or pass `duration_s`, `pre_trigger_s`, and
`post_trigger_s` as keyword arguments to `record_transient`. Omit `--csv` to skip saving.

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
