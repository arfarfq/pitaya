# Pitaya / Racetrap signal processing

Configure the experiment in `config.yaml`, then run:

```sh
uv run python main.py
```

`main.py` loads `config.yaml` at startup, using a path relative to the source
file rather than the working directory. Settings are grouped under `mode`,
`connection`, `waveform`, `noise`, `acquisition`, `output`, and `plot`.
Output paths are resolved against the YAML file's directory. For example:

```yaml
mode:
  dummy: false
connection:
  host: rp-f0f7aa.local
```

`src/start_test.py` contains waveform/upload helpers, and
`src/receive_signal.py` contains acquisition, CSV saving and plotting helpers.
They receive the loaded settings as function arguments. The flow is
`prepare_signal()` → `upload_signal()` → `capture_signal()`. Upload only stores
and configures the waveform; capture arms the ADC and starts playback.
`main.py` is the sole execution entry point; helper modules have no `main()`
functions or execution blocks. Settings are read directly from local config
sections; debug scaling is calculated inside `run(config, rng)`. The random
generator is created once in `main()` and passed into `run()`.
Restart the script after
editing the YAML file. `config_old.yaml` is retained as the previous configuration.

The default waveform is a 105 kHz triangular transient lasting 10 ms, with a
3 ms exponential decay and 20 dB added white noise. `waveform.amplitude_v: 0.8` sets the
DAC amplitude. Signal and noise share a fixed scaling factor
`1 + noise.headroom_sigma * sigma` to avoid DAC overflow while preserving SNR.
Use `noise.enabled: false` for a noiseless waveform. A fixed noise seed
reproduces the sequence between program launches; repeated calls within one
process use fresh noise from the same random generator.

`waveform.one_second: true` scales the default duration and decay by `waveform.debug_time_factor` and divides
frequency by the same factor, giving a 1 s transient at 1,050 Hz. `FACTOR` is derived from
that flag. The standard arbitrary waveform buffer requires 16,384 samples.

With `mode.dummy: true`, `main.run(config, rng)` returns the generated timestamps and
voltage samples without uploading or calling acquisition.

With `mode.dummy: false`, connect OUT1 to IN1 and start the board's SCPI server.
Check that `connection.host` matches the board and `acquisition.adc_rate_hz` matches its ADC clock.
The workflow uploads without playback, starts sampling, arms IN1 on `AWG_PE`,
triggers the generator, waits for capture completion and downloads the samples.
The default capture includes 1 ms before and after the transient. DMA allocation
uses bytes; trigger delays and read positions use samples. OUT1 and acquisition
are disabled after capture, including on failure.

Both modes return `(time_s, voltage_v)` from `main.run(config, rng)`. `main.main()` also saves
samples to a timestamped CSV in `outputs/` when `output.save_data: true`, with columns
`time_s` and `voltage_v`. Files start with `capture_` for hardware or `generated_`
for dummy mode. `plot.enabled: true` displays voltage against time in milliseconds.
The processing helper remains a placeholder.

The older `src/start_signal.py` pulse/IN2 acquisition workflow is separate from
this OUT1-to-IN1 pipeline. Use `main.py` for the triangular-transient experiment.
