import math
import socket
import time
import numpy as np
import matplotlib.pyplot as plt

HOST = "rp-f0f7aa.local"  # Replace if your board has a different hostname/IP.
PORT = 5000

ONE_SECOND = False  # Protocol: 105 kHz, 10 ms. True selects 1 s debug mode.
SEND_SIGNAL = False  # Standalone preview only; main.py controls hardware playback.
WHITE_NOISE_ENABLED = True

if ONE_SECOND:
    FACTOR = 100
else:
    FACTOR = 1

FUNDAMENTAL_HZ = 105_000 / FACTOR
WAVEFORM_SAMPLES = 16_384
DURATION_S = 0.010 * FACTOR
PERIOD_S = 1.0 / FUNDAMENTAL_HZ

DECAY_TIME_S = 0.003 * FACTOR # Configurable: the protocol does not prescribe tau.
AMPLITUDE_V = 0.8
PHASE_RAD = 0.0

# Single square pulse within DURATION_S; levels are normalized DAC values.
# Physical levels are these values multiplied by AMPLITUDE_V.
SQUARE_START_S = 0.0
SQUARE_WIDTH_S = 0.008
SQUARE_HIGH = 1.0
SQUARE_LOW = 0.0


WHITE_NOISE_SNR_DB = 20.0  # Protocol examples: 20, 10, or 0 dB.
WHITE_NOISE_SEED = None  # Set an integer for a repeatable sequence of uploads.
_noise_rng = np.random.default_rng(WHITE_NOISE_SEED)


def add_white_noise(samples, snr_db=WHITE_NOISE_SNR_DB, *, rng=None):
    """Add independent Gaussian noise at the requested whole-transient SNR.

    Return normalized DAC samples. A fixed scale C = 1 + 6*sigma gives six
    standard deviations of headroom above the unit signal bound. C depends
    only on the noiseless signal and SNR, never on the random realization.
    Both signal and noise are attenuated by C, preserving their SNR. Gaussian
    noise is unbounded, so reject any exceptional overflow instead of clipping.
    Reuse a Generator for repeatable, distinct noise on successive calls.
    """
    samples = np.asarray(samples, dtype=float)


    if samples.ndim != 1 or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("samples must be a nonempty, finite one-dimensional array")
    if np.any(np.abs(samples) > 1):
        raise ValueError("noiseless samples must be within [-1, 1]")
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        sigma = np.sqrt(np.mean(samples ** 2)) * np.power(10.0, -snr_db / 20)

    scale = 1.0 + 6.0 * sigma


    if not math.isfinite(scale) or (sigma == 0 and np.any(samples)):
        raise ValueError("snr_db is outside the supported numerical range")
    if rng is None:
        rng = _noise_rng


    noisy = (samples + rng.normal(0.0, sigma, samples.shape)) / scale

    if not np.all(np.isfinite(noisy)) or np.any(np.abs(noisy) > 1):
        raise ValueError("noisy waveform exceeds DAC range; use a higher SNR")
    
    return noisy


def generate_square(duration_s=None, start_s=None, width_s=None,
                    high=None, low=None):
    """Return (time_s, samples) for a square pulse, using globals by default.

    The pulse is high on [start_s, start_s + width_s) and low elsewhere.
    DURATION_S and WAVEFORM_SAMPLES control the complete waveform buffer.
    """
    duration_s = DURATION_S if duration_s is None else duration_s
    start_s = SQUARE_START_S if start_s is None else start_s
    width_s = SQUARE_WIDTH_S if width_s is None else width_s
    high = SQUARE_HIGH if high is None else high
    low = SQUARE_LOW if low is None else low

    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if not math.isfinite(start_s) or not 0 <= start_s < duration_s:
        raise ValueError("start_s must be within the waveform duration")
    if not math.isfinite(width_s) or not 0 < width_s <= duration_s - start_s:
        raise ValueError("width_s must be positive and fit within the waveform")
    if not all(math.isfinite(level) and -1 <= level <= 1 for level in (high, low)):
        raise ValueError("high and low must be finite normalized levels in [-1, 1]")

    time_s = np.arange(WAVEFORM_SAMPLES) * (duration_s / WAVEFORM_SAMPLES)
    active = (time_s >= start_s) & (time_s < start_s + width_s)
    square = np.where(active, high, low)
    return time_s, square

def generate_triangle(duration_s=DURATION_S, tau_s=DECAY_TIME_S, phase_rad=0.0):
    """Return (time_s, normalized samples) for the protocol's noiseless transient.

    Samples are at k*T/Ng, excluding T. The fixed unit scaling preserves the
    exponential envelope; A0 is applied separately through the DAC amplitude.
    """

    validate_duration(duration_s)
    if not math.isfinite(tau_s) or tau_s <= 0:
        raise ValueError("tau_s must be finite and positive")
    if not math.isfinite(phase_rad):
        raise ValueError("phase_rad must be finite")
    time_s = np.arange(WAVEFORM_SAMPLES) * (duration_s / WAVEFORM_SAMPLES)
    triangle = (2 / np.pi) * np.arcsin(
        np.sin(2 * np.pi * FUNDAMENTAL_HZ * time_s + phase_rad)
    )
    return time_s, np.exp(-time_s / tau_s) * triangle


def validate_duration(duration_s):
    """Require a protocol duration, stretched by FACTOR in debug mode."""
    if not math.isfinite(duration_s) or not 0.001 * FACTOR <= duration_s <= 0.010 * FACTOR:
        raise ValueError(
            f"duration_s must be between {0.001 * FACTOR:g} and "
            f"{0.010 * FACTOR:g} seconds"
        )


def prepare_signal(duration_s=None):
    """Build one transient and optionally add noise once, without hardware I/O."""

    duration_s = DURATION_S if duration_s is None else duration_s
    time_s, samples = generate_triangle(duration_s, DECAY_TIME_S, PHASE_RAD)

    if WHITE_NOISE_ENABLED:
        samples = add_white_noise(samples, WHITE_NOISE_SNR_DB)
    return time_s, samples


def send_signal(samples, host=HOST, port=PORT, duration_s=DURATION_S,
                amplitude_v=AMPLITUDE_V, trigger=True, dummy=False):
    """Send the supplied normalized waveform once on OUT1.

    Set trigger=False to upload only; the receiver then arms IN1 before playback.
    """
    validate_duration(duration_s)

    samples = np.asarray(samples, dtype=float)
    if not math.isfinite(amplitude_v) or not 0 < amplitude_v <= 1:
        raise ValueError("amplitude_v must be greater than 0 and at most 1 V")
    if samples.shape != (WAVEFORM_SAMPLES,) or not np.isfinite(samples).all():
        raise ValueError(f"samples must contain {WAVEFORM_SAMPLES} finite values")
    if np.any(np.abs(samples) > 1):
        raise ValueError("samples must be within [-1, 1]")

    if dummy:
        return
    
    waveform = ",".join(f"{sample:.9g}" for sample in samples)

    with socket.create_connection((host, port), timeout=5) as sock:
        def send(command):
            sock.sendall((command + "\r\n").encode("ascii"))

        try:
            send("OUTPUT1:STATE OFF")

            send("SOUR1:FUNC ARBITRARY")
            send(f"SOUR1:TRAC:DATA:DATA {waveform}")
            # The buffer contains all f0*T oscillations; it plays at 1/T Hz.
            send(f"SOUR1:FREQ:FIX {1 / duration_s:.12f}")
            send(f"SOUR1:VOLT {amplitude_v:g}")
            send("SOUR1:VOLT:OFFS 0")
            send("SOUR1:PHAS 0")

            # Generate exactly one waveform cycle.
            send("SOUR1:BURS:STAT BURST")
            send("SOUR1:BURS:NCYC 1")
            send("SOUR1:BURS:NOR 1")
            send("SOUR1:INITValue 0")
            send("SOUR1:BURS:LASTValue 0")
            send("SOUR1:TRig:SOUR INT")

            if trigger:
                send("OUTPUT1:STATE ON")
                send("SOUR1:TRig:INT")

                print(
                    f"Sending waveform on OUT1: T={duration_s * 1000:g} ms, "
                    f"DAC amplitude={amplitude_v:g} V."
                )
                # Keep the output enabled until the entire waveform finishes.
                time.sleep(duration_s + 0.1)

        finally:
            send("OUTPUT1:STATE OFF")

    if trigger:
        print("Transient complete. OUT1 is off.")



if __name__ == "__main__":
    try:

        time_s, samples = prepare_signal()

        if WHITE_NOISE_ENABLED:
            print(f"White noise enabled: target SNR={WHITE_NOISE_SNR_DB:g} dB; "
                  "signal and noise scaled together for DAC headroom.")

        if SEND_SIGNAL:
            send_signal(samples=samples)

        print(f"Triangle period: {PERIOD_S * 1e6:.3f} µs")

        print(f"Fundamental frequency: {FUNDAMENTAL_HZ:g} Hz")
        print(f"Transient duration: {DURATION_S * 1000:.3f} ms")
        print(f"Oscillations per transient: {FUNDAMENTAL_HZ * DURATION_S:g}")
        print(f"Sample interval: {DURATION_S / WAVEFORM_SAMPLES * 1e6:.3f} µs")

    
        voltage_v = AMPLITUDE_V * samples

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        axes[0].plot(time_s * 1e3, voltage_v)
        axes[0].set(xlabel="Time (ms)", ylabel="Voltage (V)",
                    title="Full transient")

        zoom_duration_s = 50e-6 * FACTOR
        zoom_scale, zoom_unit = (1e3, "ms") if zoom_duration_s >= 1e-3 else (1e6, "µs")
        mask = time_s < zoom_duration_s
        axes[1].plot(time_s[mask] * zoom_scale, voltage_v[mask], ".-")
        axes[1].set(xlabel=f"Time ({zoom_unit})", ylabel="Voltage (V)",
                    title=f"First {zoom_duration_s * zoom_scale:g} {zoom_unit} "
                          "— waveform points shown")

        for ax in axes:
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    except KeyboardInterrupt:
        print("\nInterrupted.")
