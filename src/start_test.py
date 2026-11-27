"""Waveform helpers. Configure and run the experiment through main.py."""

import math
import socket
import numpy as np


def add_white_noise(samples, snr_db, *, headroom_sigma, rng=None):
    """Add independent Gaussian noise at the requested whole-transient SNR.

    Return normalized DAC samples. A fixed scale C = 1 + headroom_sigma*sigma
    gives the configured noise headroom above the unit signal bound. C depends
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

    if not math.isfinite(headroom_sigma) or headroom_sigma <= 0:
        raise ValueError("headroom_sigma must be finite and positive")
    scale = 1.0 + headroom_sigma * sigma


    if not math.isfinite(scale) or (sigma == 0 and np.any(samples)):
        raise ValueError("snr_db is outside the supported numerical range")
    if rng is None:
        rng = np.random.default_rng()


    noisy = (samples + rng.normal(0.0, sigma, samples.shape)) / scale

    if not np.all(np.isfinite(noisy)) or np.any(np.abs(noisy) > 1):
        raise ValueError("noisy waveform exceeds DAC range; use a higher SNR")
    
    return noisy


def generate_square(duration_s, start_s, width_s, high, low, *, waveform_samples):
    """Return (time_s, samples) for a square pulse.

    The pulse is high on [start_s, start_s + width_s) and low elsewhere.
    duration_s and waveform_samples control the complete waveform buffer.
    """

    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if not math.isfinite(start_s) or not 0 <= start_s < duration_s:
        raise ValueError("start_s must be within the waveform duration")
    if not math.isfinite(width_s) or not 0 < width_s <= duration_s - start_s:
        raise ValueError("width_s must be positive and fit within the waveform")
    if not all(math.isfinite(level) and -1 <= level <= 1 for level in (high, low)):
        raise ValueError("high and low must be finite normalized levels in [-1, 1]")

    time_s = np.arange(waveform_samples) * (duration_s / waveform_samples)
    active = (time_s >= start_s) & (time_s < start_s + width_s)
    square = np.where(active, high, low)
    return time_s, square

def generate_triangle(duration_s, tau_s, phase_rad, *, fundamental_hz, waveform_samples):
    """Return (time_s, normalized samples) for the protocol's noiseless transient.

    Samples are at k*T/Ng, excluding T. The fixed unit scaling preserves the
    exponential envelope; A0 is applied separately through the DAC amplitude.
    """

    validate_duration(duration_s)
    if not math.isfinite(tau_s) or tau_s <= 0:
        raise ValueError("tau_s must be finite and positive")
    if not math.isfinite(phase_rad):
        raise ValueError("phase_rad must be finite")
    if not math.isfinite(fundamental_hz) or fundamental_hz <= 0:
        raise ValueError("fundamental_hz must be finite and positive")
    time_s = np.arange(waveform_samples) * (duration_s / waveform_samples)
    triangle = (2 / np.pi) * np.arcsin(
        np.sin(2 * np.pi * fundamental_hz * time_s + phase_rad)
    )
    return time_s, np.exp(-time_s / tau_s) * triangle


def validate_duration(duration_s):
    """Require a positive finite duration; main.py enforces the protocol range."""
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")


def prepare_signal(duration_s, *, tau_s, phase_rad, fundamental_hz, waveform_samples,
                   noise_enabled, snr_db, headroom_sigma, rng):
    """Build one transient and optionally add noise once, without hardware I/O."""

    time_s, samples = generate_triangle(duration_s, tau_s, phase_rad,
                                      fundamental_hz=fundamental_hz,
                                      waveform_samples=waveform_samples)

    if noise_enabled:
        samples = add_white_noise(samples, snr_db, headroom_sigma=headroom_sigma, rng=rng)
    return time_s, samples


def upload_signal(samples, host, port, *, duration_s, amplitude_v, socket_timeout_s):
    """Upload/configure one OUT1 transient, leaving playback to capture_signal()."""
    validate_duration(duration_s)

    samples = np.asarray(samples, dtype=float)
    if not math.isfinite(amplitude_v) or not 0 < amplitude_v <= 1:
        raise ValueError("amplitude_v must be greater than 0 and at most 1 V")
    # The standard arbitrary waveform buffer has a fixed hardware length.
    if samples.shape != (16_384,) or not np.isfinite(samples).all():
        raise ValueError("samples must contain 16384 finite values")
    if np.any(np.abs(samples) > 1):
        raise ValueError("samples must be within [-1, 1]")

    waveform = ",".join(f"{sample:.9g}" for sample in samples)

    with socket.create_connection((host, port), timeout=socket_timeout_s) as sock:
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

        finally:
            send("OUTPUT1:STATE OFF")
