import math
import socket
import time
import numpy as np
import matplotlib.pyplot as plt

HOST = "rp-f0f7aa.local"
PORT = 5000

ONE_SECOND = True

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


def square_wave():
    time_s = np.arange(WAVEFORM_SAMPLES) * (DURATION_S / WAVEFORM_SAMPLES)
    square = np.where(time_s < 0.008, 1.0, 0.0)
    return time_s, square

def generate_triangle(duration_s=DURATION_S, tau_s=DECAY_TIME_S, phase_rad=0.0):
    """Return (time_s, normalized samples) for the protocol's noiseless transient.

    Samples are at k*T/Ng, excluding T. The fixed unit scaling preserves the
    exponential envelope; A0 is applied separately through the DAC amplitude.

    if not math.isfinite(duration_s) or not 0.001 <= duration_s <= 0.010:
        raise ValueError("duration_s must be between 0.001 and 0.010 seconds")
    if not math.isfinite(tau_s) or tau_s <= 0:
        raise ValueError("tau_s must be finite and positive")
    if not math.isfinite(phase_rad):
        raise ValueError("phase_rad must be finite")
    """


    time_s = np.arange(WAVEFORM_SAMPLES) * (duration_s / WAVEFORM_SAMPLES)
    triangle = (2 / np.pi) * np.arcsin(
        np.sin(2 * np.pi * FUNDAMENTAL_HZ * time_s + phase_rad)
    )
    return time_s, np.exp(-time_s / tau_s) * triangle


def send_triangle(host, port=PORT, *, duration_s=DURATION_S,
                  tau_s=DECAY_TIME_S, amplitude_v=AMPLITUDE_V, phase_rad=0.0):
    """Send one exponentially decaying 105 kHz triangular transient on OUT1.

    amplitude_v is A0 in volts (peak, not peak-to-peak), with zero DC offset.
    The complete transient occupies one arbitrary-waveform buffer period.
    """
    _, samples = generate_triangle(duration_s, tau_s, phase_rad)

    if not math.isfinite(amplitude_v) or not 0 < amplitude_v <= 1:
        raise ValueError("amplitude_v must be greater than 0 and at most 1 V")
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

            send("OUTPUT1:STATE ON")
            send("SOUR1:TRig:INT")

            print(
                f"Sending triangular transient on OUT1: {FUNDAMENTAL_HZ:g} Hz, "
                f"T={duration_s * 1000:g} ms, tau={tau_s * 1000:g} ms, "
                f"A0={amplitude_v:g} V."
            )

            # Keep the output enabled until the entire waveform finishes.
            time.sleep(duration_s + 0.1)

        finally:
            send("OUTPUT1:STATE OFF")

    print("Transient complete. OUT1 is off.")


if __name__ == "__main__":
    try:

        send_triangle(HOST, PORT, duration_s=DURATION_S, tau_s=DECAY_TIME_S,amplitude_v=AMPLITUDE_V, phase_rad=PHASE_RAD)
        print(f"Triangle period: {PERIOD_S * 1e6:.3f} µs")

        print(f"Triangle period: {1e6 / FUNDAMENTAL_HZ:.3f} µs")
        print(f"Transient duration: {DURATION_S * 1000:.3f} ms")
        print(f"Oscillations per transient: {FUNDAMENTAL_HZ * DURATION_S:g}")
        print(f"Sample interval: {DURATION_S / WAVEFORM_SAMPLES * 1e6:.3f} µs")

    
        time_s, samples = generate_triangle(
            duration_s=DURATION_S,
            tau_s=DECAY_TIME_S,
            phase_rad=PHASE_RAD,
        )
        voltage_v = AMPLITUDE_V * samples

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        axes[0].plot(time_s * 1e3, voltage_v)
        axes[0].set(xlabel="Time (ms)", ylabel="Voltage (V)",
                    title="Full transient")

        mask = time_s < 50e-6
        axes[1].plot(time_s[mask] * 1e6, voltage_v[mask], ".-")
        axes[1].set(xlabel="Time (µs)", ylabel="Voltage (V)",
                    title="First 50 µs — waveform points shown")

        for ax in axes:
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    except KeyboardInterrupt:
        print("\nInterrupted.")
