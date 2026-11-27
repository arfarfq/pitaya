"""Prepare a waveform locally, or upload it and acquire one hardware capture."""

from datetime import datetime
from pathlib import Path

from src.receive_signal import receive_signal, save_signal, plot_signal
from src import start_test
from src.start_test import prepare_signal, send_signal


DUMMY_MODE = False  # Hardware: upload OUT1 and capture IN1. True prepares locally.
HOST = start_test.HOST  # Set your board hostname/IP in src/start_test.py.
PORT = start_test.PORT  # SCPI server port (normally 5000).
DURATION_S = start_test.DURATION_S
AMPLITUDE_V = start_test.AMPLITUDE_V
SAVE_DATA = True
PLOT_DATA = True
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def run(dummy=DUMMY_MODE, port=PORT, host=HOST, duration_s=DURATION_S, amplitude_v=AMPLITUDE_V):
    """Return generated data in dummy mode, or acquired data in hardware mode."""

    time_s, samples = prepare_signal(duration_s)
    if dummy:
        return time_s, amplitude_v * samples

    # Upload only: acquisition must be armed before the single playback starts.
    send_signal(samples, host=host, port=port, duration_s=duration_s,
                amplitude_v=amplitude_v, trigger=False)

    # The receiver arms IN1, triggers OUT1, waits, then reads the capture.
    return receive_signal(host=host, port=port, duration_s=duration_s)


def main():
    time_s, voltage_v = run(dummy=DUMMY_MODE, host=HOST, port=PORT,
                            duration_s=DURATION_S, amplitude_v=AMPLITUDE_V)
    label = "Generated waveform (dummy)" if DUMMY_MODE else "Hardware capture"
    print(f"{label}: {voltage_v.size:,} samples (seconds, volts).")
    if SAVE_DATA:
        prefix = "generated" if DUMMY_MODE else "capture"
        filename = f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
        path = save_signal(time_s, voltage_v, OUTPUT_DIR / filename)
        print(f"Saved data to {path}")
    if PLOT_DATA:
        plot_signal(time_s, voltage_v, title=label)
    return time_s, voltage_v


if __name__ == "__main__":
    main()
