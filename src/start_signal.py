import math
import socket
import time

import numpy as np

from .data_processor import DataProcessor
from . import start_test

HOST = "rp-xxxxxx.local"  # Replace with your Red Pitaya hostname or IP.
ADC_RATE_HZ = 125_000_000  # Check your board model.
DECIMATION = 64
PRE_TRIGGER_S = 0.001
POST_TRIGGER_S = 0.001
TRIGGER_LEVEL_V = 0.25


def record_transient(host, csv_path=None, *, duration_s=None,
                     pre_trigger_s=PRE_TRIGGER_S, post_trigger_s=POST_TRIGGER_S):
    """Capture IN1 using the pulse on IN2 as trigger, then return its FFT.

    Analysis starts as soon as the complete capture has been downloaded.
    csv_path optionally archives the raw capture in seconds and volts.
    duration_s defaults to start_test.DURATION_S. The record includes the
    requested pre/post-trigger margins; time zero is the IN2 trigger.
    This standalone capture still generates its own excitation pulse on OUT1.
    """
    duration_s = start_test.DURATION_S if duration_s is None else duration_s
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if not all(math.isfinite(value) and value >= 0
               for value in (pre_trigger_s, post_trigger_s)):
        raise ValueError("trigger margins must be finite and nonnegative")
    fs_hz = ADC_RATE_HZ / DECIMATION
    pre_samples = math.ceil(pre_trigger_s * fs_hz)
    post_samples = math.ceil((duration_s + post_trigger_s) * fs_hz)
    sample_count = pre_samples + post_samples
    if sample_count < 2:
        raise ValueError("capture must contain at least two samples")
    buffer_samples = 8 * math.ceil(sample_count / 8)

    # One 8 µs waveform: 2 µs low, 4 µs high, 2 µs low.
    waveform = ",".join(["0"] * 4096 + ["1"] * 8192 + ["0"] * 4096)

    with socket.create_connection((host, 5000), timeout=5) as sock:
        with sock.makefile("rb") as replies:
            def send(command):
                sock.sendall((command + "\r\n").encode("ascii"))

            def query(command):
                send(command)
                reply = replies.readline()
                if not reply:
                    raise ConnectionError("Red Pitaya closed the connection")
                return reply.decode("ascii").strip()

            send("OUTPUT1:STATE OFF")
            send("ACQ:RST")
            address = int(query("ACQ:AXI:START?"))
            available_bytes = int(query("ACQ:AXI:SIZE?"))
            if 2 * buffer_samples > available_bytes:
                raise RuntimeError("Insufficient Deep Memory for this capture")

            send(f"ACQ:AXI:DEC {DECIMATION}")
            send("ACQ:DATA:FORMAT ASCII")
            send("ACQ:AXI:DATA:UNITS VOLTS")
            send(f"ACQ:AXI:SOUR1:SET:Buffer {address},{buffer_samples}")
            send(f"ACQ:AXI:SOUR1:TRIG:DLY {buffer_samples - pre_samples}")
            send(f"ACQ:TRIG:LEV {TRIGGER_LEVEL_V}")

            send("SOUR1:FUNC ARBITRARY")
            send(f"SOUR1:TRAC:DATA:DATA {waveform}")
            send("SOUR1:FREQ:FIX 125000")
            send("SOUR1:VOLT 0.8")
            send("SOUR1:VOLT:OFFS 0")
            send("SOUR1:BURS:STAT BURST")
            send("SOUR1:BURS:NCYC 1")
            send("SOUR1:BURS:NOR 1")
            send("SOUR1:INITValue 0")
            send("SOUR1:BURS:LASTValue 0")
            send("SOUR1:TRig:SOUR INT")

            try:
                send("OUTPUT1:STATE ON")
                send("ACQ:AXI:SOUR1:ENable ON")
                send("ACQ:START")
                time.sleep(max(0.02, pre_samples / fs_hz))
                send("ACQ:TRIG CH2_PE")

                if query("ACQ:TRIG:STAT?") != "WAIT":
                    raise RuntimeError("Acquisition did not arm on IN2")

                send("SOUR1:TRig:INT")

                deadline = time.monotonic() + max(3, buffer_samples / fs_hz + 1)
                while query("ACQ:AXI:SOUR1:TRIG:FILL?") != "1":
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Capture timed out. Check the pulse on IN2.")
                    time.sleep(0.01)

                send("ACQ:STOP")
                trigger_pos = int(query("ACQ:AXI:SOUR1:TRIG:POS?"))

                voltage_v = np.empty(sample_count)
                offset = 0
                while offset < sample_count:
                    position = (trigger_pos - pre_samples + offset) % buffer_samples
                    count = min(8192, sample_count - offset, buffer_samples - position)
                    reply = query(f"ACQ:AXI:SOUR1:DATA:START:N? {position},{count}")
                    values = np.fromstring(reply.strip("{}"), sep=",")
                    if values.size != count:
                        raise RuntimeError("Red Pitaya returned an incomplete data block")
                    voltage_v[offset:offset + count] = values
                    offset += count
            finally:
                send("ACQ:STOP")
                send("ACQ:AXI:SOUR1:ENable OFF")
                send("OUTPUT1:STATE OFF")

    time_s = (np.arange(sample_count) - pre_samples) / fs_hz
    processor = DataProcessor(time_s * 1e6, voltage_v * 1e3)
    processor.dft()

    if csv_path is not None:
        np.savetxt(
            csv_path,
            np.column_stack((time_s, voltage_v)),
            delimiter=",",
            header="time_s,voltage_v",
            comments="",
        )
        print(f"Saved {sample_count:,} samples at {fs_hz:,.0f} Hz to {csv_path}")
    return processor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Capture a transient and plot its FFT.")
    parser.add_argument("host", help="Red Pitaya hostname or IP address")
    parser.add_argument("--csv", help="Optional path to save the raw capture")
    parser.add_argument("--duration", type=float, default=None,
                        help="Transient duration in seconds (default: start_test.DURATION_S)")
    args = parser.parse_args()
    record_transient(args.host, args.csv, duration_s=args.duration).plot()


