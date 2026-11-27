"""Acquire one playback of an already uploaded OUT1 waveform on IN1.

Protocol: docs/Protocolo Red Pitaya.pdf, sections 6 and 7.
Connect OUT1 directly to IN1 and start the board's SCPI server. main.py uploads
the waveform through start_test.py before calling this receiver. It then replays
that stored waveform; it cannot recover a transient that already finished.
The uploaded generator must remain configured for one internally triggered
burst. Its waveform, frequency, amplitude and burst settings are preserved.

For the coordinated workflow, edit the globals in main.py and run python main.py.
All experiment settings are defined in main.py and passed into these helpers.
Uses Deep Memory Acquisition to retain the full transient.
Requires a board/FPGA image with ACQ:AXI support (125 MHz ADC assumed below).
"""

import math
from pathlib import Path
import socket
import time

import numpy as np
import matplotlib.pyplot as plt


def capture_signal(host, *, duration_s, port, adc_rate_hz, decimation,
                   pre_trigger_s, post_trigger_s, socket_timeout_s, arm_delay_s,
                   capture_timeout_s, poll_interval_s, read_block_samples):
    """Return (time_s, voltage_v), with time zero at the AWG trigger.

    Capture T + 2 ms, including 1 ms before the trigger. Only acquisition and
    playback are performed; processing, plotting and saving are left for later.
    This function uses hardware. main.py bypasses it completely in dummy mode.
    """
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if not host:
        raise ValueError("host is required for hardware acquisition")

    if not all(math.isfinite(value) and value >= 0 for value in (pre_trigger_s, post_trigger_s)):
        raise ValueError("trigger margins must be finite and nonnegative")
    if adc_rate_hz <= 0 or decimation <= 0 or read_block_samples <= 0:
        raise ValueError("sampling rate, decimation and read block size must be positive")
    fs_hz = adc_rate_hz / decimation
    pre_samples = math.ceil(pre_trigger_s * fs_hz)
    sample_count = pre_samples + math.ceil((duration_s + post_trigger_s) * fs_hz)
    time_s = (np.arange(sample_count) - pre_samples) / fs_hz

    # Align the DMA buffer to 64 bytes, with extra samples trimmed after reading.
    buffer_samples = 32 * math.ceil(sample_count / 32)
    buffer_bytes = 2 * buffer_samples  # DMA stores each ADC sample as int16.

    with socket.create_connection((host, port), timeout=socket_timeout_s) as sock:
        with sock.makefile("rb") as replies:
            def send(command):
                sock.sendall((command + "\r\n").encode("ascii"))

            def query(command):
                send(command)
                reply = replies.readline()
                if not reply:
                    raise ConnectionError("Red Pitaya closed the connection")
                return reply.decode("ascii").strip()

            try:
                send("OUTPUT1:STATE OFF")
                send("ACQ:RST")
                address = int(query("ACQ:AXI:START?"))
                available_bytes = int(query("ACQ:AXI:SIZE?"))
                if buffer_bytes > available_bytes:
                    raise RuntimeError("Insufficient Deep Memory for this capture")
                send(f"ACQ:AXI:DEC {decimation}")
                send("ACQ:DATA:FORMAT ASCII")
                send("ACQ:AXI:DATA:UNITS VOLTS")
                # SET:Buffer takes bytes; trigger delay and read positions use samples.
                send(f"ACQ:AXI:SOUR1:SET:Buffer {address},{buffer_bytes}")
                send(f"ACQ:AXI:SOUR1:TRIG:DLY {buffer_samples - pre_samples}")
                send("ACQ:AXI:SOUR1:ENable ON")

                # Start sampling and fill the pre-trigger history.
                send("ACQ:START")
                time.sleep(max(arm_delay_s, pre_samples / fs_hz))
                # Arm IN1 to wait for the generator's internal trigger.
                send("ACQ:TRIG AWG_PE")
                if query("ACQ:TRIG:STAT?") != "WAIT":
                    raise RuntimeError("Acquisition did not arm on AWG_PE")
                send("OUTPUT1:STATE ON")
                # Play the uploaded transient; this also triggers IN1 acquisition.
                send("SOUR1:TRig:INT")

                deadline = time.monotonic() + buffer_samples / fs_hz + capture_timeout_s
                while query("ACQ:AXI:SOUR1:TRIG:FILL?") != "1":
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Capture timed out waiting for AWG playback")
                    time.sleep(poll_interval_s)
                send("ACQ:STOP")

                # The sample after the final write is the oldest in the ring.
                write_pos = int(query("ACQ:AXI:SOUR1:WRITE:POS?"))
                position = (write_pos + 1) % buffer_samples
                voltage_v = np.empty(sample_count)
                offset = 0
                while offset < sample_count:
                    count = min(read_block_samples, sample_count - offset, buffer_samples - position)
                    reply = query(f"ACQ:AXI:SOUR1:DATA:START:N? {position},{count}")
                    values = np.fromstring(reply.strip("{}"), sep=",")
                    if values.size != count or not np.isfinite(values).all():
                        raise RuntimeError("Red Pitaya returned an invalid data block")
                    voltage_v[offset:offset + count] = values
                    offset += count
                    position = (position + count) % buffer_samples
            finally:
                send("ACQ:STOP")
                send("ACQ:AXI:SOUR1:ENable OFF")
                send("OUTPUT1:STATE OFF")

    return time_s, voltage_v


def process_signal(time_s, voltage_v):
    """Placeholder for later signal processing."""
    pass


def plot_signal(time_s, voltage_v, *, title, figsize, linewidth, grid_alpha, show=True):
    """Plot voltage against time in milliseconds; return the figure and axes."""
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.plot(np.asarray(time_s) * 1000, voltage_v, linewidth=linewidth)
    ax.set(title=title, xlabel="Time (ms)", ylabel="Voltage (V)")
    ax.grid(True, alpha=grid_alpha)
    if show:
        plt.show()
    return fig, ax


def save_signal(time_s, voltage_v, path):
    """Save raw time/voltage samples as CSV, in seconds and volts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack((time_s, voltage_v)), delimiter=",",
               header="time_s,voltage_v", comments="")
    return path
