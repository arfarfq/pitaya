from contextlib import ExitStack
import math
import socket
import time

import numpy as np

from .data_processor import DataProcessor
from .config import DEFAULTS

ADC_RATE_HZ = 125_000_000  # Check your board model.
DECIMATION = 64
CAPTURE_S = 0.100
TRIGGER_LEVEL_V = 0.25


def record_transient(host, csv_path=None, *, acquisition=None, processor_options=None):
    """Capture IN1 using the pulse on IN2 as trigger, then return a processor.

    The caller runs the FFT and analysis after the capture has been downloaded.
    csv_path optionally archives the raw capture in seconds and volts.
    """
    from copy import deepcopy

    settings = deepcopy(DEFAULTS)
    settings["acquisition"].update(adc_rate_hz=ADC_RATE_HZ, decimation=DECIMATION,
                                   capture_s=CAPTURE_S, trigger_level_v=TRIGGER_LEVEL_V)
    if acquisition is not None:
        settings["acquisition"].update(acquisition)
    if processor_options is not None:
        settings["fft"].update(processor_options)
    acquisition = settings["acquisition"]
    fs_hz = acquisition["adc_rate_hz"] / acquisition["decimation"]
    sample_count = math.ceil(acquisition["capture_s"] * fs_hz)
    buffer_samples = 8 * math.ceil(sample_count / 8)

    low = acquisition["pulse_low_samples"]
    high = acquisition["pulse_high_samples"]
    waveform = ",".join(["0"] * low + ["1"] * high
                        + ["0"] * (acquisition["pulse_samples"] - low - high))

    with socket.create_connection((host, acquisition["port"]),
                                  timeout=acquisition["socket_timeout_s"]) as sock:
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

            send(f"ACQ:AXI:DEC {acquisition['decimation']}")
            send("ACQ:DATA:FORMAT ASCII")
            send("ACQ:AXI:DATA:UNITS VOLTS")
            send(f"ACQ:AXI:SOUR1:SET:Buffer {address},{buffer_samples}")
            send(f"ACQ:AXI:SOUR1:TRIG:DLY {buffer_samples}")
            send(f"ACQ:TRIG:LEV {acquisition['trigger_level_v']}")

            send("SOUR1:FUNC ARBITRARY")
            send(f"SOUR1:TRAC:DATA:DATA {waveform}")
            send(f"SOUR1:FREQ:FIX {acquisition['pulse_frequency_hz']:g}")
            send(f"SOUR1:VOLT {acquisition['pulse_amplitude_v']:g}")
            send(f"SOUR1:VOLT:OFFS {acquisition['pulse_offset_v']:g}")
            send("SOUR1:BURS:STAT BURST")
            send("SOUR1:BURS:NCYC 1")
            send("SOUR1:BURS:NOR 1")
            send("SOUR1:INITValue 0")
            send("SOUR1:BURS:LASTValue 0")
            send("SOUR1:TRig:SOUR INT")

            with ExitStack() as cleanup:
                cleanup.callback(send, "OUTPUT1:STATE OFF")
                cleanup.callback(send, "ACQ:AXI:SOUR1:ENable OFF")
                cleanup.callback(send, "ACQ:STOP")
                send("OUTPUT1:STATE ON")
                send("ACQ:AXI:SOUR1:ENable ON")
                send("ACQ:START")
                time.sleep(acquisition["arm_delay_s"])
                send("ACQ:TRIG CH2_PE")

                if query("ACQ:TRIG:STAT?") != "WAIT":
                    raise RuntimeError("Acquisition did not arm on IN2")

                send("SOUR1:TRig:INT")

                deadline = time.monotonic() + acquisition["capture_timeout_s"]
                while query("ACQ:AXI:SOUR1:TRIG:FILL?") != "1":
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Capture timed out. Check the pulse on IN2.")
                    time.sleep(acquisition["poll_interval_s"])

                send("ACQ:STOP")
                trigger_pos = int(query("ACQ:AXI:SOUR1:TRIG:POS?"))

                voltage_v = np.empty(sample_count)
                for offset in range(0, sample_count, acquisition["block_samples"]):
                    position = (trigger_pos + offset) % buffer_samples
                    count = min(acquisition["block_samples"], sample_count - offset)
                    reply = query(f"ACQ:AXI:SOUR1:DATA:START:N? {position},{count}")
                    values = np.fromstring(reply.strip("{}"), sep=",")
                    if values.size != count:
                        raise RuntimeError("Red Pitaya returned an incomplete data block")
                    voltage_v[offset:offset + count] = values

    time_s = np.arange(sample_count) / fs_hz
    processor = DataProcessor(time_s, voltage_v, **settings["fft"])

    if csv_path is not None:
        from pathlib import Path

        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            csv_path,
            np.column_stack((time_s, voltage_v)),
            delimiter=",",
            header="time_s,voltage_v",
            comments="",
        )
        print(f"Saved {sample_count:,} samples at {fs_hz:,.0f} Hz to {csv_path}")
    return processor
