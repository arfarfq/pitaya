import math
import socket
import time

import numpy as np

HOST = "rp-xxxxxx.local"  # Replace with your Red Pitaya hostname or IP.
ADC_RATE_HZ = 125_000_000  # Check your board model.
DECIMATION = 64
CAPTURE_S = 0.100
TRIGGER_LEVEL_V = 0.25


def record_transient(host, csv_path):
    fs_hz = ADC_RATE_HZ / DECIMATION
    sample_count = math.ceil(CAPTURE_S * fs_hz)
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
            send(f"ACQ:AXI:SOUR1:TRIG:DLY {buffer_samples}")
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
                time.sleep(0.02)
                send("ACQ:TRIG CH2_PE")

                if query("ACQ:TRIG:STAT?") != "WAIT":
                    raise RuntimeError("Acquisition did not arm on IN2")

                send("SOUR1:TRig:INT")

                deadline = time.monotonic() + 3
                while query("ACQ:AXI:SOUR1:TRIG:FILL?") != "1":
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Capture timed out. Check the pulse on IN2.")
                    time.sleep(0.01)

                send("ACQ:STOP")
                trigger_pos = int(query("ACQ:AXI:SOUR1:TRIG:POS?"))

                voltage_v = np.empty(sample_count)
                for offset in range(0, sample_count, 8192):
                    position = (trigger_pos + offset) % buffer_samples
                    count = min(8192, sample_count - offset)
                    reply = query(f"ACQ:AXI:SOUR1:DATA:START:N? {position},{count}")
                    values = np.fromstring(reply.strip("{}"), sep=",")
                    if values.size != count:
                        raise RuntimeError("Red Pitaya returned an incomplete data block")
                    voltage_v[offset:offset + count] = values
            finally:
                send("ACQ:STOP")
                send("ACQ:AXI:SOUR1:ENable OFF")
                send("OUTPUT1:STATE OFF")

    time_s = np.arange(sample_count) / fs_hz
    np.savetxt(
        csv_path,
        np.column_stack((time_s, voltage_v)),
        delimiter=",",
        header="time_s,voltage_v",
        comments="",
    )
    print(f"Saved {sample_count:,} samples at {fs_hz:,.0f} Hz to {csv_path}")


