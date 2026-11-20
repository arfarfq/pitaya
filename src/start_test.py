import argparse
import socket
import time

HOST = "rp-xxxxxx.local"  # Replace with your Red Pitaya hostname or IP.
PORT = 5000

PULSE_HIGH_S = 3.0
AMPLITUDE_V = 0.8
OFFSET_V = 0.0

# The waveform is high for half its period.
WAVEFORM_PERIOD_S = 2 * PULSE_HIGH_S
WAVEFORM_FREQUENCY_HZ = 1 / WAVEFORM_PERIOD_S


def send_pulse(host, port=PORT):
    """Send one 3-second high pulse from Red Pitaya OUT1."""
    waveform = ",".join(
        ["0"] * 4096
        + ["1"] * 8192
        + ["0"] * 4096
    )

    with socket.create_connection((host, port), timeout=5) as sock:
        def send(command):
            sock.sendall((command + "\r\n").encode("ascii"))

        try:
            send("OUTPUT1:STATE OFF")

            send("SOUR1:FUNC ARBITRARY")
            send(f"SOUR1:TRAC:DATA:DATA {waveform}")
            send(f"SOUR1:FREQ:FIX {WAVEFORM_FREQUENCY_HZ:.12f}")
            send(f"SOUR1:VOLT {AMPLITUDE_V:g}")
            send(f"SOUR1:VOLT:OFFS {OFFSET_V:g}")

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
                f"Sending pulse on OUT1: "
                f"{PULSE_HIGH_S:g} seconds high."
            )

            # Keep the output enabled until the entire waveform finishes.
            time.sleep(WAVEFORM_PERIOD_S + 0.1)

        finally:
            send("OUTPUT1:STATE OFF")

    print("Pulse complete. OUT1 is off.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send one 3-second pulse from Red Pitaya OUT1."
    )
    parser.add_argument(
        "host",
        nargs="?",
        default=HOST,
        help="Red Pitaya hostname or IP address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help="SCPI port (default: 5000)",
    )
    args = parser.parse_args()

    try:
        send_pulse(args.host, args.port)
    except KeyboardInterrupt:
        print("\nInterrupted.")