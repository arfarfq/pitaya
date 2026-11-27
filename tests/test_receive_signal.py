import io
import unittest
from unittest.mock import patch

import numpy as np

from src import receive_signal


class FakeSocket:
    def __init__(self, replies):
        self.replies = io.BytesIO(replies.encode("ascii"))
        self.commands = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def makefile(self, mode):
        return self.replies

    def sendall(self, command):
        self.commands.append(command.decode("ascii").strip())


class ReceiveSignalTests(unittest.TestCase):
    def capture(self, sock):
        with (
            patch.object(receive_signal.socket, "create_connection", return_value=sock),
            patch.object(receive_signal.time, "sleep"),
            patch.object(receive_signal, "ADC_RATE_HZ", 64_000_000),
            patch.object(receive_signal, "PRE_TRIGGER_S", 4e-6),
            patch.object(receive_signal, "POST_TRIGGER_S", 4e-6),
        ):
            return receive_signal.receive_signal("test-board", duration_s=8e-6)

    def test_capture_wraps_in_time_order_and_preserves_uploaded_waveform(self):
        # Final write at 29 in a 32-sample ring: oldest sample is at 30.
        sock = FakeSocket("16777216\n4096\nWAIT\n0\n1\n29\n{0,1}\n"
                          "{2,3,4,5,6,7,8,9,10,11,12,13,14,15}\n")
        time_s, volts = self.capture(sock)
        np.testing.assert_allclose(time_s, np.arange(-4, 12) * 1e-6)
        np.testing.assert_array_equal(volts, np.arange(16))
        # SCPI buffer allocation is in bytes, unlike delay/read commands in samples.
        self.assertIn("ACQ:AXI:SOUR1:SET:Buffer 16777216,64", sock.commands)
        self.assertIn("ACQ:AXI:SOUR1:TRIG:DLY 28", sock.commands)
        self.assertIn("ACQ:AXI:SOUR1:DATA:START:N? 30,2", sock.commands)
        self.assertIn("ACQ:AXI:SOUR1:DATA:START:N? 0,14", sock.commands)
        self.assertLess(sock.commands.index("ACQ:START"),
                        sock.commands.index("ACQ:TRIG AWG_PE"))
        self.assertLess(sock.commands.index("ACQ:TRIG AWG_PE"),
                        sock.commands.index("SOUR1:TRig:INT"))
        self.assertEqual([c for c in sock.commands if c.startswith("SOUR")],
                         ["SOUR1:TRig:INT"])
        self.assertNotIn("GEN:RST", sock.commands)
        self.assert_cleanup(sock)

    def assert_cleanup(self, sock):
        self.assertEqual(sock.commands[-3:], [
            "ACQ:STOP", "ACQ:AXI:SOUR1:ENable OFF", "OUTPUT1:STATE OFF",
        ])

    def test_timeout_stops_acquisition_and_output(self):
        sock = FakeSocket("16777216\n4096\nWAIT\n0\n")
        with patch.object(receive_signal.time, "monotonic", side_effect=[0, 4]):
            with self.assertRaises(TimeoutError):
                self.capture(sock)
        self.assert_cleanup(sock)

    def test_failed_arming_does_not_trigger_playback(self):
        sock = FakeSocket("16777216\n4096\nTD\n")
        with self.assertRaisesRegex(RuntimeError, "did not arm"):
            self.capture(sock)
        self.assertNotIn("SOUR1:TRig:INT", sock.commands)
        self.assertNotIn("OUTPUT1:STATE ON", sock.commands)
        self.assert_cleanup(sock)

    def test_incomplete_download_stops_acquisition_and_output(self):
        sock = FakeSocket("16777216\n4096\nWAIT\n1\n31\n{0,1}\n")
        with self.assertRaisesRegex(RuntimeError, "invalid data block"):
            self.capture(sock)
        self.assert_cleanup(sock)

    def test_insufficient_memory_does_not_trigger_playback(self):
        sock = FakeSocket("16777216\n63\n")
        with self.assertRaisesRegex(RuntimeError, "Insufficient Deep Memory"):
            self.capture(sock)
        self.assertNotIn("SOUR1:TRig:INT", sock.commands)
        self.assert_cleanup(sock)

    def test_invalid_duration_does_not_connect(self):
        for duration in (0, -1, float("nan"), float("inf")):
            with self.subTest(duration=duration), patch.object(
                receive_signal.socket, "create_connection"
            ) as connect:
                with self.assertRaises(ValueError):
                    receive_signal.receive_signal("test-board", duration_s=duration)
                connect.assert_not_called()
