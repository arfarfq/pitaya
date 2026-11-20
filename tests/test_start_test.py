import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src import start_test


class TriangleTests(unittest.TestCase):
    def test_sample_grid_triangle_and_decay(self):
        for duration in (0.001, 0.010):
            with self.subTest(duration=duration):
                t, samples = start_test.generate_triangle(duration, 0.003)
                self.assertEqual(len(samples), 16384)
                np.testing.assert_allclose(t, np.arange(16384) * duration / 16384)
                self.assertLess(t[-1], duration)
                # Independent piecewise-linear triangle, starting at zero rising.
                cycles = (105_000 * t + 0.25) % 1
                expected = (1 - 4 * np.abs(cycles - 0.5)) * np.exp(-t / 0.003)
                np.testing.assert_allclose(samples, expected, atol=2e-12)
                self.assertTrue(np.all(np.abs(samples) <= 1))

    def test_fundamental_and_initial_phase(self):
        t, samples = start_test.generate_triangle(0.010, 1e6, np.pi / 2)
        self.assertAlmostEqual(samples[0], 1)
        frequencies = np.fft.rfftfreq(len(t), t[1] - t[0])
        self.assertAlmostEqual(frequencies[np.argmax(np.abs(np.fft.rfft(samples)))],
                               105_000)

    def test_invalid_parameters(self):
        for kwargs in ({"duration_s": 0.0009}, {"duration_s": 0.011},
                       {"duration_s": float("nan")}, {"tau_s": 0},
                       {"tau_s": float("inf")}, {"phase_rad": float("nan")},
                       {"amplitude_v": 1.1}):
            with self.subTest(kwargs=kwargs):
                with patch.object(start_test.socket, "create_connection") as connect:
                    with self.assertRaises(ValueError):
                        start_test.send_triangle("test-board", **kwargs)
                    connect.assert_not_called()

    def test_single_transient_upload_and_cleanup(self):
        sock = MagicMock()
        with (
            patch.object(start_test.socket, "create_connection") as connect,
            patch.object(start_test.time, "sleep") as sleep,
        ):
            connect.return_value.__enter__.return_value = sock
            start_test.send_triangle("test-board", 5001, duration_s=0.002)
            connect.assert_called_once_with(("test-board", 5001), timeout=5)
            sleep.assert_called_once()
            self.assertAlmostEqual(sleep.call_args.args[0], 0.102)
        commands = [call.args[0].decode("ascii").strip()
                    for call in sock.sendall.call_args_list]
        uploaded = np.fromstring(commands[2].split(" ", 1)[1], sep=",")
        np.testing.assert_allclose(uploaded, start_test.generate_triangle(0.002)[1],
                                   atol=5e-10)
        for command in ("SOUR1:FREQ:FIX 500.000000000000", "SOUR1:VOLT 0.8",
                        "SOUR1:VOLT:OFFS 0", "SOUR1:BURS:NCYC 1",
                        "SOUR1:BURS:NOR 1", "SOUR1:INITValue 0",
                        "SOUR1:BURS:LASTValue 0"):
            self.assertIn(command, commands)
        self.assertEqual(commands[-3:], ["OUTPUT1:STATE ON", "SOUR1:TRig:INT",
                                        "OUTPUT1:STATE OFF"])

    def test_interrupt_disables_output(self):
        sock = MagicMock()
        with (
            patch.object(start_test.socket, "create_connection") as connect,
            patch.object(start_test.time, "sleep", side_effect=KeyboardInterrupt),
        ):
            connect.return_value.__enter__.return_value = sock
            with self.assertRaises(KeyboardInterrupt):
                start_test.send_triangle("test-board")
        self.assertEqual(sock.sendall.call_args.args[0], b"OUTPUT1:STATE OFF\r\n")


if __name__ == "__main__":
    unittest.main()
