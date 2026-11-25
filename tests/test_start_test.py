import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src import start_test


class WhiteNoiseTests(unittest.TestCase):
    def test_snr_statistics_and_independent_repetitions(self):
        _, clean = start_test.generate_triangle()
        signal_rms = np.sqrt(np.mean(clean ** 2))
        rng = np.random.default_rng(42)
        for snr_db in (20, 10, 0):
            with self.subTest(snr_db=snr_db):
                sigma = signal_rms / 10 ** (snr_db / 20)
                scale = 1 + 6 * sigma
                first = start_test.add_white_noise(clean, snr_db, rng=rng)
                second = start_test.add_white_noise(clean, snr_db, rng=rng)
                noise = first * scale - clean
                noise2 = second * scale - clean
                measured_snr = 20 * np.log10(signal_rms / np.sqrt(np.mean(noise ** 2)))
                self.assertAlmostEqual(measured_snr, snr_db, delta=0.25)
                self.assertLess(abs(noise.mean()), 0.03 * sigma)
                self.assertLess(abs(np.corrcoef(noise, noise2)[0, 1]), 0.03)
                self.assertLess(abs(np.corrcoef(noise[:-1], noise[1:])[0, 1]), 0.03)
                self.assertTrue(np.all(np.abs(first) <= 1))

    def test_seed_reproduces_sequence_without_mutating_input(self):
        clean = np.linspace(-1, 1, 16384)
        original = clean.copy()
        first = start_test.add_white_noise(clean, rng=np.random.default_rng(12))
        second = start_test.add_white_noise(clean, rng=np.random.default_rng(12))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(clean, original)

    def test_invalid_noise_parameters(self):
        for snr in (float("nan"), float("inf"), -10000, 10000):
            with self.subTest(snr=snr), self.assertRaises(ValueError):
                start_test.add_white_noise(np.ones(10), snr)

    def test_prepared_noise_is_uploaded_unchanged(self):
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                patch.object(start_test, "WHITE_NOISE_ENABLED", enabled),
                patch.object(start_test, "_noise_rng", np.random.default_rng(42)),
                patch.object(start_test.socket, "create_connection") as connect,
                patch.object(start_test.time, "sleep"),
            ):
                clean = start_test.generate_triangle()[1]
                samples = start_test.add_white_noise(clean) if enabled else clean.copy()
                with patch.object(start_test, "add_white_noise") as add_noise:
                    start_test.send_signal(samples, "test-board")
                    add_noise.assert_not_called()
                commands = connect.return_value.__enter__.return_value.sendall.call_args_list
                uploaded = np.fromstring(commands[2].args[0].decode().split(" ", 1)[1], sep=",")
                np.testing.assert_allclose(uploaded, samples, atol=5e-10)
                if enabled:
                    self.assertFalse(np.array_equal(samples, clean))
                else:
                    np.testing.assert_array_equal(samples, clean)


class TriangleTests(unittest.TestCase):
    def test_sample_grid_triangle_and_decay(self):
        for duration in (0.001, 0.010):
            with self.subTest(duration=duration):
                t, samples = start_test.generate_triangle(duration, 0.003)
                self.assertEqual(len(samples), 16384)
                np.testing.assert_allclose(t, np.arange(16384) * duration / 16384)
                self.assertLess(t[-1], duration)
                # Independent piecewise-linear triangle, starting at zero rising.
                cycles = (start_test.FUNDAMENTAL_HZ * t + 0.25) % 1
                expected = (1 - 4 * np.abs(cycles - 0.5)) * np.exp(-t / 0.003)
                np.testing.assert_allclose(samples, expected, atol=2e-12)
                self.assertTrue(np.all(np.abs(samples) <= 1))

    def test_fundamental_and_initial_phase(self):
        t, samples = start_test.generate_triangle(0.010, 1e6, np.pi / 2)
        self.assertAlmostEqual(samples[0], 1)
        frequencies = np.fft.rfftfreq(len(t), t[1] - t[0])
        self.assertAlmostEqual(frequencies[np.argmax(np.abs(np.fft.rfft(samples)))],
                               start_test.FUNDAMENTAL_HZ, delta=100)

    def test_invalid_parameters(self):
        for kwargs in ({"duration_s": 0.0009}, {"duration_s": 0.011},
                       {"duration_s": float("nan")}, {"tau_s": 0},
                       {"tau_s": float("inf")}, {"phase_rad": float("nan")}):
            with self.subTest(kwargs=kwargs):
                with patch.object(start_test.socket, "create_connection") as connect:
                    with self.assertRaises(ValueError):
                        start_test.generate_triangle(**kwargs)
                    connect.assert_not_called()

    def test_invalid_upload_rejected_before_connecting(self):
        clean = start_test.generate_triangle()[1]
        for samples, kwargs in (
            (clean, {"amplitude_v": 1.1}),
            (clean, {"amplitude_v": float("nan")}),
            (clean, {"duration_s": 0}),
            (clean, {"duration_s": float("inf")}),
            (np.zeros(10), {}),
            (np.full(16384, float("nan")), {}),
            (np.full(16384, 1.1), {}),
        ):
            with self.subTest(kwargs=kwargs), patch.object(start_test.socket, "create_connection") as connect:
                with self.assertRaises(ValueError):
                    start_test.send_signal(samples, "test-board", **kwargs)
                connect.assert_not_called()

    def test_debug_mode_preserves_stretched_waveform(self):
        with patch.object(start_test, "FACTOR", 100), patch.object(start_test, "FUNDAMENTAL_HZ", 1050):
            t, samples = start_test.generate_triangle(1.0, 0.3)
            cycles = (1050 * t + 0.25) % 1
            expected = (1 - 4 * np.abs(cycles - 0.5)) * np.exp(-t / 0.3)
            np.testing.assert_allclose(samples, expected, atol=2e-12)
            with self.assertRaises(ValueError):
                start_test.generate_triangle(0.010)

    def test_single_transient_upload_and_cleanup(self):
        sock = MagicMock()
        with (
            patch.object(start_test.socket, "create_connection") as connect,
            patch.object(start_test.time, "sleep") as sleep,
        ):
            connect.return_value.__enter__.return_value = sock
            samples = start_test.generate_triangle(0.002)[1]
            start_test.send_signal(samples, "test-board", 5001, duration_s=0.002)
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
                start_test.send_signal(start_test.generate_triangle()[1], "test-board")
        self.assertEqual(sock.sendall.call_args.args[0], b"OUTPUT1:STATE OFF\r\n")


if __name__ == "__main__":
    unittest.main()
