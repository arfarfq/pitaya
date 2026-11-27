import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src import start_signal
from src.data_processor import DataProcessor


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


class CaptureTests(unittest.TestCase):
    def capture(self, sock, csv_path=None):
        with (
            patch.object(start_signal.socket, "create_connection", return_value=sock),
            patch.object(start_signal.time, "sleep"),
            patch.object(start_signal, "ADC_RATE_HZ", 64_000_000),
        ):
            return start_signal.record_transient(
                "test-board", csv_path, duration_s=16e-6,
                pre_trigger_s=0, post_trigger_s=0,
            )

    def test_capture_returns_fft_with_correct_units_and_csv_roundtrip(self):
        volts = 0.2 * np.sin(2 * np.pi * np.arange(16) / 8)
        data = "{" + ",".join(map(str, volts)) + "}"
        sock = FakeSocket(f"16777216\n4096\nWAIT\n1\n0\n{data}\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.csv"
            processor = self.capture(sock, path)
            reloaded = DataProcessor.load_data(path)
        np.testing.assert_allclose(processor.x, np.arange(16))
        np.testing.assert_allclose(processor.y, volts * 1000)
        np.testing.assert_allclose(reloaded.x, processor.x)
        np.testing.assert_allclose(reloaded.y, processor.y, atol=1e-12)
        self.assertAlmostEqual(processor.dt_seconds, 1e-6)
        self.assertEqual(processor.dft_freq[np.argmax(processor.dft_mag)], 125_000)
        self.assertIn("ACQ:AXI:SOUR1:DATA:START:N? 0,16", sock.commands)
        self.assertEqual(sock.commands[-3:], [
            "ACQ:STOP", "ACQ:AXI:SOUR1:ENable OFF", "OUTPUT1:STATE OFF",
        ])

    def test_incomplete_capture_disables_output_and_dma(self):
        sock = FakeSocket("16777216\n4096\nWAIT\n1\n0\n{0,1}\n")
        with self.assertRaisesRegex(RuntimeError, "incomplete data block"):
            self.capture(sock)
        self.assertEqual(sock.commands[-2:], [
            "ACQ:AXI:SOUR1:ENable OFF", "OUTPUT1:STATE OFF",
        ])

    def test_missing_trigger_times_out_and_disables_output(self):
        sock = FakeSocket("16777216\n4096\nWAIT\n0\n")
        with patch.object(start_signal.time, "monotonic", side_effect=[0, 4]):
            with self.assertRaisesRegex(TimeoutError, "Capture timed out"):
                self.capture(sock)
        self.assertEqual(sock.commands[-1], "OUTPUT1:STATE OFF")

    def test_legacy_scope_csv_still_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scope.csv"
            path.write_text("0,0,0,0,0.1\n0,0,0,0.000001,0.2\n")
            processor = DataProcessor.load_data(path)
        np.testing.assert_allclose(processor.x, [0, 1])
        np.testing.assert_allclose(processor.y, [100, 200])

    def test_pretrigger_margin_wrap_and_time_axis(self):
        # 4 samples before the trigger at index 2 => first read begins at 14.
        sock = FakeSocket("16777216\n4096\nWAIT\n1\n2\n{0,1}\n"
                          "{2,3,4,5,6,7,8,9,10,11,12,13,14,15}\n")
        with (
            patch.object(start_signal.socket, "create_connection", return_value=sock),
            patch.object(start_signal.time, "sleep"),
            patch.object(start_signal, "ADC_RATE_HZ", 64_000_000),
        ):
            processor = start_signal.record_transient(
                "test-board", duration_s=8e-6, pre_trigger_s=4e-6, post_trigger_s=4e-6,
            )
        np.testing.assert_allclose(processor.x, np.arange(-4, 12))
        np.testing.assert_allclose(processor.y, np.arange(16) * 1000)
        self.assertIn("ACQ:AXI:SOUR1:TRIG:DLY 12", sock.commands)
        self.assertIn("ACQ:AXI:SOUR1:DATA:START:N? 14,2", sock.commands)
        self.assertIn("ACQ:AXI:SOUR1:DATA:START:N? 0,14", sock.commands)

    def test_capture_duration_follows_generator_configuration(self):
        for duration in (0.010, 1.0):
            sock = FakeSocket("16777216\n16777216\nWAIT\n0\n")
            with (
                self.subTest(duration=duration),
                patch.dict(start_signal.settings.DEFAULTS["waveform"],
                           duration_s=duration, one_second=False),
                patch.object(start_signal.socket, "create_connection", return_value=sock),
                patch.object(start_signal.time, "sleep"),
                patch.object(start_signal.time, "monotonic", side_effect=[0, 10]),
            ):
                with self.assertRaises(TimeoutError):
                    start_signal.record_transient("test-board")
                fs = start_signal.ADC_RATE_HZ / start_signal.DECIMATION
                pre = int(np.ceil(start_signal.PRE_TRIGGER_S * fs))
                post = int(np.ceil((duration + start_signal.POST_TRIGGER_S) * fs))
                buffer_size = 8 * int(np.ceil((pre + post) / 8))
                self.assertIn(f"ACQ:AXI:SOUR1:SET:Buffer 16777216,{buffer_size}", sock.commands)
                self.assertIn(f"ACQ:AXI:SOUR1:TRIG:DLY {buffer_size - pre}", sock.commands)

    def test_invalid_capture_settings_do_not_connect(self):
        for kwargs in ({"duration_s": 0}, {"duration_s": float("nan")},
                       {"pre_trigger_s": -1}, {"post_trigger_s": float("inf")}):
            with self.subTest(kwargs=kwargs), patch.object(start_signal.socket, "create_connection") as connect:
                with self.assertRaises(ValueError):
                    start_signal.record_transient("test-board", **kwargs)
                connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
