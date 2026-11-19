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
    def test_configured_acquisition_and_fft_options(self):
        sock = FakeSocket("16777216\n4096\nWAIT\n1\n0\n{0,1,0,-1,0,1,0,-1}\n"
                          "{0,1,0,-1,0,1,0,-1}\n")
        with (patch.object(start_signal.socket, "create_connection", return_value=sock) as connect,
              patch.object(start_signal.time, "sleep")):
            processor = start_signal.record_transient("configured-board", acquisition={
                "port": 5001, "adc_rate_hz": 64000000, "decimation": 64,
                "capture_s": 16e-6, "trigger_level_v": 0.4, "block_samples": 8,
                "pulse_samples": 16, "pulse_low_samples": 4, "pulse_high_samples": 8,
                "pulse_frequency_hz": 100000, "pulse_amplitude_v": 0.5,
            }, processor_options={"padding_factor": 4, "window": "boxcar"})
        connect.assert_called_once_with(("configured-board", 5001), timeout=5)
        self.assertIn("ACQ:TRIG:LEV 0.4", sock.commands)
        self.assertIn("SOUR1:FREQ:FIX 100000", sock.commands)
        self.assertIn("SOUR1:VOLT 0.5", sock.commands)
        self.assertIn("ACQ:AXI:SOUR1:DATA:START:N? 8,8", sock.commands)
        self.assertEqual(processor.padding_factor, 4)
        self.assertEqual(len(processor.dft()[0]), 33)

    def capture(self, sock, csv_path=None):
        with (
            patch.object(start_signal.socket, "create_connection", return_value=sock),
            patch.object(start_signal.time, "sleep"),
            patch.object(start_signal, "ADC_RATE_HZ", 64_000_000),
            patch.object(start_signal, "CAPTURE_S", 16e-6),
        ):
            return start_signal.record_transient("test-board", csv_path)

    def test_capture_returns_raw_processor_with_correct_units_and_csv_roundtrip(self):
        volts = 0.2 * np.sin(2 * np.pi * np.arange(16) / 8)
        data = "{" + ",".join(map(str, volts)) + "}"
        sock = FakeSocket(f"16777216\n4096\nWAIT\n1\n12\n{data}\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.csv"
            processor = self.capture(sock, path)
            reloaded = DataProcessor.load_data(path)
        np.testing.assert_allclose(processor.x, np.arange(16) * 1e-6)
        np.testing.assert_allclose(processor.y, volts)
        np.testing.assert_allclose(reloaded.x, processor.x)
        np.testing.assert_allclose(reloaded.y, processor.y, atol=1e-12)
        self.assertAlmostEqual(processor.dt_seconds, 1e-6)
        self.assertIsNone(processor.dft_freq)
        self.assertIsNone(processor.dft_mag)
        processor.dft()
        self.assertEqual(processor.dft_freq[np.argmax(processor.dft_mag)], 125)
        self.assertIn("ACQ:AXI:SOUR1:DATA:START:N? 12,16", sock.commands)
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
        np.testing.assert_allclose(processor.x, [0, 1e-6])
        np.testing.assert_allclose(processor.y, [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
