from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

import main


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.config = deepcopy(main.DEFAULTS)

    def run_signal(self, rng=None):
        if rng is None:
            rng = np.random.default_rng(self.config["noise"]["seed"])
        return main.run(self.config, rng)

    def test_dummy_uses_settings_and_skips_all_hardware(self):
        self.config["mode"]["dummy"] = True
        self.config["noise"]["enabled"] = False
        self.config["waveform"].update(fundamental_hz=50_000, phase_rad=np.pi / 2, amplitude_v=0.4)
        with (
            patch.object(main, "upload_signal") as upload,
            patch.object(main, "capture_signal") as capture,
            patch("socket.create_connection", side_effect=AssertionError("Unexpected network")),
        ):
            times, volts = self.run_signal()
        self.assertEqual(times.size, 16384)
        self.assertAlmostEqual(volts[0], 0.4)
        expected = 0.4 * np.exp(-times / self.config["waveform"]["decay_time_s"]) * (2 / np.pi) * np.arcsin(
            np.sin(2 * np.pi * 50_000 * times + np.pi / 2))
        np.testing.assert_allclose(volts, expected, atol=1e-12)
        upload.assert_not_called()
        capture.assert_not_called()

    def test_hardware_forwards_connection_and_acquisition_settings(self):
        self.config["mode"]["dummy"] = False
        self.config["connection"].update(host="test-board", port=5001, socket_timeout_s=7)
        self.config["acquisition"].update(decimation=128, pre_trigger_s=0.002)
        samples = np.zeros(16384)
        capture = (np.arange(3), np.zeros(3))
        order = []
        with (
            patch.object(main, "prepare_signal", return_value=(None, samples)),
            patch.object(main, "upload_signal", side_effect=lambda *a, **kw: order.append("upload")) as upload,
            patch.object(main, "capture_signal", side_effect=lambda **kw: order.append("capture") or capture) as receive,
        ):
            self.assertIs(self.run_signal(), capture)
        self.assertEqual(order, ["upload", "capture"])
        for call in (upload.call_args, receive.call_args):
            self.assertEqual(call.kwargs["host"], "test-board")
            self.assertEqual(call.kwargs["port"], 5001)
            self.assertEqual(call.kwargs["socket_timeout_s"], 7)
        self.assertEqual(receive.call_args.kwargs["decimation"], 128)
        self.assertEqual(receive.call_args.kwargs["pre_trigger_s"], 0.002)

    def test_reusing_generator_produces_fresh_reproducible_noise(self):
        self.config["mode"]["dummy"] = True
        self.config["noise"]["enabled"] = True
        rng = np.random.default_rng(42)
        first = self.run_signal(rng)[1]
        second = self.run_signal(rng)[1]
        repeated = self.run_signal(np.random.default_rng(42))[1]
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, second))

    def test_protocol_duration_is_checked_before_hardware(self):
        self.config["waveform"]["duration_s"] = 0.1
        with patch.object(main, "upload_signal") as upload:
            with self.assertRaises(ValueError):
                self.run_signal()
        upload.assert_not_called()

    def test_main_uses_output_and_plot_settings(self):
        self.config["output"].update(save_data=True, directory="custom-output")
        self.config["plot"].update(enabled=True, linewidth=1.2)
        with (
            patch.object(main, "run", return_value=(np.arange(3), np.zeros(3))),
            patch.object(main, "save_signal") as save,
            patch.object(main, "plot_signal") as plot,
        ):
            main.main(self.config)
        save.assert_called_once()
        self.assertEqual(save.call_args.args[2].parent, main.CONFIG_PATH.parent / "custom-output")
        self.assertEqual(plot.call_args.kwargs["linewidth"], 1.2)

    def test_debug_duration_frequency_and_decay_scaling(self):
        self.config["mode"]["dummy"] = True
        self.config["waveform"]["one_second"] = True
        self.config["noise"]["enabled"] = False
        times, volts = self.run_signal()
        self.assertGreater(times[-1], 0.99)
        self.assertLess(times[-1], 1)
        expected = self.config["waveform"]["amplitude_v"] * np.exp(-times / 0.3) * (2 / np.pi) * np.arcsin(
            np.sin(2 * np.pi * 1050 * times))
        np.testing.assert_allclose(volts, expected)
