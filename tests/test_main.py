import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import main
from src import start_test


class WorkflowTests(unittest.TestCase):
    def test_dummy_runs_both_debug_and_protocol_modes_without_network(self):
        for factor in (1, 100):
            with (
                self.subTest(factor=factor),
                patch.object(start_test, "FACTOR", factor),
                patch.object(start_test, "FUNDAMENTAL_HZ", 105_000 / factor),
                patch.object(start_test, "DURATION_S", 0.010 * factor),
                patch.object(start_test, "DECAY_TIME_S", 0.003 * factor),
                patch.object(start_test, "WHITE_NOISE_ENABLED", False),
                patch("socket.create_connection", side_effect=AssertionError("Network used")),
            ):
                times, volts = main.run(dummy=True, duration_s=start_test.DURATION_S)
                self.assertEqual(volts.size, start_test.WAVEFORM_SAMPLES)
                self.assertGreater(np.max(np.abs(volts)), 0.1)
                self.assertLessEqual(np.max(np.abs(volts)), start_test.AMPLITUDE_V)
                self.assertEqual(times[0], 0)
                self.assertLess(times[-1], start_test.DURATION_S)
                np.testing.assert_allclose(np.diff(times),
                                           start_test.DURATION_S / start_test.WAVEFORM_SAMPLES)

    def test_dummy_preserves_prepared_noise_and_applies_amplitude_once(self):
        samples = np.full(start_test.WAVEFORM_SAMPLES, 0.25)
        generated_times = np.arange(samples.size) * 1e-6
        with (
            patch.object(main, "prepare_signal", return_value=(generated_times, samples)) as prepare,
            patch.object(main, "send_signal") as send,
            patch.object(main, "receive_signal") as receive,
            patch("socket.create_connection", side_effect=AssertionError("Network used")),
        ):
            times, volts = main.run(dummy=True, amplitude_v=0.4)
        prepare.assert_called_once()
        self.assertIs(times, generated_times)
        np.testing.assert_array_equal(volts, 0.25 * 0.4)
        send.assert_not_called()
        receive.assert_not_called()

    def test_hardware_uploads_without_trigger_before_receiver(self):
        samples = np.zeros(start_test.WAVEFORM_SAMPLES)
        capture = (np.arange(3), np.zeros(3))
        calls = MagicMock()
        with (
            patch.object(main, "prepare_signal", return_value=(None, samples)) as prepare,
            patch.object(main, "send_signal") as upload,
            patch.object(main, "receive_signal", return_value=capture) as receive,
        ):
            calls.attach_mock(upload, "upload")
            calls.attach_mock(receive, "receive")
            self.assertIs(main.run(dummy=False, host="test-board", port=5001,
                                   duration_s=0.5, amplitude_v=0.4), capture)
        self.assertEqual([call[0] for call in calls.mock_calls], ["upload", "receive"])
        prepare.assert_called_once_with(0.5)
        self.assertFalse(upload.call_args.kwargs["trigger"])
        self.assertIs(upload.call_args.args[0], samples)
        self.assertEqual(upload.call_args.kwargs["amplitude_v"], 0.4)
        self.assertEqual(upload.call_args.kwargs["host"], "test-board")
        self.assertEqual(upload.call_args.kwargs["port"], 5001)
        receive.assert_called_once_with(host="test-board", port=5001, duration_s=0.5)

    def test_upload_only_does_not_start_playback_or_sleep(self):
        with (
            patch.object(start_test.socket, "create_connection") as connect,
            patch.object(start_test.time, "sleep") as sleep,
        ):
            start_test.send_signal(np.zeros(start_test.WAVEFORM_SAMPLES), "test-board", trigger=False)
        commands = [call.args[0].decode().strip() for call in
                    connect.return_value.__enter__.return_value.sendall.call_args_list]
        self.assertIn("SOUR1:BURS:NCYC 1", commands)
        self.assertNotIn("SOUR1:TRig:INT", commands)
        self.assertNotIn("OUTPUT1:STATE ON", commands)
        self.assertEqual(commands[-1], "OUTPUT1:STATE OFF")
        sleep.assert_not_called()

    def test_dummy_upload_still_rejects_invalid_waveforms_before_network(self):
        with patch("socket.create_connection") as connect:
            with self.assertRaises(ValueError):
                start_test.send_signal(np.full(start_test.WAVEFORM_SAMPLES, 2),
                                       "test-board", dummy=True)
        connect.assert_not_called()

    def test_main_uses_configured_mode(self):
        for dummy in (True, False):
            with (
                self.subTest(dummy=dummy),
                patch.object(main, "DUMMY_MODE", dummy),
                patch.object(main, "run", return_value=(np.arange(3), np.zeros(3))) as run,
                patch.object(main, "save_signal") as save,
                patch.object(main, "plot_signal") as plot,
            ):
                main.main()
            self.assertEqual(run.call_args.kwargs["dummy"], dummy)
            save.assert_called_once()
            plot.assert_called_once()
