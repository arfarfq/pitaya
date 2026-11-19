import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import main as workflow
import yaml


class MainTests(unittest.TestCase):
    def test_analyze_and_save_without_display(self):
        with tempfile.TemporaryDirectory() as directory:
            waveform = Path(directory) / "capture.csv"
            output = Path(directory) / "spectrum.png"
            timestamps = np.arange(1024) / 1e6
            np.savetxt(waveform, np.column_stack((timestamps, np.sin(2 * np.pi * 125e3 * timestamps))),
                       delimiter=",", header="time_s,voltage_v", comments="")
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(yaml.safe_dump({
                "input": {"file": str(waveform)},
                "plot": {"show": False, "save": str(output)},
            }))
            with (patch.object(plt, "show") as show,
                  patch.object(workflow, "CONFIG_PATH", config_path)):
                workflow.main()
            show.assert_not_called()
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(plt.get_fignums(), [])

    def test_acquisition_settings_are_read_from_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(yaml.safe_dump({
                "input": {"source": "acquisition"},
                "acquisition": {"host": "test-board", "port": 5001, "csv": "raw.csv"},
                "fft": {"window": "hann"},
            }))
            with (patch.object(workflow, "CONFIG_PATH", config_path),
                  patch.object(workflow, "record_transient") as capture,
                  patch.object(workflow, "analyze") as analyze,
                  patch.object(workflow, "display") as display,
                  patch.object(workflow.DataProcessor, "load_data") as load):
                workflow.main()
            load.assert_not_called()
            args, kwargs = capture.call_args
            self.assertEqual(args, ("test-board", str((Path(directory) / "raw.csv").resolve())))
            self.assertEqual(kwargs["acquisition"]["port"], 5001)
            self.assertEqual(kwargs["processor_options"]["window"], "hann")
            analyze.assert_called_once()
            display.assert_called_once()
