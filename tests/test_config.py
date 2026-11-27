import os
from copy import deepcopy
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.config import CONFIG_PATH, DEFAULTS


class ConfigTests(unittest.TestCase):
    def test_main_uses_loaded_settings_and_debug_scaling(self):
        config = deepcopy(DEFAULTS)
        config["mode"]["dummy"] = True
        config["waveform"]["one_second"] = True
        config["noise"]["enabled"] = False
        config["connection"]["host"] = "test-board"
        config["acquisition"]["decimation"] = 128
        config["output"]["directory"] = "custom-output"
        config["plot"]["linewidth"] = 1.2
        with (
            patch("src.config.DEFAULTS", config),
            patch("socket.create_connection", side_effect=AssertionError("Unexpected network")),
        ):
            namespace = runpy.run_path(str(CONFIG_PATH.parent / "main.py"))
            import numpy as np
            times, volts = namespace["run"](config, np.random.default_rng(config["noise"]["seed"]))
        self.assertIs(namespace["DEFAULTS"], config)
        self.assertNotIn("DURATION_S", namespace)
        self.assertEqual(len(volts), 16384)
        self.assertLess(times[-1], 1.0)
        self.assertGreater(times[-1], 0.99)

    def test_processor_loads_config_outside_repository(self):
        root = Path(__file__).resolve().parent.parent
        env = {**os.environ, "PYTHONPATH": str(root), "MPLBACKEND": "Agg"}
        with tempfile.TemporaryDirectory() as directory:
            env["MPLCONFIGDIR"] = directory
            result = subprocess.run(
                [sys.executable, "-c", """
import numpy as np
from src.config import CONFIG_PATH, DEFAULTS
from src.data_processor import DataProcessor

assert CONFIG_PATH.is_file()
processor = DataProcessor(np.arange(16) * 1e-6, np.ones(16))
assert processor.padding_factor == DEFAULTS['fft']['padding_factor']
assert processor.cutoff_khz == DEFAULTS['fft']['cutoff_khz']
frequencies, amplitudes = processor.dft()
assert len(frequencies) > 0
assert np.all(amplitudes == 0)
"""],
                cwd=directory, env=env, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
