import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ConfigTests(unittest.TestCase):
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
