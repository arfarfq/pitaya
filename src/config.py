"""YAML configuration shared by analysis and acquisition commands."""

from copy import deepcopy
from pathlib import Path

import yaml

# YAML overrides these defaults; omitted settings keep their values here.
DEFAULTS = {
    # Choose a saved waveform or a new board capture.
    "input": {
        "source": "file",
        "file": "data/k_20ms_avg128_26.5degC.csv",
    },
    # Windowing and transform settings; frequencies are expressed in kHz.
    "fft": {
        "window": "blackmanharris",
        "window_parameter": None,
        "window_symmetric": True,
        "detrend": "constant",
        "padding_factor": 8,
        "fft_length": None,
        "cutoff_khz": 500,
        "sampling_rtol": 0.01,
    },
    # Noise thresholds and integration regions for detected FFT peaks.
    "peaks": {
        "min_distance_khz": 0.5,
        "prominence_sigma": 2,
        "height_sigma": 5,
        "width_rel_height": 0.5,
        "integration_width_factor": 2,
    },
    # Match integer frequency multiples within a relative tolerance.
    "harmonics": {
        "max_harmonic": 10,
        "tolerance": 0.01,
    },
    # Reference values for converting frequency to mass-to-charge (Th).
    "calibration": {
        "enabled": True,
        "ref_khz": 105.912,
        "ref_mz": 38.9637065,
        "harmonic_order": 1,
        "ref_harmonic_order": 1,
        "offset_khz": 0.0,
        "mz_range": [1, 200],
        "calibration_label": "Unconfirmed reference",
    },
    # Display limits and optional figure export; None leaves limits unset.
    "plot": {
        "mz_limits": [1, 200],
        "frequency_limits_khz": None,
        "clean_mz": True,
        "show_harmonics": True,
        "show": True,
        "save": None,
        "dpi": 150,
    },
    # Board connection, capture timing, and excitation pulse settings.
    "acquisition": {
        "host": "rp-xxxxxx.local",
        "port": 5000,
        "adc_rate_hz": 125000000,
        "decimation": 64,
        "capture_s": 0.1,
        "trigger_level_v": 0.25,
        "socket_timeout_s": 5,
        "capture_timeout_s": 3,
        "arm_delay_s": 0.02,
        "poll_interval_s": 0.01,
        "block_samples": 8192,
        "pulse_samples": 16384,
        "pulse_low_samples": 4096,
        "pulse_high_samples": 8192,
        "pulse_frequency_hz": 125000,
        "pulse_amplitude_v": 0.8,
        "pulse_offset_v": 0.0,
        "csv": None,
    },
}


def load_config(path):
    """Load an explicit YAML path, merging omitted settings with defaults."""
    # Copy nested sections so loading one file cannot change future defaults.
    config = deepcopy(DEFAULTS)
    path = Path(path).resolve()
    base = path.parent
    with path.open(encoding="utf-8") as file:
        supplied = yaml.safe_load(file)
    if supplied is None:
        supplied = {}
    # Apply only the supplied settings, preserving defaults for missing keys.
    for section, settings in supplied.items():
        for key, value in settings.items():
            config[section][key] = value
    # Resolve input and output paths relative to the YAML file, not the shell.
    for section, key in (("input", "file"), ("plot", "save"), ("acquisition", "csv")):
        if config[section][key] is not None:
            config[section][key] = str((base / config[section][key]).resolve())
    return config
