"""Prepare a waveform locally, or upload it and acquire one hardware capture."""

from datetime import datetime

import numpy as np

from src.config import CONFIG_PATH, DEFAULTS
from src.receive_signal import capture_signal, save_signal, plot_signal
from src.start_test import prepare_signal, upload_signal


def run(config, rng):
    """Return generated data in dummy mode, or acquired data in hardware mode."""
    connection = config["connection"]
    waveform = config["waveform"]
    noise = config["noise"]
    acquisition = config["acquisition"]

    factor = waveform["debug_time_factor"] if waveform["one_second"] else 1
    duration_s = waveform["duration_s"] * factor
    min_duration_s = waveform["min_duration_s"] * factor
    max_duration_s = waveform["max_duration_s"] * factor
    if not min_duration_s <= duration_s <= max_duration_s:
        raise ValueError(f"duration_s must be between {min_duration_s:g} and {max_duration_s:g}")
    if waveform["samples"] != 16_384:
        raise ValueError("The standard hardware waveform buffer requires 16384 samples")

    time_s, samples = prepare_signal(
        duration_s, tau_s=waveform["decay_time_s"] * factor,
        phase_rad=waveform["phase_rad"], fundamental_hz=waveform["fundamental_hz"] / factor,
        waveform_samples=waveform["samples"], noise_enabled=noise["enabled"],
        snr_db=noise["snr_db"], headroom_sigma=noise["headroom_sigma"], rng=rng,
    )
    if config["mode"]["dummy"]:
        return time_s, waveform["amplitude_v"] * samples

    # Upload only: acquisition must be armed before playback starts.
    upload_signal(
        samples, host=connection["host"], port=connection["port"], duration_s=duration_s,
        amplitude_v=waveform["amplitude_v"], socket_timeout_s=connection["socket_timeout_s"],
    )
    # Keep arming, playback, capture and cleanup together.
    return capture_signal(
        host=connection["host"], port=connection["port"], duration_s=duration_s,
        adc_rate_hz=acquisition["adc_rate_hz"], decimation=acquisition["decimation"],
        pre_trigger_s=acquisition["pre_trigger_s"], post_trigger_s=acquisition["post_trigger_s"],
        socket_timeout_s=connection["socket_timeout_s"], arm_delay_s=acquisition["arm_delay_s"],
        capture_timeout_s=acquisition["capture_timeout_s"],
        poll_interval_s=acquisition["poll_interval_s"], read_block_samples=acquisition["read_block_samples"],
    )


def main(config=None):
    config = DEFAULTS if config is None else config
    rng = np.random.default_rng(config["noise"]["seed"])
    time_s, voltage_v = run(config, rng)
    dummy = config["mode"]["dummy"]
    label = "Generated waveform (dummy)" if dummy else "Hardware capture"
    print(f"{label}: {voltage_v.size:,} samples (seconds, volts).")

    if config["output"]["save_data"]:
        prefix = "generated" if dummy else "capture"
        filename = f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
        output_dir = CONFIG_PATH.parent / config["output"]["directory"]
        path = save_signal(time_s, voltage_v, output_dir / filename)
        print(f"Saved data to {path}")
    plot = config["plot"]
    if plot["enabled"]:
        plot_signal(time_s, voltage_v, title=label, figsize=tuple(plot["figsize"]),
                    linewidth=plot["linewidth"], grid_alpha=plot["grid_alpha"])
    return time_s, voltage_v


if __name__ == "__main__":
    main()
