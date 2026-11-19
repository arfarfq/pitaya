from pathlib import Path

import matplotlib.pyplot as plt

from src.config import load_config
from src.data_processor import DataProcessor
from src.start_signal import record_transient

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def analyze(processor, config):
    """Apply all detection and calibration settings, in frequency space first."""
    # Compute and cache the spectrum before measuring peaks or harmonic families.
    processor.dft()
    processor.detect_peaks(**config["peaks"])
    processor.detect_harmonics(**config["harmonics"])
    # Calibrate after detection so peak grouping stays in frequency space.
    calibration = dict(config["calibration"])
    if calibration.pop("enabled"):
        processor.frequency_to_mz(**calibration)
    else:
        processor.estimated_mz = processor.mz_axis = processor.mz_mask = None
        processor.calibration = None
    return processor


def display(processor, config):
    options = dict(config["plot"])
    output = options.pop("save")
    dpi = options.pop("dpi")
    show = options.pop("show")
    figure, axes = processor.plot(**options, show=False)
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure, axes


def main():
    # Load the settings.
    config = load_config(CONFIG_PATH)

    if config["input"]["source"] == "acquisition":
        acquisition = config["acquisition"]
        processor = record_transient(
            acquisition["host"], acquisition["csv"],
            acquisition=acquisition, processor_options=config["fft"],
        )
    else:
        processor = DataProcessor.load_data(config["input"]["file"], **config["fft"])

    analyze(processor, config)
    display(processor, config)


if __name__ == "__main__":
    main()
