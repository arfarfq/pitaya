from pathlib import Path

import matplotlib.pyplot as plt

from src.config import load_config
from src.data_processor import DataProcessor
from src.start_signal import record_transient

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def analyze(processor, config):
    """Analyze one capture and store results on its processor.

    Use a new processor for each input. Reanalyzing the same processor reuses
    its FFT but recomputes detections with the supplied settings.
    """
    # Calculate the FFT once. detect_peaks() and frequency_to_mz() also call
    # dft(), but those calls reuse this processor's stored spectrum.
    # This explicit call is optional: detect_peaks() can initialize it itself.
    processor.dft()
    # New detections clear previous harmonic groups and cleaning results.
    processor.detect_peaks(**config["peaks"])
    processor.detect_harmonics(**config["harmonics"])
      
    # Group peaks in frequency space before mapping to m/z. Copy settings so
    # removing 'enabled' does not change the config used for subsequent inputs.
    calibration = dict(config["calibration"])
    if calibration.pop("enabled"):
        processor.frequency_to_mz(**calibration)
    else:
        # Remove any previous mapping if this processor is analyzed again.
        processor.estimated_mz = processor.mz_axis = processor.mz_mask = None
        processor.calibration = None
    return processor


def display(processor, config):
    """Render results, save if requested, then show or close the figure."""
    # Keep export/display options separate without modifying the shared config.
    options = dict(config["plot"])
    output = options.pop("save")
    dpi = options.pop("dpi")
    show = options.pop("show")
    # Save before opening a window. The envelope uses an existing decay fit.
    figure, axes = processor.plot(**options, show=False)
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi)
    if show:
        plt.show()
    else:
        # Avoid accumulating open figures when processing multiple captures.
        plt.close(figure)
    return figure, axes


def main():
    # Load the settings.
    config = load_config(CONFIG_PATH)

    # Both input paths create a fresh processor with no results from past inputs.
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
