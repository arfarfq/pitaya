from src.data_processor import DataProcessor
from pathlib import Path
import pandas as pd

FILE_PATH = "../data/k_20ms_avg128_26.5degC.csv"

def main():
    waveform_path = Path(FILE_PATH)
    if not waveform_path.is_file():
        waveform_path = Path("data") / waveform_path.name

    processor = DataProcessor.load_data(waveform_path)

    # Optional: select a settled interval in µs before calculating the FFT.
    # use = (processor.x >= 0) & (processor.x < 10_000)
    # processor = DataProcessor(processor.x[use], processor.y[use])

    processor.frequency_to_mz()
    peaks = processor.detect_peaks()

    if peaks:
        summary = pd.DataFrame(peaks)[
            ["freq", "mass", "height", "width_kHz", "resolution_mass"]
        ].rename(columns={
            "freq": "Frequency (kHz)",
            "mass": "m/z (Th)",
            "height": "Amplitude (mV)",
            "width_kHz": "Width (kHz)",
            "resolution_mass": "Mass resolving power",
        })
        print(f"\nDetected peaks ({len(peaks)})")
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    else:
        print("\nNo peaks met the detection thresholds.")

    processor.plot()



if __name__ == "__main__":
    main()

