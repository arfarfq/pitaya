from src.data_processor import DataProcessor
from pathlib import Path


REF_HZ = 116e3
REF_MZ = 23
HARMONIC_ORDER = 1
REF_HARMONIC_ORDER = 1
MZ_LIMITS = (1, 100)  # use None to display every mapped FFT bin
FILE_PATH = "../data/k_20ms_avg128_26.5degC.csv"

def main():
    waveform_path = Path(FILE_PATH)
    if not waveform_path.is_file():
        waveform_path = Path("data") / waveform_path.name
    processor = DataProcessor.load_data(waveform_path)

    # Optional: select a settled interval, in µs, before calculating the FFT:
    # use = (processor.x >= 0) & (processor.x < 10_000)
    # processor = DataProcessor(processor.x[use], processor.y[use])

    processor.dft()
    processor.frequency_to_mz(
        ref_hz=REF_HZ, ref_mz=REF_MZ, harmonic_order=HARMONIC_ORDER,
        ref_harmonic_order=REF_HARMONIC_ORDER,
        calibration_label=None,
    )
    fig, axes = processor.plot(mz_limits=MZ_LIMITS)


if __name__ == "__main__":
    main()

