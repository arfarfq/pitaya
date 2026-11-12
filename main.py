from src.data_processor import DataProcessor
from pathlib import Path



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
    processor.frequency_to_mz()
    fig, axes = processor.plot()


if __name__ == "__main__":
    main()

