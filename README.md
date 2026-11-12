# Pitaya / Racetrap signal processing

An experimental notebook framework for converting detector transients to a frequency spectrum and an exploratory mass-to-charge spectrum. Decay analysis is deferred.

Open [notebooks/pipeline.ipynb](notebooks/pipeline.ipynb), select the project's `.venv` Python kernel, and run all cells. See [the environment guide](pitaya-environment-guide.md) for setup. Only NumPy, pandas, Matplotlib and the existing IPython kernel are needed.

Edit the **Experiment settings** cell to choose the waveform, CSV columns and unit conversions, analysis and baseline intervals, window, zero padding, calibration references and harmonic, peak search, plot limits and optional export. Run all cells again after changing settings. The notebook keeps seconds, hertz and volts internally and reports sampling and FFT diagnostics.

**The default calibration is an unconfirmed demonstration:** nominal m/z 39 at approximately 120 kHz from the handwritten sketch. It does not establish the species or calibrated masses in the supplied raw trace. For experimental use, choose `calibration.mode = "references"` and enter measured reference frequencies, known m/z values and detector harmonics for the current trap conditions. The starting model is `m/z = A / (observed_frequency / harmonic)^2`; residuals and extrapolation flags support review. A single reference cannot validate the calibration or its uncertainty.

The raw input is `data/k_20ms_avg128_26.5degC.csv`, with time in column 3 (seconds) and voltage in column 4 (volts), using zero-based indices. The `clean_FFT` and `clean_SPEC` files are already processed results from a different, 4 ms experiment; the notebook displays them separately for context. Their amplitude normalization and calibration provenance are unknown.

The notebook includes executable numerical sanity checks and optional CSV/JSON export to ignored `outputs/`. Spectral amplitude remains detector voltage; it is not ion abundance or pressure. Existing exploratory work in `notebooks/test.ipynb` and `src/data_processor.py` is preserved. Local plans and the work log live in ignored `agent.md`.
