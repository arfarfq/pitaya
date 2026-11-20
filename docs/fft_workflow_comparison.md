# FFT and decay analysis workflow comparison

This document records the original workflow comparison of [FFT_decay_analysis_Nov_2026.py](../src/FFT_decay_analysis_Nov_2026.py) and [data_processor.py](../src/data_processor.py), before the alignment described below. The original comparison was based on source inspection; neither analysis was executed at that stage.

**Subsequent alignment:** `DataProcessor` now uses a symmetric Blackman–Harris window, an 8N-point FFT by default, November's amplitude normalization and 500 kHz cutoff, uniform time-axis reconstruction, a reusable DC-corrected waveform, and metadata-tolerant legacy CSV loading. Its default reference is now 38.9637065 at 105.912 kHz with zero offset, and its default mapping filters to 1–200. Plotting uses the corrected waveform and a matching 1–200 display range. The class retains microseconds/millivolts/Hz, native capture loading, caching, and explicit harmonic-order options; padding, cutoff, offset, and mapping range are configurable. Regression tests compare its FFT against the November function in equivalent units. Decay fitting, peak/family analysis, abundances, and file reporting remain exclusive to the November script. The sections below describe the pre-alignment behavior.

## Overview

**Current peak-analysis update:** `DataProcessor` now also provides `detect_peaks()`, `detect_harmonics()`, and `clean_spectrum()`. Mapping to m/z runs these automatically with November's default noise thresholds, 0.50 kHz minimum separation, harmonic orders through 10, and 1% relative tolerance. Cleaning retains the same ±2-width neighborhoods used for peak integration and zeros other bins for the m/z plot. Raw FFT values remain available; `plot(clean_mz=False)` displays their original mapped amplitudes. Decay fitting, abundance calculation, and file reporting remain exclusive to the November script. The comparison below is historical.

**Current units after the subsequent unit migration:** `DataProcessor` now uses seconds and volts for constructor inputs, CSV loading, and waveform storage; FFT outputs are kHz and volts. Frequency arguments are named `cutoff_khz`, `ref_khz`, `offset_khz`, and `frequency_limits_khz`. The historical units described elsewhere below apply to the original comparison.

Both files remove a constant baseline, apply a Blackman–Harris window, calculate a real FFT, normalize its amplitude, and map frequency through an inverse-square calibration.

The November script is a complete, single-file analysis pipeline: it also fits transient decay, measures spectral peaks, groups harmonics, estimates relative abundances, and exports a report. `DataProcessor` is a reusable class whose caller chooses when to compute the FFT, map it to m/z, and plot the available results.

## 1. November analysis script

### Execution and configuration

The script runs its analysis at module level, including when imported: there is no `if __name__ == "__main__"` guard. Paths, calibration references, fitting settings, and thresholds are module constants. Its configured input directory is a Windows desktop path, so it needs adjustment on other machines.

### Workflow

1. **Prepare calibration.** Calculate a candidate frequency offset from two references: 38.9637065 Da at 105.912 kHz and 40.9618253 Da at 103.311 kHz. Then explicitly override the offset with `f0 = 0` and calculate `K` from reference 1:

   ```text
   K = m_ref_1 × (f_ref_1_khz − f0)²
   m = K / (f_khz − f0)²
   ```

2. **Load the waveform CSV.** Read comma-separated lines manually, taking time from column 4 and voltage from column 5 (indices 3 and 4). Skip short lines and rows whose selected fields cannot be converted to floats. Use seconds and volts directly. The input is described as background-subtracted; this script does not subtract a separate background recording.

3. **Validate and reconstruct time.** Require at least two samples, matching array lengths, and strictly increasing timestamps. Calculate and print the mean and median sampling intervals. Use the mean interval to replace the timestamps with a uniform axis while preserving the original start time. Voltage samples are retained without interpolation.

4. **Remove DC.** Subtract the mean voltage once to produce `voltage_ac`, used by both decay analysis and the FFT.

5. **Fit transient decay.** Detect waveform maxima with `find_peaks`, using a minimum spacing of approximately 70% of the expected oscillation period (default expected frequency: 100 kHz) and prominence of 5% of the maximum absolute corrected voltage. Retain positive maxima within the configured 10–40 ms interval. Require at least three fitting points, fit `ln(amplitude)` against time, and require a negative slope. Derive:

   ```text
   tau = −1 / slope
   estimated power-spectrum FWHM = 1 / (π × tau)
   estimated amplitude-spectrum FWHM = √3 / (π × tau)
   ```

   Record the fit's R² and create a logarithmic-envelope plot. These linewidths are decay-based estimates, separate from measured FFT peak widths. A decay fitting failure stops the script before FFT computation and export.

6. **Compute and save the FFT.** Apply `blackmanharris(N)` (the default symmetric window), zero-pad to `8 × N`, and calculate a SciPy real FFT. Divide magnitudes by the window sum and double them, then halve DC again. Convert the frequency axis to kHz and retain frequencies up to 500 kHz. Save the spectrum to CSV.

7. **Print integral diagnostics.** Integrate squared corrected voltage over time and squared FFT amplitude over frequency. The source explicitly notes that these are not a direct Parseval comparison: the FFT is windowed and its amplitudes are not a power spectral density.

8. **Map and save the mass spectrum.** Apply the calibration to frequencies above `f0`, retain masses from 1 to 200 Da, and sort by increasing mass. Carry FFT amplitudes through unchanged and export the mapped spectrum.

9. **Detect and measure FFT peaks.** Estimate the noise median and a robust spread (`1.4826 × median absolute deviation`). Require prominence of at least twice that spread, height of at least the median plus five times the spread, and separation of at least 0.50 kHz. Measure widths at half prominence and convert their frequency boundaries into mass widths. Integrate amplitude and squared amplitude over a neighborhood extending approximately two measured widths on either side. Calculate frequency and mass resolving powers.

10. **Group harmonic families.** Process detected peaks from lowest frequency upward. Treat each unused peak as a fundamental, then look for the closest unused peak within 1% of each integer multiple through order 10. Assign each peak at most once and sum the family's integrated powers. After grouping, retain families with fundamental masses from 1 to 300 Da.

11. **Calculate relative abundances.** Normalize retained families' fundamental powers to percentages, and separately normalize their total harmonic powers to percentages. The two abundance columns use different denominators.

12. **Create plots and report.** Plot the FFT and mapped mass spectrum, mark family fundamentals, and annotate mass peaks with fundamental-power abundance. Create PDF pages for the acquisition/calibration summary, ion families, individual FFT peaks, decay results, spectra, waveform, and envelope fit. Tables paginate when needed. Save the PDF and display plots with `plt.show()`.

### Outputs

Files are written beside the configured input, using its filename stem:

| Output | Contents |
| --- | --- |
| `<stem>_FFT.csv` | Frequency in kHz and normalized amplitude in V |
| `<stem>_SPEC.csv` | Filtered mass axis (`Mass_u`) and amplitude in V |
| `<stem>_Analysis_Report.pdf` | Summary, measurements, tables, and figures |

Peak and family tables appear in the PDF; they are not exported as separate CSV files.

## 2. `DataProcessor`

### Execution and input

Importing the module defines constants and the class without running an analysis. The caller can construct `DataProcessor(x, y)` with time in microseconds and voltage in millivolts, or use `DataProcessor.load_data(path)` to read a CSV.

### Workflow

1. **Load and convert data.** If the CSV headers are exactly `time_s,voltage_v`, read those named columns. Otherwise, read a headerless legacy CSV using columns 4 and 5. In both cases, convert seconds to microseconds and volts to millivolts. Numeric conversion uses `errors="raise"`; malformed numeric content is not silently skipped.

2. **Initialize the processor.** Copy the input arrays as floating-point arrays and mark them read-only. Derive the sampling interval in seconds from the first timestamp, last timestamp, and sample count:

   ```text
   dt_seconds = (x[-1] − x[0]) × 10⁻⁶ / (N − 1)
   ```

   Preserve the supplied time axis and initialize the cached FFT and calibration state. There are no explicit checks here for matching lengths, sufficient samples, monotonicity, or uniform sampling.

3. **Compute the FFT on demand with `dft()`.** On the first call, subtract the mean voltage, apply a periodic Blackman–Harris window (`sym=False`), and calculate a NumPy real FFT with no zero padding or frequency cutoff. Normalize by the window sum, then double positive-frequency bins except the Nyquist bin for even sample counts. Cache and return frequencies in Hz and peak amplitudes in mV. Later calls reuse the cached arrays.

4. **Optionally map frequency to m/z.** `frequency_to_mz()` calls `dft()` if necessary, excludes DC, and maps all positive-frequency bins using:

   ```text
   m/z = ref_mz × [(ref_hz / ref_harmonic_order)
                  / (frequency_hz / harmonic_order)]²
   ```

   Defaults are an observed reference frequency of 116 kHz, reference m/z of 23, and harmonic orders of 1. Reference and spectrum harmonic orders can be set independently, but one spectrum harmonic order applies to the entire mapping. Store the resulting m/z array and a descriptive calibration label; the default label is `Unconfirmed reference`.

5. **Plot the available state with `plot()`.** Show the original waveform alone, waveform plus FFT, or waveform plus FFT plus m/z, depending on which calculations have been performed. Plotting itself does not trigger the calculations. Sort m/z and its matching amplitudes for display. Optional frequency limits and default m/z limits of 1–100 change the view without filtering stored results. Display the figure and return its figure and axes objects.

### Outputs

The class returns or stores numerical arrays and returns plot objects. It does not detect spectral peaks, fit decay, identify harmonic families, calculate abundances, or write CSV/PDF files.

## 3. Side-by-side comparison

| Aspect | November script | `DataProcessor` |
| --- | --- | --- |
| Control flow | Fixed pipeline executed at module level | Caller-controlled methods with cached FFT |
| CSV inputs | Legacy columns 4/5; skips nonnumeric rows | Native named columns or legacy columns 4/5; numeric errors raise |
| Working units | Seconds, volts; FFT in kHz | Microseconds, millivolts; FFT in Hz |
| Time handling | Replaces timestamps with a uniform axis | Preserves timestamps; derives an effective interval from endpoints |
| DC correction | Corrected waveform reused throughout analysis | Applied inside FFT; time plot shows original voltage |
| Window | Symmetric Blackman–Harris | Periodic Blackman–Harris |
| FFT length | `8 × N` by default | `N` |
| Frequency coverage | Up to 500 kHz, bounded by available FFT bins | Full real-FFT range through Nyquist |
| Nyquist scaling | Doubles the final bin even when it is Nyquist | Explicitly leaves even-N Nyquist amplitude undoubled |
| Calibration | Offset-capable equation; currently forces zero offset and uses reference 1 | Zero-offset ratio equation with explicit harmonic-order parameters |
| Axis meaning | Labels result as mass in Da/u | Labels result as mass-to-charge in Th |
| Harmonic treatment | Greedy grouping of detected peaks | User selects one order for the entire mapped spectrum |
| Mass range | Filters spectrum to 1–200; families separately to 1–300 | Stores all positive-bin mappings; default display zoom is 1–100 |
| Quantification | Peak widths, integrated powers, resolving power, abundance estimates | Frequency and mapped detector amplitudes |
| Reporting | CSVs, PDF, console results, interactive plots | In-memory arrays and interactive plots |

## 4. Implications when comparing results

- **Convert units first.** For equivalent physical amplitudes, `DataProcessor` values in mV are 1,000 times values in V; its frequencies in Hz are 1,000 times frequencies in kHz.
- **Align calibration before comparing mass axes.** The files use different reference values. With zero offset and both harmonic orders equal to 1, their mappings share the same inverse-square mathematical form, but their default calibration constants differ. Mass and m/z labels also require a consistent charge-state interpretation.
- **Expect different FFT grids and small window differences.** Eightfold padding samples the spectrum more densely but does not increase the observation-time-limited resolution. A direct bin-by-bin comparison requires matching the FFT length and window convention.
- **Neither implementation measures sampling deviations against a tolerance.** The November script declares `MAX_SAMPLING_DEVIATION = 0.01` but never uses it. Its reconstructed time axis does not resample the voltage. For matching input endpoints and sample count, both implementations derive the same effective sampling interval, though only the November script replaces the time coordinates.
- **Harmonic assumptions differ substantially.** The November script makes automatic, order-dependent assignments starting with the lowest frequencies. `DataProcessor` leaves identification to the caller; choosing one order does not separate a mixture of fundamentals and harmonics.
- **Mapped amplitude is not automatically abundance.** Both retain detector amplitude when relabeling the horizontal axis. Only the November script computes normalized family-power estimates. Its abundance denominator includes retained families up to 300 Da, even though the exported spectrum stops at 200 Da.
- **The decay fit does not correct the FFT.** In the November script, decay and FFT analysis use the same DC-corrected waveform, but the fitted decay constant is reported rather than used to compensate or deconvolve the spectrum.
