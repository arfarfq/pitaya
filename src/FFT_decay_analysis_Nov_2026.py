# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 10:38 2026
@author: cadame

Combined FFT analysis + transient decay analysis.

Workflow:
1. Load background-subtracted CSV
2. Remove DC offset
3. Check sampling uniformity
4. Extract transient upper envelope
5. Fit logarithmic envelope to obtain decay time constant
6. Compute FFT
7. Detect FFT peaks
8. Mass calibration
9. Identify harmonic families
10. Calculate relative abundances
11. Generate mass spectrum
12. Export CSV files and PDF report
"""
import os
import pandas as pd
import numpy as np
from scipy import fft
from scipy.signal import find_peaks, peak_widths
from scipy.signal.windows import blackmanharris
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# CONFIGURATION
# ============================================================
BASE_DIR = r"C:\Users\cadam\Desktop\6_11"
INPUT_FILE = "2.09e-7_100kpts_100ms_32avg.csv"
INPUT_NAME = os.path.splitext(INPUT_FILE)[0]

FFT_OUT = os.path.join( BASE_DIR, INPUT_NAME + "_FFT.csv")
MASS_SPECTRUM = os.path.join( BASE_DIR, INPUT_NAME + "_SPEC.csv")
REPORT_PDF = os.path.join( BASE_DIR, INPUT_NAME + "_Analysis_Report.pdf")

# FFT settings
FFT_CUTOFF_KHZ = 500
PADDING_FACTOR = 8

# Transient decay settings
EXPECTED_FREQUENCY_HZ = 100000
FIT_START = 0.010       # seconds
FIT_END = 0.040         # seconds
PROMINENCE_FRACTION = 0.05

# Sampling uniformity tolerance
MAX_SAMPLING_DEVIATION = 0.01

# Peak detection
MIN_PEAK_DISTANCE_KHZ = 0.50

# Mass spectrum range
MASS_MIN = 1
MASS_MAX = 200

# Harmonic-family settings
MAX_HARMONIC = 10
HARMONIC_TOLERANCE = 0.01

# MASS CALIBRATION
# ============================================================
# Reference 1
m_ref_1 = 38.9637065
f_ref_1_khz = 105.912

# Reference 2
m_ref_2 = 40.9618253
f_ref_2_khz = 103.311

# Model:
# m = K / (f - f0)^2

sqrt_m1 = np.sqrt(m_ref_1)
sqrt_m2 = np.sqrt(m_ref_2)

# Calculate two-reference frequency offset
calculated_f0 = (f_ref_1_khz * sqrt_m1 - f_ref_2_khz * sqrt_m2) / (sqrt_m1 - sqrt_m2)

# Forcing offset to zero is improving calibration accuracy
f0 = 0.0

# If you intend to use the two-reference offset instead,
# replace the line above with:
# f0 = calculated_f0

sqrt_K = (f_ref_1_khz - f0) * sqrt_m1
K = sqrt_K ** 2

print("\nMass calibration:")
print(f"Reference 1: {m_ref_1:.6f} Da at {f_ref_1_khz:.6f} kHz")
print(f"Reference 2: {m_ref_2:.6f} Da at {f_ref_2_khz:.6f} kHz")
print(f"Calculated two-reference offset = {calculated_f0:.6f} kHz")
print(f"Applied frequency offset f0 = {f0:.6f} kHz")
print(f"Calibration constant K = {K:.6e} kHz²·Da")

# FUNCTION: COMPUTE FFT
# ============================================================
def compute_fft(signal, dt, cutoff_khz=500, padding_factor=8):
    """
    Compute a single-sided FFT.

    The input signal is expected to have already had its DC
    offset removed.

    Processing:
        - Blackman-Harris window
        - Zero padding
        - Amplitude normalization

    Returns:
        freqs_khz
        fft_mag
    """
    N = len(signal)

    if N < 2:
        raise ValueError("Not enough samples to calculate FFT.")

    Tobs = N * dt
    df = 1.0 / Tobs

    print(f"\nObservation time = {Tobs * 1e3:.3f} ms")
    print(f"Nominal frequency resolution = {df:.3f} Hz")

    # Windowing
    window = blackmanharris(N)
    signal_win = signal * window

    # Zero padding
    Nfft = padding_factor * N

    fft_bin_spacing = 1.0 / (Nfft * dt)

    print(
        "FFT bin spacing after zero padding = "
        f"{fft_bin_spacing:.3f} Hz"
    )

    # FFT
    fft_vals = fft.rfft(signal_win, n=Nfft)
    freqs = fft.rfftfreq(Nfft, d=dt)

    # Amplitude normalization
    fft_mag = np.abs(fft_vals)
    fft_mag = fft_mag / np.sum(window) * 2.0

    # DC component should not be doubled
    fft_mag[0] /= 2.0

    # Convert Hz to kHz
    freqs_khz = freqs / 1e3

    # Apply frequency cutoff
    mask = freqs_khz <= cutoff_khz

    return freqs_khz[mask], fft_mag[mask]


# ============================================================
# FUNCTION: FREQUENCY TO MASS
# ============================================================

def frequency_to_mass(
    freq_khz,
    amp,
    K,
    f0,
    m_min=0.5,
    m_max=1000
):
    """
    Convert frequency to mass using:

        m = K / (f - f0)^2
    """

    freq_khz = np.asarray(freq_khz)
    amp = np.asarray(amp)

    # Avoid frequencies at or below the offset
    mask = freq_khz > f0

    corrected_freq = freq_khz[mask] - f0

    mass = K / corrected_freq**2
    amp = amp[mask]

    # Mass range
    mask = (mass >= m_min) & (mass <= m_max)

    mass = mass[mask]
    amp = amp[mask]

    # Sort by mass
    idx = np.argsort(mass)

    return mass[idx], amp[idx]


# ============================================================
# FUNCTION: EXTRACT FREQUENCY PEAKS
# ============================================================

def extract_frequency_peaks(freqs, mags, label, K, f0):
    """
    Detect FFT peaks and calculate:

        - Peak frequency (kHz)
        - Peak width (kHz)
        - Peak mass (Da)
        - Peak mass width (Da)
        - Integrated area
        - Integrated power
        - Frequency resolving power
        - Mass resolving power

    Peak widths are measured at half prominence.
    """

    # Estimate noise level
    noise_median = np.median(mags)

    noise_sigma = 1.4826 * np.median(
        np.abs(mags - noise_median)
    )
    '''
    # Peak detection
    peaks, properties = find_peaks(
        mags,
        prominence=2 * noise_sigma,
        height=noise_median + 5 * noise_sigma,
        distance=64
    )'''
    # Frequency resolution of the FFT
    frequency_resolution = freqs[1] - freqs[0]
    # Convert frequency separation (Hz) into FFT bins
    min_peak_distance_bins = max(1,int(np.ceil(MIN_PEAK_DISTANCE_KHZ / frequency_resolution)))

    peaks, properties = find_peaks(mags,
        prominence=2 * noise_sigma,
        height=noise_median + 5 * noise_sigma,
        distance=min_peak_distance_bins)

    print(f"\n===== {label} =====")
    print(f"Number of peaks found: {len(peaks)}")
    print(f"Noise median: {noise_median:.3e}")
    print(f"Noise sigma: {noise_sigma:.3e}")

    # Calculate peak widths at half prominence
    widths, width_heights, left_ips, right_ips = peak_widths(
        mags,
        peaks,
        rel_height=0.5
    )

    peak_list = []

    # Power spectrum
    power_spectrum = mags**2

    # Array indices for interpolation
    indices = np.arange(len(freqs))

    for i, peak in enumerate(peaks):

        # Peak frequency boundaries
        f_left = np.interp(
            left_ips[i],
            indices,
            freqs
        )

        f_right = np.interp(
            right_ips[i],
            indices,
            freqs
        )

        # Peak width in kHz
        width_khz = f_right - f_left

        # Peak mass using the same calibration as the spectrum
        peak_mass = K / (freqs[peak] - f0)**2

        # Mass width
        if f_left > f0:

            mass_left = K / (f_left - f0)**2
            mass_right = K / (f_right - f0)**2

            width_u = abs(mass_left - mass_right)

        else:
            width_u = np.nan

        # Integration boundaries
        halfwidth = int(np.ceil(widths[i]))

        left = max(0, peak - 2 * halfwidth)
        right = min(len(mags) - 1, peak + 2 * halfwidth)

        # Integrated power
        power = np.trapezoid(
            power_spectrum[left:right + 1],
            freqs[left:right + 1]
        )

        # Integrated amplitude
        area = np.trapezoid(
            mags[left:right + 1],
            freqs[left:right + 1]
        )

        # Frequency resolving power
        if width_khz > 0:
            resolution_frequency = (
                (freqs[peak] - f0) / width_khz
            )
        else:
            resolution_frequency = np.nan

        # Mass resolving power
        if width_u > 0:
            resolution_mass = peak_mass / width_u
        else:
            resolution_mass = np.nan

        peak_list.append({
            "freq": freqs[peak],
            "height": mags[peak],
            "width_kHz": width_khz,
            "width_u": width_u,
            "left_freq_kHz": f_left,
            "right_freq_kHz": f_right,
            "mass": peak_mass,
            "area": area,
            "power": power,
            "resolution_frequency": resolution_frequency,
            "resolution_mass": resolution_mass
        })

    return peak_list


# ============================================================
# FUNCTION: BUILD HARMONIC FAMILIES
# ============================================================

def build_harmonic_families(
    peaks,
    K,
    calibration_f0,
    max_harm=10,
    tol_rel=0.01
):
    """
    Group detected FFT peaks into harmonic families.
    """

    # Lowest frequency first
    peaks = sorted(peaks, key=lambda p: p["freq"])

    used = set()
    families = []

    for i, peak in enumerate(peaks):

        if i in used:
            continue

        # Fundamental frequency
        fundamental_freq = peak["freq"]

        if fundamental_freq <= calibration_f0:
            continue

        # Mass calibration
        mass = K / (fundamental_freq - calibration_f0)**2

        family = {
            "freq": fundamental_freq,
            "mass": mass,
            "height": peak["height"],
            "fundamental_power": peak["power"],
            "total_power": peak["power"],
            "harmonics": [
                {
                    "order": 1,
                    "freq": fundamental_freq,
                    "power": peak["power"]
                }
            ]
        }

        used.add(i)

        # Search for harmonics
        for n in range(2, max_harm + 1):

            target = n * fundamental_freq

            best_index = None
            best_error = np.inf

            for j, candidate in enumerate(peaks):

                if j in used:
                    continue

                error = abs(candidate["freq"] - target) / target

                if error < tol_rel and error < best_error:
                    best_error = error
                    best_index = j

            # Harmonic found
            if best_index is not None:

                used.add(best_index)

                harmonic = peaks[best_index]

                family["harmonics"].append({
                    "order": n,
                    "freq": harmonic["freq"],
                    "power": harmonic["power"]
                })

                family["total_power"] += harmonic["power"]

        families.append(family)

    return families


# ============================================================
# FUNCTION: COMPUTE RELATIVE ABUNDANCE
# ============================================================

def compute_relative_abundance(families):
    """
    Calculate relative abundances using:

        1. Fundamental power only
        2. Total harmonic power
    """

    total_fundamental = sum(
        f["fundamental_power"] for f in families
    )

    total_harmonics = sum(
        f["total_power"] for f in families
    )

    for f in families:

        if total_fundamental > 0:
            f["abundance_fundamental"] = (
                100 * f["fundamental_power"] / total_fundamental
            )
        else:
            f["abundance_fundamental"] = 0

        if total_harmonics > 0:
            f["abundance_total"] = (
                100 * f["total_power"] / total_harmonics
            )
        else:
            f["abundance_total"] = 0

    return families


# ============================================================
# FUNCTION: PLOT HARMONIC FAMILIES
# ============================================================

def plot_harmonic_families(ax, families):

    for f in families:

        ax.plot(f["mass"], f["height"], "ro")

        ax.annotate(
            (
                f"{f['mass']:.2f} u\n"
                f"{f['abundance_fundamental']:.1f}%"
            ),
            (f["mass"], f["height"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            rotation=90
        )


# ============================================================
# LOAD OSCILLOSCOPE DATA
# ============================================================

fullpath = os.path.join(BASE_DIR, INPUT_FILE)

time_values = []
voltage_values = []

with open(fullpath, "r", encoding="utf-8-sig") as file:

    for line in file:

        line = line.rstrip("\r\n")
        parts = line.split(",")

        # Require at least 5 columns
        if len(parts) < 5:
            continue

        # Extract columns 4 and 5
        time = parts[3]
        voltage_value = parts[4]

        try:
            time_value = float(time)
            voltage_value = float(voltage_value)

        except ValueError:
            continue

        time_values.append(time_value)
        voltage_values.append(voltage_value)


# ============================================================
# CONVERT TO NUMPY ARRAYS AND VALIDATE DATA
# ============================================================

time_s = np.array(time_values, dtype=float)
voltage = np.array(voltage_values, dtype=float)

if len(time_s) == 0:
    raise ValueError(
        f"No numerical waveform data found in:\n{fullpath}"
    )

if len(time_s) != len(voltage):
    raise ValueError(
        "Time and voltage arrays have different lengths."
    )

if len(time_s) < 2:
    raise ValueError(
        "Not enough data points for FFT."
    )


# ============================================================
# SAMPLING INTERVAL AND TIME AXIS
# ============================================================

dt_array = np.diff(time_s)

if np.any(dt_array <= 0):
    raise ValueError("Time values must be strictly increasing.")

# Original time-axis diagnostics
dt_mean = np.mean(dt_array)
dt_median = np.median(dt_array)

print("\nORIGINAL TIME AXIS")
print("------------------")
print(f"Mean interval:   {dt_mean*1e9:.6f} ns")
print(f"Median interval: {dt_median*1e9:.6f} ns")

# The mean interval is approximately 100 ns.
# Reconstruct a uniform time axis using the measured
# average interval, rather than the irregular exported
# individual timestamps.

dt = dt_mean
fs = 1.0 / dt

# Reconstruct time axis while preserving the original
# starting timestamp.
time_s = time_s[0] + np.arange(len(time_s)) * dt

print("\nRECONSTRUCTED TIME AXIS")
print("-----------------------")
print(f"Sampling interval: {dt*1e9:.6f} ns")
print(f"Sampling frequency: {fs:.6f} Hz")
print(f"Sampling frequency: {fs/1e6:.6f} MHz")
print(f"Number of samples: {len(time_s):,}")
print(f"Total duration: {(time_s[-1]-time_s[0])*1e3:.6f} ms")


# ============================================================
# REMOVE DC OFFSET ONCE
# ============================================================

voltage_ac = voltage - np.mean(voltage)

print(
    f"\nMean voltage before DC correction: "
    f"{np.mean(voltage):.6e} V"
)

print(
    f"Mean voltage after DC correction: "
    f"{np.mean(voltage_ac):.6e} V"
)


# ============================================================
# TRANSIENT DECAY TIME CONSTANT
# Upper envelope + logarithmic linear fit
# ============================================================

# Minimum distance between consecutive maxima.
# This is set to 70% of the expected oscillation period.

min_distance = max(
    1,
    int(0.7 / (EXPECTED_FREQUENCY_HZ * dt))
)

# Minimum peak prominence
signal_maximum = np.max(np.abs(voltage_ac))

if signal_maximum <= 0:
    raise ValueError(
        "The waveform has no measurable amplitude."
    )

min_prominence = (
    PROMINENCE_FRACTION * signal_maximum
)

# Find significant positive maxima
envelope_peaks, properties = find_peaks(
    voltage_ac,
    distance=min_distance,
    prominence=min_prominence
)

t_peaks = time_s[envelope_peaks]
upper_envelope = voltage_ac[envelope_peaks]

# Select fitting interval
fit_mask = (
    (t_peaks >= FIT_START)
    & (t_peaks <= FIT_END)
    & (upper_envelope > 0)
)

t_fit = t_peaks[fit_mask]
envelope_fit = upper_envelope[fit_mask]

if len(t_fit) < 3:
    raise ValueError(
        "Fewer than three envelope maxima were found in the "
        "fitting interval. Check EXPECTED_FREQUENCY_HZ, "
        "PROMINENCE_FRACTION, FIT_START and FIT_END."
    )

# Natural logarithm of the upper envelope
ln_envelope = np.log(envelope_fit)

# Linear regression:
# ln(A) = intercept + slope * t

fit = linregress(t_fit, ln_envelope)

slope = fit.slope
intercept = fit.intercept
r_squared = fit.rvalue**2

if slope >= 0:
    raise ValueError(
        "The fitted envelope does not decay: the regression "
        "slope is zero or positive. Check the fitting interval."
    )

# Decay time constant
tau = -1.0 / slope
tau_ms = tau * 1000.0

# Ideal Lorentzian FWHM associated with exponential decay
# Amplitude spectrum convention
decay_linewidth_power = 1.0 / (np.pi * tau)
decay_linewidth_amplitude = (np.sqrt(3.0) / (np.pi * tau))

# Fitted line
ln_fit = intercept + slope * t_fit


# ============================================================
# PRINT TRANSIENT DECAY RESULTS
# ============================================================

print("\nTRANSIENT DECAY ANALYSIS")
print("------------------------")
print(f"Expected oscillation frequency: {EXPECTED_FREQUENCY_HZ:.1f} Hz")
print(f"Number of envelope maxima detected: {len(t_peaks)}")
print(f"Number of maxima used in fit: {len(t_fit)}")
print(
    f"Fitting interval: "
    f"{FIT_START * 1000:.1f}–{FIT_END * 1000:.1f} ms"
)
print(f"Linear fit slope: {slope:.6e} s^-1")
print(f"Linear fit intercept: {intercept:.6f}")
print(f"Decay time constant: {tau_ms:.3f} ms")
print(f"Estimated Lorentzian FWHM (power spectrum): {decay_linewidth_power:.3f} Hz")
print(f"Estimated Lorentzian FWHM (amplitude spectrum): "f"{decay_linewidth_amplitude:.3f} Hz")
print(f"Fit R-squared: {r_squared:.5f}")


# ============================================================
# TRANSIENT DECAY PLOT
# ============================================================

decay_fig, decay_ax = plt.subplots(figsize=(10, 6))

decay_ax.scatter(
    t_fit * 1000,
    ln_envelope,
    s=12,
    label="Logarithm of upper envelope"
)

decay_ax.plot(
    t_fit * 1000,
    ln_fit,
    "r--",
    label=f"Linear fit (τ = {tau_ms:.2f} ms)"
)

decay_ax.set_xlabel("Time (ms)")
decay_ax.set_ylabel("ln(Amplitude)")
decay_ax.set_title(
    f"Transient decay analysis (R² = {r_squared:.4f})"
)

decay_ax.legend()
decay_ax.grid(True)

decay_fig.tight_layout()


# ============================================================
# COMPUTE FFT
# ============================================================

freq_voltage, mag_voltage = compute_fft(
    voltage_ac,
    dt,
    cutoff_khz=FFT_CUTOFF_KHZ,
    padding_factor=PADDING_FACTOR
)


# ============================================================
# SAVE FFT DATA
# ============================================================

fft_df = pd.DataFrame({
    "frequency_kHz": freq_voltage,
    "magnitude": mag_voltage
})

fft_df.to_csv(FFT_OUT, index=False)


# ============================================================
# FFT / TIME-DOMAIN INTEGRAL DIAGNOSTIC
# ============================================================

# Note:
# These two integrals are not a direct Parseval comparison.
# mag_voltage is an amplitude spectrum, not a power spectral
# density, and the FFT includes windowing.

fft_energy = np.trapezoid(
    mag_voltage**2,
    freq_voltage
)

time_energy = np.trapezoid(
    voltage_ac**2,
    time_s
)

print("\nINTEGRAL DIAGNOSTIC")
print("-------------------")
print(f"Time-domain integral = {time_energy:.6e}")
print(f"FFT magnitude integral = {fft_energy:.6e}")

if time_energy != 0:
    print(
        f"FFT / time ratio = "
        f"{fft_energy / time_energy:.6e}"
    )


# ============================================================
# MASS CALIBRATION
# ============================================================

mass_axis, amp_axis = frequency_to_mass(
    freq_voltage,
    mag_voltage,
    K,
    f0,
    m_min=MASS_MIN,
    m_max=MASS_MAX
)


# ============================================================
# SAVE MASS SPECTRUM
# ============================================================

mass_spectrum_df = pd.DataFrame({
    "Mass_u": mass_axis,
    "Amplitude": amp_axis
})

mass_spectrum_df.to_csv(
    MASS_SPECTRUM,
    index=False
)


# ============================================================
# FFT PEAK ANALYSIS
# ============================================================

fft_peaks = extract_frequency_peaks(
    freq_voltage,
    mag_voltage,
    "Background-Subtracted Voltage FFT",
    K,
    f0
)


# ============================================================
# PRINT DETECTED FFT PEAKS
# ============================================================

print("\nDETECTED FFT PEAKS")

headers = [
    ("Frequency", 12),
    ("Mass", 14),
    ("Width (kHz)", 14),
    ("Width(Da)", 12),
    ("Amplitude", 14),
    ("f/df", 12),
    ("m/dm", 12),
]

header_line = "".join(
    f"{name:>{width}}"
    for name, width in headers
)

separator = "-" * len(header_line)

print(separator)
print(header_line)
print(separator)

for p in fft_peaks:

    print(
        f"{p['freq']:12.3f}"
        f"{p['mass']:14.2f}"
        f"{p['width_kHz']:14.3e}"
        f"{p['width_u']:12.3e}"
        f"{p['height']:14.3e}"
        f"{p['resolution_frequency']:12.1f}"
        f"{p['resolution_mass']:12.1f}"
    )

print(separator)


# ============================================================
# CREATE FFT PEAK DATAFRAME
# ============================================================

peaks_df = pd.DataFrame(fft_peaks)

if not peaks_df.empty:

    peaks_df = peaks_df[[
        "freq",
        "mass",
        "width_kHz",
        "width_u",
        "height",
        "resolution_frequency",
        "resolution_mass"
    ]]

    peaks_df.columns = [
        "Frequency (kHz)",
        "Mass(Da)",
        "Width (kHz)",
        "Width(Da)",
        "Amplitude (V)",
        "f/df",
        "m/dm"
    ]

else:

    peaks_df = pd.DataFrame(columns=[
        "Frequency (kHz)",
        "Mass(Da)",
        "Width (kHz)",
        "Width(Da)",
        "Amplitude (V)",
        "f/df",
        "m/dm"
    ])


# ============================================================
# BUILD HARMONIC FAMILIES
# ============================================================

harmonic_families = build_harmonic_families(
    fft_peaks,
    K,
    f0,
    max_harm=MAX_HARMONIC,
    tol_rel=HARMONIC_TOLERANCE
)

# Mass range filter
harmonic_families = [
    f for f in harmonic_families
    if 1 <= f["mass"] <= 300
]

# Relative abundances
harmonic_families = compute_relative_abundance(
    harmonic_families
)


# ============================================================
# PRINT HARMONIC FAMILIES
# ============================================================

print("\nACCUMULATED POWERS")
print("---------------------------------------------")

for p in harmonic_families:

    print(
        f"Mass = {p['mass']:.3f} u, "
        f"Harmonics = {len(p['harmonics'])}, "
        f"Fundamental = {p['fundamental_power']:.3e}, "
        f"Total = {p['total_power']:.3e}"
    )


# ============================================================
# PRINT FINAL ION REPORT
# ============================================================

print("\nDETECTED IONS")
print("---------------------------------------------")
print(" Mass(Da)   Harmonics   Fundamental(%)   Total(%)")
print("---------------------------------------------")

for p in harmonic_families:

    print(
        f"{p['mass']:8.3f}   "
        f"{len(p['harmonics']):5d}        "
        f"{p['abundance_fundamental']:8.2f}%      "
        f"{p['abundance_total']:8.2f}%"
    )

print("\nProcessing complete.")


# ============================================================
# PLOT FFT AND MASS SPECTRUM
# ============================================================

fig, axs = plt.subplots(
    2,
    1,
    figsize=(10, 12)
)

# Plot 1: FFT
axs[0].plot(
    freq_voltage,
    mag_voltage,
    linewidth=1
)

axs[0].set_xlim(0, FFT_CUTOFF_KHZ)
axs[0].set_title("Background-Subtracted Voltage FFT")
axs[0].set_xlabel("Frequency (kHz)")
axs[0].set_ylabel("Magnitude (V)")
axs[0].grid(True)

# Mark detected harmonic-family fundamentals
if len(harmonic_families) > 0:

    axs[0].scatter(
        [p["freq"] for p in harmonic_families],
        [p["height"] for p in harmonic_families],
        color="red"
    )

# Plot 2: Mass spectrum
axs[1].plot(
    mass_axis,
    amp_axis,
    linewidth=1
)

plot_harmonic_families(
    axs[1],
    harmonic_families
)

axs[1].set_xlim(0, 200)
axs[1].set_title("Mass Spectrum")
axs[1].set_xlabel("Mass (Da)")
axs[1].set_ylabel("Amplitude")
axs[1].grid(True)

plt.tight_layout()


# ============================================================
# PREPARE PDF TABLES
# ============================================================

# Detected ion-family table
ions_df = pd.DataFrame([
    {
        "Mass(Da)": p["mass"],
        "Fundamental frequency (kHz)": p["freq"],
        "Harmonics": len(p["harmonics"]),
        "Fundamental power": p["fundamental_power"],
        "Total harmonic power": p["total_power"],
        "Fundamental abundance (%)": p["abundance_fundamental"],
        "Total abundance (%)": p["abundance_total"]
    }
    for p in harmonic_families
])

# Individual FFT peak table
report_peaks_df = peaks_df.copy()

# Transient decay results table
decay_df = pd.DataFrame({
    "Parameter": [
        "Number of envelope maxima detected",
        "Number of maxima used in fit",
        "Fitting interval",
        "Expected oscillation frequency",
        "Linear fit slope",
        "Linear fit intercept",
        "Decay time constant",
        "Estimated Lorentzian FWHM (power spectrum)",
        "Estimated Lorentzian FWHM (amplitude spectrum)",
        "Fit R-squared"
    ],

    "Value": [
        str(len(t_peaks)),
        str(len(t_fit)),
        f"{FIT_START * 1000:.1f} - {FIT_END * 1000:.1f} ms",
        f"{EXPECTED_FREQUENCY_HZ:.1f} Hz",
        f"{slope:.6e} s^-1",
        f"{intercept:.6f}",
        f"{tau_ms:.3f} ms",
        f"{decay_linewidth_power:.3f} Hz",
        f"{decay_linewidth_amplitude:.3f} Hz",
        f"{r_squared:.5f}"
    ]
})


# ============================================================
# FUNCTION: ADD DATAFRAME TABLE TO PDF
# ============================================================

def add_table_to_pdf(
    pdf,
    dataframe,
    title,
    rows_per_page=25
):
    """
    Add a DataFrame to the PDF report.

    Large tables are automatically split over multiple pages.
    """

    if dataframe.empty:

        table_fig, table_ax = plt.subplots(
            figsize=(14, 8.5)
        )

        table_ax.axis("off")
        table_ax.set_title(
            title,
            fontsize=16,
            pad=20
        )

        table_ax.text(
            0.5,
            0.5,
            "No results detected",
            ha="center",
            va="center"
        )

        pdf.savefig(
            table_fig,
            bbox_inches="tight"
        )

        plt.close(table_fig)
        return

    for start in range(
        0,
        len(dataframe),
        rows_per_page
    ):

        page_df = dataframe.iloc[
            start:start + rows_per_page
        ]

        table_fig, table_ax = plt.subplots(
            figsize=(14, 8.5)
        )

        table_ax.axis("off")

        table_ax.set_title(
            f"{title} "
            f"(rows {start + 1}–{start + len(page_df)})",
            fontsize=15,
            pad=20
        )

        cell_data = []

        for _, row in page_df.iterrows():

            formatted_row = []

            for column, value in row.items():

                if isinstance(
                    value,
                    (int, float, np.number)
                ):

                    if pd.isna(value):
                        formatted_value = "N/A"

                    elif column == "Mass(Da)":
                        formatted_value = f"{value:.3f}"

                    elif column == "Harmonics":
                        formatted_value = f"{int(value)}"

                    elif "abundance" in column.lower():
                        formatted_value = f"{value:.2f}%"

                    elif "frequency" in column.lower():
                        formatted_value = f"{value:.3e}"

                    else:
                        formatted_value = f"{value:.3e}"

                else:
                    formatted_value = str(value)

                formatted_row.append(formatted_value)

            cell_data.append(formatted_row)

        table = table_ax.table(
            cellText=cell_data,
            colLabels=page_df.columns,
            loc="center",
            cellLoc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)

        pdf.savefig(
            table_fig,
            bbox_inches="tight"
        )

        plt.close(table_fig)


# ============================================================
# CREATE WAVEFORM FIGURE
# ============================================================

waveform_fig, waveform_ax = plt.subplots(
    figsize=(11, 6)
)

waveform_ax.plot(
    time_s * 1e3,
    voltage_ac,
    linewidth=0.8
)

waveform_ax.set_title(
    "Background-Subtracted Waveform"
)

waveform_ax.set_xlabel("Time (ms)")
waveform_ax.set_ylabel("Voltage (V)")
waveform_ax.grid(True)

waveform_fig.tight_layout()


# ============================================================
# GENERATE PDF ANALYSIS REPORT
# ============================================================

with PdfPages(REPORT_PDF) as pdf:

    # --------------------------------------------------------
    # PAGE 1: ANALYSIS SUMMARY
    # --------------------------------------------------------

    summary_fig, summary_ax = plt.subplots(
        figsize=(11, 8.5)
    )

    summary_ax.axis("off")

    summary_text = f"""
MASS SPECTROMETRY ANALYSIS REPORT

Input file:
{INPUT_FILE}


ACQUISITION INFORMATION
Number of samples:              {len(time_s)}
Sampling interval:              {dt:.6e} s
Sampling frequency:             {fs:.6e} Hz
Nyquist frequency:              {fs / 2 / 1e3:.3f} kHz
Observation time:               {len(time_s) * dt * 1e3:.3f} ms
Nominal frequency resolution:   {1 / (len(time_s) * dt):.3f} Hz
Zero-padded FFT bin spacing:    {1 / (PADDING_FACTOR * len(time_s) * dt):.3f} Hz


MASS CALIBRATION
Reference 1: {m_ref_1:.6f} Da at {f_ref_1_khz:.6f} kHz
Reference 2: {m_ref_2:.6f} Da at {f_ref_2_khz:.6f} kHz
Calculated offset: {calculated_f0:.6f} kHz
Applied offset: f0 = {f0:.6f} kHz
Calibration constant: K = {K:.6e} kHz²·Da


TRANSIENT DECAY ANALYSIS
Expected oscillation frequency: {EXPECTED_FREQUENCY_HZ:.1f} Hz
Envelope maxima detected:       {len(t_peaks)}
Maxima used in fit:             {len(t_fit)}
Fitting interval:               {FIT_START * 1000:.1f}–{FIT_END * 1000:.1f} ms
Linear fit slope:               {slope:.6e} s^-1
Decay time constant:            {tau_ms:.3f} ms
Estimated Lorentzian FWHM (power spectrum):      {decay_linewidth_power:.3f} Hz
Estimated Lorentzian FWHM (amplitude spectrum): {decay_linewidth_amplitude:.3f} Hz"
Linear fit R-squared:            {r_squared:.5f}


PEAK DETECTION
Number of detected FFT peaks: {len(fft_peaks)}
Number of detected ion families: {len(harmonic_families)}


OUTPUT FILES
FFT data: {os.path.basename(FFT_OUT)}
Mass spectrum: {os.path.basename(MASS_SPECTRUM)}
Analysis report: {os.path.basename(REPORT_PDF)}
"""

    summary_ax.text(
        0.05,
        0.95,
        summary_text,
        transform=summary_ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace"
    )

    pdf.savefig(
        summary_fig,
        bbox_inches="tight"
    )

    plt.close(summary_fig)


    # --------------------------------------------------------
    # PAGE 2: DETECTED ION FAMILIES
    # --------------------------------------------------------

    add_table_to_pdf(
        pdf,
        ions_df,
        "Detected Ion Families"
    )


    # --------------------------------------------------------
    # PAGE 3: INDIVIDUAL FFT PEAKS
    # --------------------------------------------------------

    add_table_to_pdf(
        pdf,
        report_peaks_df,
        "Individual FFT Peak Measurements"
    )


    # --------------------------------------------------------
    # PAGE 4: TRANSIENT DECAY RESULTS
    # --------------------------------------------------------

    add_table_to_pdf(
        pdf,
        decay_df,
        "Transient Decay Analysis"
    )


    # --------------------------------------------------------
    # PAGE 5: FFT AND MASS SPECTRUM
    # --------------------------------------------------------

    pdf.savefig(
        fig,
        bbox_inches="tight"
    )


    # --------------------------------------------------------
    # PAGE 6: TIME-DOMAIN WAVEFORM
    # --------------------------------------------------------

    pdf.savefig(
        waveform_fig,
        bbox_inches="tight"
    )

    plt.close(waveform_fig)


    # --------------------------------------------------------
    # PAGE 7: LOGARITHMIC ENVELOPE FIT
    # --------------------------------------------------------

    pdf.savefig(
        decay_fig,
        bbox_inches="tight"
    )

    plt.close(decay_fig)


# ============================================================
# FINISH
# ============================================================

print("\nPDF report saved to:")
print(REPORT_PDF)

print("\nFFT data saved to:")
print(FFT_OUT)

print("\nMass spectrum saved to:")
print(MASS_SPECTRUM)

print("\nAll processing completed successfully.")

# Display plots
plt.show()
