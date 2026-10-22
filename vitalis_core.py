"""
VITALIS core utilities
======================
Camera-based heart-rate estimation (remote photoplethysmography, rPPG)
and ECG reference processing. Shared by the evaluation notebook and the
live webcam script so both use exactly the same signal processing.

Research prototype - NOT a medical device.
"""

import cv2
import numpy as np
from scipy import signal

# Plausible resting/normal heart-rate band: 0.7-3.0 Hz = 42-180 BPM
HR_LOW_HZ = 0.7
HR_HIGH_HZ = 3.0


# --------------------------------------------------------------------------
# Video -> RGB trace
# --------------------------------------------------------------------------
def extract_rgb_signal(video_path):
    """Return (fps, rgb) where rgb has shape (n_frames, 3) with the mean
    R, G, B value of every frame (the videos are already cropped to a
    skin region)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    means = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bgr_mean = frame.reshape(-1, 3).mean(axis=0)
        means.append(bgr_mean[::-1])  # BGR -> RGB
    cap.release()

    if not means:
        raise RuntimeError(f"No frames could be read from: {video_path}")
    return fps, np.asarray(means, dtype=float)


def resample_uniform(timestamps, values, fs=30.0):
    """Interpolate an irregularly sampled signal (e.g. webcam frames) onto
    a uniform time grid. `values` may be 1-D or (n, channels)."""
    t = np.asarray(timestamps, dtype=float)
    v = np.asarray(values, dtype=float)
    t_new = np.arange(t[0], t[-1], 1.0 / fs)
    if v.ndim == 1:
        return t_new, np.interp(t_new, t, v)
    out = np.column_stack([np.interp(t_new, t, v[:, i]) for i in range(v.shape[1])])
    return t_new, out


# --------------------------------------------------------------------------
# rPPG: POS algorithm + filtering + spectral heart-rate estimate
# --------------------------------------------------------------------------
def pos_algorithm(rgb, fps, window_seconds=1.6):
    """Plane-Orthogonal-to-Skin (POS) pulse extraction
    (Wang et al., IEEE TBME 2017), using the sliding-window overlap-add
    formulation.

    rgb : array (n_frames, 3) of mean R, G, B per frame
    returns : pulse signal, shape (n_frames,)
    """
    rgb = np.asarray(rgb, dtype=float)
    n = rgb.shape[0]
    win = int(round(window_seconds * fps))
    if n < win:
        raise ValueError("Signal is shorter than the POS window.")

    projection = np.array([[0.0, 1.0, -1.0],     # G - B
                           [-2.0, 1.0, 1.0]])    # -2R + G + B
    pulse = np.zeros(n)

    for start in range(0, n - win + 1):
        block = rgb[start:start + win].T                       # 3 x win
        block = block / (block.mean(axis=1, keepdims=True) + 1e-8)
        s = projection @ block                                 # 2 x win
        p = s[0] + (s[0].std() / (s[1].std() + 1e-8)) * s[1]
        pulse[start:start + win] += p - p.mean()

    return pulse


def bandpass(x, fs, low=HR_LOW_HZ, high=HR_HIGH_HZ, order=3):
    """Zero-phase Butterworth band-pass filter."""
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def _spectrum(x, fs, nfft=2 ** 14):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    return signal.welch(x, fs=fs, window="hann",
                        nperseg=len(x), nfft=max(nfft, len(x)))


def estimate_hr(x, fs, low=HR_LOW_HZ, high=HR_HIGH_HZ, nfft=2 ** 14):
    """Heart rate in BPM from the dominant spectral peak. Zero-padding
    (nfft) gives a fine frequency grid so the BPM values are not
    artificially quantised."""
    freqs, power = _spectrum(x, fs, nfft)
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return float("nan")
    return float(freqs[mask][np.argmax(power[mask])] * 60.0)


def pulse_snr(x, fs, low=HR_LOW_HZ, high=HR_HIGH_HZ, half_width_hz=0.1):
    """Simple signal-quality index (dB): power around the dominant peak
    and its first harmonic versus the rest of the heart-rate band."""
    freqs, power = _spectrum(x, fs)
    band = (freqs >= low) & (freqs <= high)
    f_peak = freqs[band][np.argmax(power[band])]
    near = (np.abs(freqs - f_peak) <= half_width_hz) | \
           (np.abs(freqs - 2 * f_peak) <= half_width_hz)
    signal_power = power[band & near].sum()
    noise_power = power[band & ~near].sum()
    return float(10 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12)))


# --------------------------------------------------------------------------
# ECG reference
# --------------------------------------------------------------------------
def filter_ecg(ecg, fs, low=0.5, high=40.0, order=3):
    """Band-pass filter an ECG recording."""
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(ecg, dtype=float))


def detect_r_peaks(ecg_filtered, fs, min_rr_seconds=0.4, prominence_std=0.5,
                   min_height_fraction=0.5):
    """Detect R-peaks. The trace is flipped if the dominant deflection is
    negative. A second pass drops peaks whose height is below
    `min_height_fraction` of the median peak height (filters out T-waves
    and ringing). Returns sample indices."""
    x = np.asarray(ecg_filtered, dtype=float)
    if abs(x.min()) > abs(x.max()):
        x = -x
    peaks, _ = signal.find_peaks(
        x,
        distance=int(min_rr_seconds * fs),
        prominence=prominence_std * np.std(x),
    )
    if len(peaks) > 3:
        heights = x[peaks]
        peaks = peaks[heights >= min_height_fraction * np.median(heights)]
    return peaks


def hr_from_peaks(peaks, fs, start_s, end_s, min_beats=3):
    """Mean heart rate (BPM) from R-peaks inside [start_s, end_s).
    Implausible RR intervals (<0.3 s or >1.5 s) are discarded."""
    times = np.asarray(peaks) / fs
    t = times[(times >= start_s) & (times < end_s)]
    if len(t) < min_beats:
        return float("nan")
    rr = np.diff(t)
    rr = rr[(rr >= 0.3) & (rr <= 1.5)]
    if len(rr) == 0:
        return float("nan")
    return float(60.0 / np.mean(rr))
