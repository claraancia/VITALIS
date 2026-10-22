"""
VITALIS live webcam demo
========================
Detects your face, tracks the forehead, records for CAPTURE_SECONDS and
estimates heart rate with the POS algorithm.

Usage:   python vitalis_live.py
Quit:    press "q" in the video window.

Research prototype - NOT a medical device. There is no ground truth in
this script: the number it prints is an estimate only.
"""

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

import vitalis_core as vc

CAPTURE_SECONDS = 30      # longer windows give a finer, more stable estimate
MIN_FACE_WIDTH = 100      # pixels
TARGET_FS = 30.0          # uniform rate used for processing
MIN_SNR_DB = -3.0         # below this the result is flagged as unreliable


def forehead_roi(face, frame_shape):
    """Forehead box inside a detected face, clipped to the frame."""
    fx, fy, fw, fh = face
    x1 = max(0, fx + int(0.25 * fw))
    x2 = min(frame_shape[1], fx + int(0.75 * fw))
    y1 = max(0, fy + int(0.10 * fh))
    y2 = min(frame_shape[0], fy + int(0.30 * fh))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def main():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        raise RuntimeError("Could not load the OpenCV face detector.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam.")

    rgb_trace, timestamps = [], []
    started, t_start, last_roi = False, None, None

    print("VITALIS live measurement")
    print("Face the camera in good, even light and keep still.")
    print(f"Measurement starts when a face is found and lasts {CAPTURE_SECONDS} s.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read a frame from the camera.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(MIN_FACE_WIDTH, MIN_FACE_WIDTH),
        )

        if len(faces) > 0:
            face = max(faces, key=lambda f: f[2] * f[3])
            roi = forehead_roi(face, frame.shape)
            if roi is not None:
                last_roi = roi
                fx, fy, fw, fh = face
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)

        if last_roi is not None and not started:
            started = True
            t_start = time.perf_counter()
            rgb_trace, timestamps = [], []
            print("Face found - measuring...")

        elapsed = 0.0
        if started and last_roi is not None:
            x1, y1, x2, y2 = last_roi
            patch = frame[y1:y2, x1:x2]
            # If the face is briefly lost, the last known ROI is reused.
            rgb_trace.append(patch.reshape(-1, 3).mean(axis=0)[::-1])  # BGR->RGB
            elapsed = time.perf_counter() - t_start
            timestamps.append(elapsed)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            status = f"Measuring: {max(0.0, CAPTURE_SECONDS - elapsed):.1f}s"
        else:
            status = "Position your face in the camera"

        cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Keep still", (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.imshow("VITALIS - live heart rate", frame)

        if started and elapsed >= CAPTURE_SECONDS:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(rgb_trace) < 5 * TARGET_FS:
        raise RuntimeError("Not enough camera data was captured.")

    # Webcam frames arrive at slightly irregular times -> resample.
    t_uniform, rgb_uniform = vc.resample_uniform(timestamps, rgb_trace, TARGET_FS)

    pulse = vc.bandpass(vc.pos_algorithm(rgb_uniform, TARGET_FS), TARGET_FS)
    bpm = vc.estimate_hr(pulse, TARGET_FS)
    snr = vc.pulse_snr(pulse, TARGET_FS)

    print("\n" + "=" * 50)
    print(f"Samples: {len(rgb_trace)} | recorded {timestamps[-1]:.1f} s")
    print(f"Estimated heart rate: {bpm:.1f} BPM")
    print(f"Signal quality (SNR): {snr:.1f} dB")
    if snr < MIN_SNR_DB:
        print("Low signal quality - the estimate may be unreliable. "
              "Try better lighting and stay still.")
    print("Research prototype - not a medical device.")
    print("=" * 50)

    plt.figure(figsize=(12, 4))
    plt.plot(t_uniform, pulse)
    plt.xlabel("Time (s)")
    plt.ylabel("rPPG amplitude (a.u.)")
    plt.title(f"VITALIS POS rPPG - estimated HR {bpm:.1f} BPM (SNR {snr:.1f} dB)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
