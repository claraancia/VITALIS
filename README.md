# VITALIS: Camera-Based Heart Rate Estimation

VITALIS estimates heart rate from a face video using **remote photoplethysmography (rPPG)**: tiny colour changes in the skin caused by blood flow. It uses the **POS (Plane-Orthogonal-to-Skin)** algorithm and is validated against simultaneous **ECG** recordings on a 27-subject dataset.

Built by Irakoze Ancia Clara, final-year AI student at Olivia University Bujumbura, Burundi.

> **Disclaimer:** research prototype for education and experimentation. It is **not a medical device** and must not be used for diagnosis or treatment decisions.

## Results

*Replace this table with the numbers from `results/metrics_overall.csv` after running the notebook.*

| Subset | Windows | MAE (BPM) | RMSE (BPM) | Pearson r | Within 5 BPM |
|---|---|---|---|---|---|
| All subjects | | | | | |
| Excluding arrhythmia | | | | | |

Evaluation uses non-overlapping 30-second windows, comparing the camera heart rate with the ECG heart rate from the same seconds.

![Agreement plots](results/agreement_plots.png)

## How it works

1. **RGB trace:** mean R, G, B of the skin region in every frame.
2. **POS:** projects the colour signal onto a plane orthogonal to skin tone, which suppresses motion and lighting changes (sliding 1.6 s window, overlap-add).
3. **Band-pass filter:** 0.7-3.0 Hz (42-180 BPM).
4. **Spectral peak:** dominant frequency of the zero-padded spectrum, converted to BPM.
5. **ECG reference:** band-pass 0.5-40 Hz, R-peak detection, mean RR interval per window.

## Repository structure

```
vitalis_core.py            shared signal-processing functions
VITALIS_evaluation.ipynb   dataset evaluation: camera vs ECG
vitalis_live.py            live webcam demo (run locally)
requirements.txt
results/                   metrics and figures
```

## Setup

```bash
git clone https://github.com/claraancia/vitalis.git
cd vitalis
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset

The evaluation uses the **Dataset_rPPG-10** dataset (27 subjects, three skin-region videos and one ECG recording each). It is not included in this repository.

- Source and licence: **[add the dataset link and licence here]**
- Place it at `dataset/Dataset_rPPG-10/`, or set `VITALIS_DATASET=/path/to/Dataset_rPPG-10`.

### Run the evaluation

```bash
jupyter notebook VITALIS_evaluation.ipynb
```

The first run decodes every video and caches the RGB traces in `results/cache/`, so later runs are much faster.

### Run the live demo

```bash
python vitalis_live.py
```

Face the camera in even lighting and stay still for 30 seconds. Press `q` to stop. The script prints an estimate and a signal-quality value; it has no ground truth, so treat the number as indicative only.

## Limitations

- 27 subjects, resting, controlled lighting, one session each.
- Dataset videos are 64x64 pre-cropped skin regions; real webcam use adds motion and lighting problems.
- ECG sampling rate (1000 Hz) is inferred from recording length (600,000 samples for 600 s).
- Subject 4 videos could not be opened and are excluded.
- Two subjects with arrhythmia are reported separately.
- Spectral peak picking can lock onto a harmonic or noise when signal quality is low.

## Reference

Wang, W., den Brinker, A. C., Stuijk, S., de Haan, G. (2017). *Algorithmic principles of remote PPG.* IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.

## Author

Irakoze Ancia Clara: [LinkedIn](https://linkedin.com/in/clara-irakoze-a30162402) | [GitHub](https://github.com/claraancia)

## Licence

MIT, see `LICENSE`.
