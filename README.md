# 🚋 Smart Count Tramway
**Edge-AI Automatic Passenger Counting System**  
*PFE Start-up Project — SETRAM Mostaganem*

---

## Overview

Smart Count Tramway is an embedded AI system that uses existing tram cameras to automatically count passenger entries and exits at each stop, with **≥ 95% accuracy** and **< 100ms** per frame.

Built for the **Mostaganem Tramway** (14 km network), it addresses peak-hour management, tourist-season frequency adjustment, and fare evasion monitoring — without adding ground personnel.

---

## Architecture

```
Camera Feed → Face Blur (Privacy) → YOLOv8 Detection → DeepSORT Tracking
     → Virtual Line Crossing → SQLite Storage → Streamlit Dashboard
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection | YOLOv8 (Ultralytics 8.4+) |
| Tracking | DeepSORT (deep-sort-realtime 1.3.2) |
| Privacy | OpenCV Haar Cascade (Gaussian blur 99×99) |
| Backend | Python 3.12, PyTorch 2.10, OpenCV 4.10 |
| Dashboard | Streamlit 1.45 + Plotly 5.24 |
| Storage | SQLite (built-in, no server needed) |
| Hardware | Raspberry Pi 4 / Jetson Nano |

---

## System Requirements

| Requirement | Details |
|-------------|---------|
| **Python** | 3.12 exactly — 3.13/3.14 not supported by PyTorch |
| **OS** | Windows 10/11 (dev), Linux (Raspberry Pi / Jetson) |
| **RAM** | 4 GB minimum, 8 GB recommended |
| **Visual C++** | Microsoft Visual C++ Redistributable 2022 (Windows only) |

---

## Installation

### Step 1 — Install Python 3.12
Download and install Python 3.12 (check "Add to PATH" during install):  
👉 https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe

### Step 2 — Install Visual C++ Redistributable *(Windows only)*
Required by PyTorch DLLs:  
👉 https://aka.ms/vs/17/release/vc_redist.x64.exe

### Step 3 — Clone the repo
```bash
git clone https://github.com/badiesaadi/smart-count-tramway.git
cd smart-count-tramway
```

### Step 4 — Install dependencies

**Windows:**
```bash
py -3.12 -m pip install -r requirements.txt
```

**Linux / Raspberry Pi / Jetson Nano:**
```bash
pip install -r requirements.txt
```

### Step 5 — Patch DeepSORT *(Python 3.12+ fix)*
`deep-sort-realtime` uses a deprecated API removed in Python 3.12. Run the included patch script once:

```bash
py -3.12 patch_deepsort.py
```

Expected output: `Patched OK`

### Step 6 — Download YOLOv8 weights

**Windows:**
```bash
py -3.12 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
move yolov8n.pt models\
```

**Linux:**
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt models/
```

---

## Usage

### Run the counter

**Windows:**
```bash
# Test with a local video file
py -3.12 main.py --source data/videos/test.mp4 --stop Kharouba

# Live RTSP camera stream
py -3.12 main.py --source rtsp://192.168.1.100:554/stream1 --stop Salamandre

# Webcam (index 0)
py -3.12 main.py --source 0 --stop "Gare SNTF"

# Headless mode — no display window (for Pi/Jetson)
py -3.12 main.py --source 0 --stop Kharouba --headless

# Debug mode — show raw detection boxes
py -3.12 main.py --source 0 --stop Kharouba --debug
```

**Linux / Raspberry Pi:**
```bash
python main.py --source /dev/video0 --stop Kharouba --headless
```

### Launch the dashboard

**Windows:**
```bash
py -3.12 -m streamlit run dashboard/app.py
```

**Linux:**
```bash
streamlit run dashboard/app.py
```

Open your browser at **http://localhost:8501**

---

## Project Structure

```
smart-count-tramway/
├── main.py                  # Main execution loop
├── requirements.txt         # Python 3.12 compatible dependencies
├── patch_deepsort.py        # One-time fix for DeepSORT on Python 3.12+
├── src/
│   ├── detection.py         # YOLOv8 inference + at-the-edge face blurring
│   ├── tracking.py          # DeepSORT + virtual line crossing logic
│   └── database.py          # SQLite schema and operations
├── dashboard/
│   └── app.py               # Streamlit real-time dashboard
├── models/                  # YOLOv8 weights (.pt files — not tracked by git)
├── data/
│   └── videos/              # Test footage (not tracked by git)
├── config/
│   └── settings.yaml        # Configurable parameters
└── scripts/
    └── push_to_github.py    # One-click GitHub deployment utility
```

---

## Performance Targets

| Metric | Target | Implementation |
|--------|--------|---------------|
| Accuracy | ≥ 95% | YOLOv8n + DeepSORT n_init=3 |
| Processing speed | < 100ms/frame | FP16 inference, CPU fallback |
| Privacy compliance | At-the-edge | Face blur applied before any storage |

---

## Privacy

All face data is anonymised **on-device** before any storage or transmission, using Gaussian blur (kernel 99×99) applied immediately after capture. Raw face pixels never leave the device. This satisfies the anonymisation requirement defined in §3.2 of the Cahier des Charges.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `pip` not recognised on Windows | Use `py -3.12 -m pip install ...` |
| `torch` install fails | Make sure you are on Python 3.12, not 3.13/3.14 |
| `pandas` build error (vswhere) | Python version too new — install Python 3.12 |
| `DLL initialization failed` (c10.dll) | Install Visual C++ Redistributable 2022 |
| `No module named pkg_resources` | Run `py -3.12 patch_deepsort.py` |
| `resource_filename` AttributeError | Run `py -3.12 patch_deepsort.py` |
| `streamlit` not recognised | Use `py -3.12 -m streamlit run ...` or add Python312\Scripts to PATH |
| SSL certificate error on Windows | Already patched in `scripts/push_to_github.py` |
| Camera not opening | Check `--source` index (try 0, 1) or verify RTSP URL |

---

## Academic Defense

This project is the **PFE (Projet de Fin d'Études)** for the Start-up track at the University of Mostaganem.

Deliverables:
- Functional prototype (passenger counting application + monitoring interface)
- Business Plan + Pitch Deck (market study for the 14 km Mostaganem network)
- Memoire de PFE

---

*SETRAM Mostaganem · Universite de Mostaganem · 2024-2025*