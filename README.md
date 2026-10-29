# 🚋 Smart Count Tramway
**Edge-AI Automatic Passenger Counting & Re-Identification System**  
*PFE Start-up Project — SETRAM Mostaganem*

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![CI](https://github.com/badiesaadi/smart-count-tramway/actions/workflows/ci.yml/badge.svg)](https://github.com/badiesaadi/smart-count-tramway/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

Smart Count Tramway is an embedded Edge-AI computer vision system that utilizes existing onboard tram cameras to automatically count passenger entries and exits at each station with **≥ 95% accuracy** and **< 100ms** latency per frame.

Built for the **Mostaganem Tramway** (14 km network, 24 stations), it addresses peak-hour congestion, tourist-season frequency adjustment, and fare evasion monitoring — all without requiring additional ground personnel.

---

## 🏛️ Architecture

```
┌──────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│ Camera Feed  │ ──> │ Edge Face Blurring  │ ──> │  YOLOv8 Detection   │
│ (RTSP / USB) │     │ (On-device Privacy) │     │   (Person Class)    │
└──────────────┘     └─────────────────────┘     └──────────┬──────────┘
                                                            │
┌─────────────────────────┐     ┌─────────────────────┐     │
│   Streamlit Dashboard   │ <── │   SQLite Database   │ <───┼────────────────┐
│   (Real-time Metrics)   │     │ (WAL Mode, Threaded)│     │                │
└─────────────────────────┘     └─────────────────────┘     ▼                ▼
                                                     ┌─────────────┐  ┌─────────────┐
                                                     │  DeepSORT   │  │ Global ReID │
                                                     │  Tracking   │  │(MobileNetV2)│
                                                     └──────┬──────┘  └──────┬──────┘
                                                            │                │
                                                            ▼                ▼
                                                     ┌──────────────────────────────┐
                                                     │   Virtual Line Crossing &    │
                                                     │ Deduplicated Station Counter │
                                                     └──────────────────────────────┘
```

---

## ⚡ Tech Stack

| Layer | Technology | Details |
|---|---|---|
| **Detection** | YOLOv8 (Ultralytics 8.4+) | Person detection (COCO class 0), FP16 acceleration |
| **Tracking** | DeepSORT (`deep-sort-realtime` 1.3.2) | Kalman filter + Hungarian algorithm for intra-camera tracking |
| **Re-Identification** | MobileNetV2 Feature Extractor | 1280-D L2 appearance embeddings & cosine similarity |
| **Privacy Compliance** | OpenCV Haar Cascade / Gaussian Blur | Kernel 99×99 on-device face blurring prior to storage |
| **Backend & Storage** | Python 3.12, PyTorch, SQLite (WAL) | Zero-dependency local storage with ACID compliance |
| **Dashboard** | Streamlit 1.45 + Plotly 5.24 | Real-time passenger throughput and station heatmaps |
| **Target Hardware** | Edge Devices | Raspberry Pi 4 / NVIDIA Jetson Nano / x86_64 Edge PC |

---

## ⚙️ System Requirements

| Requirement | Details |
|---|---|
| **Python** | 3.12 exactly — *(PyTorch & pandas require Python 3.12 for binary wheels)* |
| **OS** | Linux (Raspberry Pi OS / JetPack Ubuntu), Windows 10/11 |
| **RAM** | 4 GB minimum (Raspberry Pi 4), 8 GB recommended |
| **Visual C++** | Microsoft Visual C++ Redistributable 2022 *(Windows only)* |

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/badiesaadi/smart-count-tramway.git
cd smart-count-tramway
```

### 2. Install Dependencies
```bash
# Windows
py -3.12 -m pip install -r requirements.txt

# Linux / Raspberry Pi / Jetson Nano
python3.12 -m pip install -r requirements.txt
```

### 3. Patch DeepSORT *(Python 3.12 Compatibility)*
`deep-sort-realtime` relies on `pkg_resources` which was updated in Python 3.12. Run the patch once:
```bash
py -3.12 patch_deepsort.py
# Expected output: Patched OK
```

### 4. Download YOLOv8 Weights
```bash
py -3.12 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mkdir -p models
mv yolov8n.pt models/
```

---

## 💻 Usage

### Running the Edge Counter
```bash
# Local test video
py -3.12 main.py --source data/videos/test.mp4 --stop Kharouba

# Live RTSP IP Camera stream
py -3.12 main.py --source rtsp://192.168.1.100:554/stream1 --stop Salamandre --headless

# Webcam with debug overlays
py -3.12 main.py --source 0 --stop "Gare SNTF" --debug

# High-performance headless mode without ReID
py -3.12 main.py --source 0 --stop Kharouba --headless --no-reid
```

### Launching the Monitoring Dashboard
```bash
py -3.12 -m streamlit run dashboard/app.py
```
Access the dashboard at `http://localhost:8501`.

---

## 📁 Project Structure

```
smart-count-tramway/
├── .github/
│   └── workflows/
│       └── ci.yml             # Automated CI pipeline (linting, tests)
├── config/
│   └── settings.yaml          # Configurable parameters (camera, model, lines)
├── dashboard/
│   ├── app.py                 # Streamlit real-time monitoring dashboard
│   └── __init__.py
├── data/                      # SQLite databases, test videos, and logs
├── docs/
│   └── HARDWARE.md            # Hardware benchmarks and Jetson/RPi pinouts
├── models/                    # YOLOv8 weight files (.pt)
├── scripts/
│   └── push_updates.py        # Safe Git sync utility
├── src/
│   ├── __init__.py
│   ├── config_loader.py       # YAML configuration parser with CLI overrides
│   ├── database.py            # SQLite schema, WAL mode, and thread-safe CRUD
│   ├── detection.py           # YOLOv8 inference + on-device face anonymization
│   ├── reid.py                # MobileNetV2 appearance extractor & deduplication
│   └── tracking.py            # DeepSORT tracking + virtual line crossing logic
├── tests/                     # Automated pytest test suite
│   ├── test_config.py
│   ├── test_database.py
│   ├── test_reid.py
│   └── test_tracking.py
├── Dockerfile                 # Multi-stage edge-optimized container
├── docker-compose.yml         # Compose stack for worker + dashboard
├── main.py                    # Master execution loop
├── patch_deepsort.py          # Python 3.12 compatibility patch
├── requirements.txt           # Verified Python 3.12 dependencies
└── TEAM_GUIDE.md              # Team collaboration guidelines
```

---

## 🔒 Privacy-by-Design

All face data is anonymized **on-device at the edge** before any storage or transmission using Gaussian blur (kernel 99×99) applied immediately after capture. Raw face pixels never leave memory, strictly satisfying the privacy requirements defined in §3.2 of the Cahier des Charges.

---

## 🧪 Testing

Run the test suite with `pytest`:
```bash
py -3.12 -m pytest -v
```

---

## 🎓 Academic Defense

This project serves as the **PFE (Projet de Fin d'Études)** for the Start-up track at the University of Mostaganem in partnership with **SETRAM Mostaganem**.

*SETRAM Mostaganem · Université de Mostaganem · 2024–2026*
