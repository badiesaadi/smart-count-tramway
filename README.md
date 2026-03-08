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
| Detection | YOLOv8 (Ultralytics) |
| Tracking | DeepSORT |
| Privacy | OpenCV Haar Cascade (Gaussian blur) |
| Backend | Python 3.8+, PyTorch, OpenCV |
| Dashboard | Streamlit + Plotly |
| Storage | SQLite |
| Hardware | Raspberry Pi 4 / Jetson Nano |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/smart-count-tramway.git
cd smart-count-tramway
pip install -r requirements.txt
```

Download YOLOv8 nano weights (auto-downloads on first run):
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt models/
```

---

## Usage

### Run the counter
```bash
# Test with a video file
python main.py --source data/videos/test.mp4 --stop Kharouba

# Live RTSP stream
python main.py --source rtsp://192.168.1.100:554/stream1 --stop Salamandre

# Headless (no display — for Pi/Jetson)
python main.py --source 0 --stop "Gare SNTF" --headless
```

### Launch the dashboard
```bash
streamlit run dashboard/app.py
```

---

## Project Structure

```
smart-count-tramway/
├── main.py                  # Main execution loop
├── requirements.txt
├── src/
│   ├── detection.py         # YOLOv8 + face blurring
│   ├── tracking.py          # DeepSORT + line crossing logic
│   └── database.py          # SQLite operations
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── models/                  # YOLOv8 weights (.pt files)
├── data/
│   └── videos/              # Test footage
└── scripts/
    └── push_to_github.py    # One-click GitHub deployment
```

---

## Performance Targets

| Metric | Target | Implementation |
|--------|--------|---------------|
| Accuracy | ≥ 95% | YOLOv8n + DeepSORT n_init=3 |
| Processing | < 100ms/frame | FP16 inference, CPU fallback |
| Privacy | At-the-edge | Face blur before storage |

---

## Privacy

All face data is anonymised **on-device** before any storage or transmission, using Gaussian blur (kernel 99×99) applied immediately after capture. Raw face pixels never leave the device.

---

## Academic Defense

This project is the **PFE (Projet de Fin d'Études)** for the Start-up track at the University of Mostaganem.  
Deliverables: Functional prototype + Business Plan + Pitch Deck + Mémoire.

---

*SETRAM Mostaganem · Université de Mostaganem · 2024–2025*
