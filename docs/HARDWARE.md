# ⚡ Hardware Guide — Smart Count Tramway
**Power Consumption, Specifications & Deployment Intel**  
*SETRAM Mostaganem — PFE Start-up 2024–2025*

---

## 1. Supported Hardware Platforms

The system supports three deployment targets, each suited for a different stage of the project:

| Platform | Stage | GPU | RAM | Price |
|----------|-------|-----|-----|-------|
| AMD Ryzen PC / Laptop | Development & Testing | Optional (NVIDIA) | 8–16 GB | Already owned |
| Raspberry Pi 4 (4GB) | Budget edge deployment | ❌ None | 4 GB | ~50€ |
| NVIDIA Jetson Nano (4GB) | Production deployment | ✅ 128-core Maxwell | 4 GB | ~150€ |

---

## 2. Power Consumption — Detailed Breakdown

### 2.1 Raspberry Pi 4 (4GB)

| State | Power Draw | Notes |
|-------|-----------|-------|
| Idle (OS only) | 2.7 W | No USB devices |
| Running main.py (CPU only) | 5.1 W | YOLOv8n inference on CPU |
| Peak load (detection + tracking) | 6.4 W | Both cores fully loaded |
| With USB camera attached | +0.5 W | USB 2.0 webcam |
| With active cooling fan | +0.3 W | Fan at full speed |
| **Typical operating power** | **~6.0 W** | Normal counting session |

**Power supply required:** USB-C, 5V / 3A (15W) — official Raspberry Pi PSU recommended.  
**Annual energy cost** (24/7 operation): 6W × 24h × 365 = **52.6 kWh/year ≈ 630 DZD/year**

---

### 2.2 NVIDIA Jetson Nano (4GB)

The Jetson Nano has two power modes selectable via `nvpmodel`:

| Power Mode | Power Draw | CPU Cores | GPU | Use Case |
|------------|-----------|-----------|-----|----------|
| 5W mode (mode 1) | 5 W max | 2 cores @ 918 MHz | 128 cores @ 460 MHz | Battery / low power |
| 10W mode (mode 0) | 10 W max | 4 cores @ 1.4 GHz | 128 cores @ 921 MHz | Full performance |

**Recommended for SETRAM:** 10W mode for maximum FPS.

| State | Power Draw | FPS achieved |
|-------|-----------|-------------|
| Idle | 1.5 W | — |
| 5W mode — YOLOv8n inference | 4.5 W | ~8 FPS |
| 10W mode — YOLOv8n inference | 8.5 W | ~15 FPS |
| 10W mode — TensorRT optimised | 9.2 W | ~25 FPS |
| Peak (all cores + GPU loaded) | 10 W | — |

**Power supply required:** DC barrel jack, 5V / 4A (20W) — or USB-C 5V/3A for 5W mode only.  
**Annual energy cost** (24/7, 10W mode): 10W × 24h × 365 = **87.6 kWh/year ≈ 1050 DZD/year**

**Switch power modes on Jetson:**
```bash
# Set to maximum performance (10W):
sudo nvpmodel -m 0
sudo jetson_clocks

# Set to power saving (5W):
sudo nvpmodel -m 1
```

---

### 2.3 Development PC / Laptop (AMD Ryzen)

| Component | Typical Power | Notes |
|-----------|--------------|-------|
| Ryzen 5 CPU (inference only) | 35–65 W TDP | Full load during YOLOv8 |
| NVIDIA GPU (if present) | 75–150 W TDP | Only if CUDA enabled |
| RAM (16 GB DDR4) | 3–5 W | |
| SSD storage | 2–4 W | |
| **Total system** | **50–250 W** | Much higher than edge devices |

---

## 3. Power Consumption Comparison Chart

```
Power Draw During Active Counting Session:

Raspberry Pi 4    ████░░░░░░░░░░░░░░░░  6 W
Jetson Nano (5W)  ████░░░░░░░░░░░░░░░░  4.5 W
Jetson Nano (10W) ████████░░░░░░░░░░░░  8.5 W
Ryzen Laptop      ████████████████████  50–80 W
Desktop PC        ████████████████████████████████  100–250 W

Scale: each █ = ~2.5W
```

**Conclusion:** Edge devices use **10–40x less power** than a full PC — critical for 24/7 tram stop deployment.

---

## 4. Camera Hardware

### Recommended Camera for Production

| Specification | Requirement | Recommended Model |
|--------------|-------------|------------------|
| Resolution | 720p minimum | 1080p preferred |
| Interface | USB or RTSP | IP camera with RTSP |
| Frame rate | 25 FPS minimum | 30 FPS |
| Field of view | 90°–120° | 110° for door coverage |
| Low light | Required | IR night vision |
| Housing | Weatherproof | IP65 or higher |
| Power | PoE (Power over Ethernet) | Simplifies wiring |

### Power Consumption — IP Cameras

| Camera Type | Power (PoE) | Notes |
|-------------|------------|-------|
| Basic 1080p IP camera | 5–7 W | Standard PoE (IEEE 802.3af) |
| PTZ camera | 15–25 W | Not needed for fixed door mount |
| IR night vision camera | 8–12 W | Required for evening/night operation |
| **Recommended: fixed IR 1080p** | **~8 W** | Good balance of power and quality |

**For SETRAM:** The cameras are already installed — no additional camera cost.  
If new cameras are needed: budget ~5,000–15,000 DZD per camera.

---

## 5. Network Hardware

### Per-Stop Network Requirements

| Device | Purpose | Power |
|--------|---------|-------|
| PoE Switch (8-port) | Powers cameras + connects Pi | 30–60 W total |
| Ethernet cable (Cat5e) | Camera → Switch, Pi → Switch | 0 W |
| WiFi dongle (if no Ethernet) | Pi → Router wirelessly | +1–2 W on Pi |

### Bandwidth Requirements

The system transmits only count numbers — not video:

| Data type | Size per transmission | Frequency | Monthly data |
|-----------|----------------------|-----------|-------------|
| Count record (JSON) | ~200 bytes | Every 30 sec | ~17 MB |
| Log data | ~50 bytes | Per event | ~5 MB |
| **Total per stop** | — | — | **~22 MB/month** |

This is negligible — even a basic 3G mobile data connection is sufficient.

---

## 6. Full System Power Budget — One Tram Stop

| Component | Quantity | Unit Power | Total |
|-----------|----------|-----------|-------|
| Jetson Nano (10W mode) | 1 | 10 W | 10 W |
| IP Camera (IR 1080p) | 1–2 | 8 W | 8–16 W |
| PoE Switch (mini, 4-port) | 1 | 15 W | 15 W |
| Ethernet cables | — | 0 W | 0 W |
| **Total per stop** | | | **33–41 W** |

**For 5 stops on the Mostaganem network:**
- Total power: 33–41W × 5 = **165–205 W**
- Annual energy: ~200W × 8760h = **1752 kWh/year**
- Annual cost (Algeria electricity rate ~4.5 DZD/kWh): **~7,884 DZD/year (~55€/year)**

This is the total running cost for the entire 5-stop network. Extremely low.

---

## 7. Thermal Management

### Raspberry Pi 4
- **Without heatsink:** Throttles at 80°C — performance drops significantly
- **With heatsink only:** Stable up to ~60°C ambient
- **With heatsink + fan:** Stable in all conditions including Algerian summer
- **Recommendation:** Always use a case with heatsink and fan for outdoor deployment

```bash
# Monitor temperature on Pi:
vcgencmd measure_temp
# Output: temp=52.1'C

# If consistently above 70°C, add cooling
```

### Jetson Nano
- Ships with a heatsink — sufficient for most conditions
- Add a 5V fan to the fan header for hot environments (Mostaganem summers reach 40°C+)
- Operating temperature range: -25°C to 80°C (well within Algeria's climate)

```bash
# Monitor temperature on Jetson:
cat /sys/devices/virtual/thermal/thermal_zone*/temp
# Divide by 1000 for Celsius
```

---

## 8. Storage Requirements

| Data | Size per day | Size per month | Size per year |
|------|-------------|----------------|---------------|
| SQLite database (counts) | ~50 KB | ~1.5 MB | ~18 MB |
| Log files | ~200 KB | ~6 MB | ~72 MB |
| OS + software | 8 GB (fixed) | — | — |
| **Total SD card needed** | — | — | **~16 GB minimum** |

**Recommendation:** 32 GB Class 10 microSD for Raspberry Pi, or 32 GB eMMC for Jetson Nano.  
**Backup:** Counts are also synced to the central PostgreSQL server — local storage is just a buffer.

---

## 9. Installation Recommendations for SETRAM

### Physical Mounting
- Mount the Pi/Jetson inside a **weatherproof IP65 enclosure** near the camera
- Use a **DIN rail mount** inside the tram stop electrical cabinet if available
- Ensure ventilation — do not seal the enclosure completely

### Power Supply
- Source power from the existing tram stop electrical infrastructure (220V AC)
- Use a **DIN rail 220V AC → 5V DC converter** (meanwell or similar) — ~500 DZD
- This eliminates the need for external USB power adapters

### Cabling
- Run **Cat5e Ethernet** from the Pi to the IP camera (PoE) and to the local network
- Maximum PoE cable length: 100 metres — sufficient for any tram stop layout
- Label all cables with stop name and camera ID

---

## 10. Quick Reference — Hardware Setup Commands

```bash
# Check Pi CPU temperature:
vcgencmd measure_temp

# Check Jetson power mode:
sudo nvpmodel -q

# Set Jetson to max performance:
sudo nvpmodel -m 0 && sudo jetson_clocks

# Check available disk space:
df -h

# Check RAM usage:
free -h

# Monitor CPU usage in real time:
htop

# Check if camera is detected:
ls /dev/video*

# Test camera feed:
python3 -c "import cv2; cap=cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL'); cap.release()"
```

---

*Smart Count Tramway — Hardware Guide*  
*SETRAM Mostaganem · Université de Mostaganem · 2024–2025*
