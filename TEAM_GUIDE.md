# 👥 Team Guide — Smart Count Tramway

Quick reference for teammates to get the project running and collaborate via Git.

---

## 1. First Time Setup

### Install Python 3.12
👉 https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe  
During install: **check "Add python.exe to PATH"**

### Install Visual C++ Redistributable
👉 https://aka.ms/vs/17/release/vc_redist.x64.exe  
Just run it and click Install.

### Clone the project
```bash
git clone https://github.com/badiesaadi/smart-count-tramway.git
cd smart-count-tramway
```

### Install dependencies
```bash
py -3.12 -m pip install -r requirements.txt
```

### Patch DeepSORT (run only once)
```bash
py -3.12 patch_deepsort.py
```

### Download YOLOv8 weights (run only once)
```bash
py -3.12 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
move yolov8n.pt models\
```

---

## 2. Run the Project

### Start the passenger counter
```bash
py -3.12 main.py --source 0 --stop Kharouba
```

### Start the dashboard (open a second terminal)
```bash
py -3.12 -m streamlit run dashboard/app.py
```
Then open **http://localhost:8501** in your browser.

---

## 3. Daily Git Workflow

### Before you start working — always pull first
```bash
git pull origin main
```
This downloads the latest changes from your teammates so you don't get conflicts.

---

### After you make changes — push your work

**Step 1 — See what files you changed:**
```bash
git status
```

**Step 2 — Stage your changes:**
```bash
git add .
```
Or stage a specific file only:
```bash
git add src/detection.py
```

**Step 3 — Commit with a clear message:**
```bash
git commit -m "feat: improved face blurring accuracy"
```

**Step 4 — Push to GitHub:**
```bash
git push origin main
```

---

## 4. Commit Message Guide

Use these prefixes so everyone understands what changed:

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature added |
| `fix:` | Bug fix |
| `docs:` | README or documentation update |
| `refactor:` | Code cleanup, no new feature |
| `test:` | Adding or fixing tests |

Examples:
```
feat: add counting line visualisation to dashboard
fix: resolve DeepSORT crash on empty frames
docs: update installation steps for Linux
```

---

## 5. Handling Conflicts

If `git push` is rejected, it means a teammate pushed before you. Fix it with:
```bash
git pull origin main --no-rebase
git push origin main
```

If Git opens a text editor asking for a merge message — just close it by typing:
```
:q
```
Then press Enter, and push again.

---

## 6. Useful Commands Cheatsheet

| Command | What it does |
|---------|-------------|
| `git status` | See which files you changed |
| `git pull origin main` | Download latest changes |
| `git add .` | Stage all changed files |
| `git add <file>` | Stage one specific file |
| `git commit -m "message"` | Save your changes locally |
| `git push origin main` | Upload your changes to GitHub |
| `git log --oneline` | See recent commits |
| `git diff` | See exactly what you changed |

---

## 7. Project Repo

🔗 https://github.com/badiesaadi/smart-count-tramway

---

*Smart Count Tramway · SETRAM Mostaganem · 2024–2025*
