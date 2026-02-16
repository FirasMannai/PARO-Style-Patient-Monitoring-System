This `README.md` is designed to be visually engaging and professional, reflecting the features and technical logic found in your `code.py`.

---

# 🎙️ PARO-Style Patient Monitoring System 🧘‍♂️

An advanced Python-based recording station that bridges an **ESP32-C6** (via BLE) with a Raspberry Pi. It captures real-time audio, calculates decibel levels, tracks physical posture using AI, and provides instant visual feedback via an RGB LED system.

---

## ✨ Features

* **🎧 Real-time Audio Processing**: Streams 16kHz audio via BLE, calculates RMS decibels, and saves both raw and processed data.
* **🧍 AI Pose Tracking**: Integrated **MediaPipe** support to track 33 body landmarks via a PiCamera, logging data to JSONL for later analysis.
* **🚨 Intelligent LED Status**: A dedicated RGB LED system provides immediate feedback on connection, recording, and battery health.
* **🔋 Power Monitoring**: Monitors ESP32 voltage levels; automatically triggers a "Battery Low" visual override if voltage drops below **2.0V**.
* **💾 Automatic Session Management**: Creates timestamped folders and generates `manifest.json` files for every recording session.
* **🔄 Smart Reconnect**: Features an **Auto-Resume** mode to maintain session continuity if the BLE connection drops.

---

## 🚦 LED Status Guide

The system uses a 3-pin RGB LED to communicate the current state:

| Color | Pattern | Meaning |
| --- | --- | --- |
| 🔵 **Blue** | Solid | **Idle**: Waiting for the first button press. |
| 🟡 **Yellow** | Blinking | **Searching**: Scanning for the ESP32C6-PARO device. |
| 🟢 **Green** | Solid | **Connected**: Device found, waiting for user confirmation. |
| 🔴 **Red** | Solid | **Recording**: Audio data is actively being saved. |
| 🟣 **Violet** | Solid | **Battery Low**: ESP32 voltage is critically low (< 2.0V). |
| ⚪ **White** | Solid | **Error**: Device not found during scan. |

---

## 🛠️ Hardware Setup

* **Controller**: Raspberry Pi (with Bluetooth support).
* **Microcontroller**: ESP32-C6 (Broadcasting as `ESP32C6-PARO`).
* **Peripherals**:
* **Button**: Connected to **BCM 12** (GND toggle).
* **RGB LED**: Pins **BCM 23** (R), **24** (G), **25** (B).
* **Camera**: Official PiCamera (optional, for Pose Tracking).



---

## 🚀 Getting Started

### 1. Install Dependencies

Ensure you have the necessary system libraries for `lgpio` and `opencv`:

```bash
sudo apt update
sudo apt install python3-lgpio
pip install bleak numpy scipy opencv-python mediapipe picamera2

```

### 2. Configuration

Open `code.py` to toggle optional features:

* `CAMERA_ENABLED = True` (Enable/Disable MediaPipe)
* `SHOW_PREVIEW = False` (Set to `True` to see the camera window)

### 3. Execution

Run the script with Python 3:

```bash
python3 code.py

```

---

## 🕹️ User Controls

The system operates on a **Single-Button Toggle** logic (BCM12 to GND):

1. **Press 1**: Start searching for the ESP32.
2. **Press 2**: Confirm connection and start recording/files.
3. **Press 3**: Stop recording, close files, and convert Raw audio to `.wav`.

---

## 📁 Data Output Structure

All data is saved to `/Downloads/Muslim Messungen/` in a timestamped folder:

* `Gesamt_raw_daten.raw/wav`: Continuous audio stream.
* `Paro_raw_daten.raw/wav`: Audio filtered for significant sound levels.
* `Strom_Spannung_Leistung.csv`: Power telemetry (mA, V, mW).
* `pose_TIMESTAMP.jsonl`: Landmark coordinates from MediaPipe.
* `manifest.json`: Session metadata (start/end times, versions, stats).

