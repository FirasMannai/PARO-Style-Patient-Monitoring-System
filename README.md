PARO BLE Monitoring & Recording System
Overview

This project implements a Raspberry Pi 5–based monitoring and data recording system developed for Project 1 – PARO in postoperative delirium therapy.

The system communicates via Bluetooth Low Energy (BLE) with an ESP32 integrated in the PARO robot and records:

Audio signals

Motor telemetry data

Camera-based pose tracking data

The design focuses on reliable data acquisition, simple user interaction, and safe storage for clinical study environments.

System Architecture

The Raspberry Pi 5 serves as the central processing unit:

BLE communication with ESP32 (audio + telemetry)

RGB LED for system status indication

Single hardware button for control

Raspberry Pi Camera Module for real-time pose detection

Automatic structured data logging

Features

BLE discovery, connection, and notification handling

One-button workflow (Start → Confirm → Stop)

Clear visual feedback via RGB LED

Automatic session folder creation with timestamps

Real-time pose estimation using MediaPipe

Immediate disk synchronization using os.fsync()

Automatic conversion of raw audio data to WAV format

Designed to run as a systemd service

Hardware Requirements

Raspberry Pi 5

ESP32 (integrated in PARO robot)

Raspberry Pi Camera Module

RGB LED

GPIO23 → Red

GPIO24 → Green

GPIO25 → Blue

Push Button

GPIO12 (BCM) → GND

SD card (approx. 220 MB per 20-minute session)

Software Components

Bleak – BLE communication with ESP32

NumPy – Audio processing (RMS and dB calculation)

SciPy – Conversion of raw audio to WAV

lgpio – GPIO control (button + RGB LED)

OpenCV – Image processing and visualization

MediaPipe – Pose detection and landmark tracking

threading – Parallel execution of camera processing

asyncio – Non-blocking BLE communication

Data Logging

Each measurement session creates a structured folder containing:

Raw audio data (.raw)

Sound level data (.csv)

Motor telemetry (.csv)

Pose landmarks (.jsonl)

Session metadata (manifest.json)

Converted audio files (.wav)

All critical files are flushed and synchronized to disk using os.fsync() to prevent data loss in case of unexpected power interruption.

Camera Pose Tracking

The camera runs in a dedicated background thread to ensure non-blocking operation.

Resolution: 640 × 480

Frame rate: 15 FPS

MediaPipe Pose detects body landmarks

Each frame logs:

x, y, z coordinates

visibility

absolute timestamp

relative session time

Pose data is stored in JSONL format for later analysis.

System Service

The application can run automatically at boot using a systemd service configuration.

Example:

[Unit]
Description=ESP32C6 PARO Recorder Service
After=bluetooth.service network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Downloads
ExecStart=/usr/bin/python3 /home/pi/Downloads/code.py
Restart=always
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target

Purpose

This system provides a robust and reproducible data acquisition platform for clinical research involving the PARO robot.
It ensures structured data storage, clear system feedback, and reliable long-term operation.