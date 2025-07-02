# 🔥 Automated-Fire-Alert-System

A real-time fire and smoke detection system using YOLOv8, OpenCV, and Python. This project also integrates sound alerts, popup notifications, and SMS alerts using Twilio for immediate response.

## Features
🔍 Fire and smoke detection using YOLOv8
📡 Live IP webcam stream support
📢 Sound alert on detection (alert.wav file)
📱 SMS alert via Twilio API
⚠️ Popup window for desktop alert using Tkinter
🧠 Dual YOLO model detection (best.pt + yolov8n.pt)

## Tech Stack
YOLOv8 (Ultralytics)
OpenCV
Twilio (for SMS)
Pygame (for sound)
Tkinter (for popup)
Python 3.10+

## Setup
1. Install dependencies
2. Configure `.env` with Twilio keys
3. Run `fire_stream.py`