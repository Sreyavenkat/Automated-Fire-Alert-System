from ultralytics import YOLO
import cv2
import pygame
import time
from datetime import datetime
from twilio.rest import Client
import tkinter as tk
from threading import Thread
from dotenv import load_dotenv
import os

load_dotenv()

# === YOLOv8 Model Setup ===
model = YOLO("best.pt")
model1 = YOLO("yolov8n.pt")

# === IP Webcam Stream ===
STREAM_URL = STREAM_URL = os.getenv("STREAM_URL")

# === Alert Settings ===
ALERT_SOUND = "alert.wav"
ALERT_COOLDOWN = 10  # seconds between sound alerts

# === Twilio SMS Setup ===
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
TO_PHONE_NUMBER = os.getenv("TO_PHONE_NUMBER")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# === Function: Show Popup Window (Aesthetic) ===
def show_fire_popup(fire_time):
    def popup():
        root = tk.Tk()
        root.title("🔥 FIRE ALERT")
        root.geometry("400x200")
        root.configure(bg="#fff0f0")
        root.eval('tk::PlaceWindow . center')
        root.attributes('-topmost', True)
        root.resizable(False, False)

        # Border frame
        frame = tk.Frame(root, bg="#ffe5e5", padx=20, pady=20, highlightbackground="red", highlightthickness=2)
        frame.pack(expand=True, fill='both')

        # Message
        title_label = tk.Label(frame, text="🔥 FIRE DETECTED!", font=("Helvetica", 18, "bold"), fg="red", bg="#ffe5e5")
        title_label.pack(pady=(0, 10))

        time_label = tk.Label(frame, text=f"Time: {fire_time}", font=("Helvetica", 14), bg="#ffe5e5", fg="#333")
        time_label.pack()

        # OK Button
        ok_button = tk.Button(
            frame,
            text="OK",
            command=root.destroy,
            font=("Helvetica", 12, "bold"),
            bg="red",
            fg="white",
            relief="flat",
            padx=10,
            pady=5,
            cursor="hand2",
            activebackground="#cc0000"
        )
        ok_button.pack(pady=(20, 0))

        root.mainloop()

    Thread(target=popup).start()

# === Function: Send SMS Alert ===
def send_fire_sms_alert(first_fire_time):
    try:
        message = client.messages.create(
            body=f"🔥 ALERT: Fire detected at {first_fire_time} in the yard. Please check immediately.",
            from_=TWILIO_FROM_NUMBER,
            to=TO_PHONE_NUMBER
        )
        print(f"📩 SMS sent successfully at {first_fire_time}: SID={message.sid}")
    except Exception as e:
        print(f"❌ Failed to send SMS: {e}")

# === Initialize Stream and Sound ===
cap = cv2.VideoCapture(STREAM_URL)
pygame.mixer.init()
sound = pygame.mixer.Sound(ALERT_SOUND)

if not cap.isOpened():
    raise RuntimeError(f"❌ Cannot open stream at {STREAM_URL}")

# === State Tracking ===
last_alert_time = 0
first_fire_time = None
sms_sent = False
popup_shown = False

print("🚨 Fire detection system running...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.sum() == 0:
        print("⚠️ Stream read failed or empty frame.")
        continue

    # YOLOv8 Detection
    results = model(frame)
    results1 = model1(frame)

    # Annotated output
    annotated_frame = results[0].plot()
    annotated_frame1 = results1[0].plot()
    combined_frame = cv2.addWeighted(annotated_frame1, 0.5, annotated_frame, 0.5, 0)
    cv2.imshow("🔥 Hybrid Fire Detection", combined_frame)

    # Fire Detection Flag
    fire_detected = False

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id].lower()
        if "fire" in class_name:
            fire_detected = True
            break

    if fire_detected:
        now = time.time()

        if first_fire_time is None:
            first_fire_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"🕒 First fire detected at: {first_fire_time}")

        if not sms_sent:
            send_fire_sms_alert(first_fire_time)
            sms_sent = True

        if now - last_alert_time > ALERT_COOLDOWN:
            print("🔥 FIRE DETECTED! Playing alert sound.")
            sound.play()
            last_alert_time = now

        if not popup_shown:
            show_fire_popup(first_fire_time)
            popup_shown = True

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows() 




