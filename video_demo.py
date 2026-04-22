import cv2
import time
import threading
import speech_recognition as sr
from ultralytics import YOLO
import serial # NEW: This lets Python talk to Arduino

# --- 1. ARDUINO CONNECTION ---
# IMPORTANT: Change this string to match the Port you found in the Arduino IDE!
arduino_port = '/dev/cu.usbmodem111401'
 

try:
    arduino = serial.Serial(arduino_port, 9600, timeout=1)
    print("✅ Successfully connected to Arduino!")
    time.sleep(2) # Give Arduino a second to wake up
except:
    arduino = None
    print("⚠️ WARNING: Arduino not found. Please check the USB port name.")

# --- 2. GLOBAL SYSTEM STATES ---
emergency_mode = False

# --- 3. BACKGROUND MICROPHONE LISTENER ---
def listen_for_emergency():
    global emergency_mode
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("\n[SYSTEM] Mic active. Say 'Emergency' to halt, 'Resume' to clear.")
    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=2, phrase_time_limit=3)
            text = recognizer.recognize_google(audio).lower()
            if "emergency" in text: emergency_mode = True
            elif "resume" in text: emergency_mode = False
        except: pass

threading.Thread(target=listen_for_emergency, daemon=True).start()

# --- 4. AI & VIDEO SETUP ---
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('kigali_traffic.mp4')
vehicle_classes = [2, 3, 5, 7]
lanes = ["North", "East", "South", "West"]
current_green_index = 0
green_start_time = time.time()
MAX_GREEN_TIME = 5

while cap.isOpened():
    if not emergency_mode:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = cap.read()

    height, width, _ = frame.shape
    mid_y, mid_x = int(height/2), int(width/2)
    display_frame = frame.copy()

    cv2.line(display_frame, (mid_x, 0), (mid_x, height), (0, 255, 255), 3)
    cv2.line(display_frame, (0, mid_y), (width, mid_y), (0, 255, 255), 3)

    results = model(display_frame, classes=vehicle_classes, conf=0.45, verbose=False)
    annotated_frame = results[0].plot()

    counts = {"North": 0, "East": 0, "South": 0, "West": 0}
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        if cx < mid_x and cy < mid_y: counts["North"] += 1
        elif cx >= mid_x and cy < mid_y: counts["East"] += 1
        elif cx >= mid_x and cy >= mid_y: counts["South"] += 1
        else: counts["West"] += 1

    # --- 5. DECISION ENGINE & ARDUINO CONTROL ---
    if emergency_mode:
        display_text = "🚨 EMERGENCY OVERRIDE: ALL RED 🚨"
        color = (0, 0, 255)
        # Tell Arduino to turn on the RED light
        if arduino: arduino.write(b'R') 
    else:
        current_lane = lanes[current_green_index]
        time_elapsed = time.time() - green_start_time
        if counts[current_lane] == 0 or time_elapsed > MAX_GREEN_TIME:
            for i in range(1, 5):
                check_index = (current_green_index + i) % 4
                if counts[lanes[check_index]] > 0:
                    current_green_index = check_index
                    green_start_time = time.time()
                    break
        current_lane = lanes[current_green_index]
        time_left = max(0, int(MAX_GREEN_TIME - (time.time() - green_start_time)))
        
        if sum(counts.values()) == 0:
            display_text = "SYSTEM STANDBY: 0 VEHICLES"
            color = (0, 255, 255) # Yellow on screen
            # Tell Arduino to turn on the YELLOW light
            if arduino: arduino.write(b'Y')
        else:
            display_text = f"GREEN LIGHT -> {current_lane} ({time_left}s)"
            color = (0, 255, 0)
            # Tell Arduino to turn on the GREEN light
            if arduino: arduino.write(b'G')

    # --- 6. PRESENTATION DASHBOARD (UI) ---
    cv2.rectangle(annotated_frame, (0, 0), (width, 120), (0, 0, 0), -1)
    counts_text = f"N: {counts['North']} | E: {counts['East']} | S: {counts['South']} | W: {counts['West']}"
    cv2.putText(annotated_frame, counts_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(annotated_frame, display_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 4)

    cv2.imshow("Smart Traffic - Arduino Integration", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('e'): emergency_mode = True
    elif key == ord('r'): emergency_mode = False

if arduino: arduino.close()
cap.release()
cv2.destroyAllWindows()
