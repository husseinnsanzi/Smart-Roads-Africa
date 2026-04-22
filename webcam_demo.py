import cv2
from ultralytics import YOLO

# 1. Load the AI model
model = YOLO('yolov8n.pt')
# 2. Set up the webcam (0 is your Mac camera)
cap = cv2.VideoCapture(0)

vehicle_classes = [2, 3, 5, 7] # Cars, motorcycles, buses, trucks
confidence_level = 0.60
congestion_threshold = 2

print("Starting Webcam Test... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model(frame, classes=vehicle_classes, conf=confidence_level)
    vehicle_count = len(results[0].boxes)
    annotated_frame = results[0].plot()

    if vehicle_count >= congestion_threshold:
        status_text = "Status: CONGESTED - Extending Green Light!"
        text_color = (0, 0, 255) # Red
    else:
        status_text = "Status: Normal Traffic"
        text_color = (0, 255, 0) # Green

    cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(annotated_frame, status_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    cv2.imshow("Smart Traffic - Congestion Logic", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
