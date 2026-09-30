import os
import cv2
import face_recognition
from datetime import datetime
import pandas as pd

# 1. Configuration & Setup
faces_dir = "faces"
attendance_file = "attendance.csv"
logged_today = set()

# Automatically create faces folder if missing
os.makedirs(faces_dir, exist_ok=True)

# 2. Load and encode database faces
known_encodings = []
known_names = []

print("🚀 Loading reference database...")
files = os.listdir(faces_dir)

if len(files) == 0:
    print("\n⚠️ The 'faces' folder is EMPTY!")
    print("👉 Action: Put a portrait photo in the 'faces' folder and restart.")
    exit()

for file_name in files:
    if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(faces_dir, file_name)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) > 0:
            # Take the first face encoding found in the photo
            known_encodings.append(encodings[0])
            # Strip extension for the name (e.g., "John_Doe.jpg" -> "John_Doe")
            known_names.append(os.path.splitext(file_name)[0])

print(f"✅ Database loaded successfully. Tracked individuals: {known_names}")

# 3. Initialize MacBook Webcam
print("\n🎥 Opening webcam... (Press 'q' in the window to quit)")
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Failed to grab frame from webcam.")
        break

    # Convert BGR (OpenCV default) to RGB (face_recognition requirement)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face positions and extract mathematical embeddings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare current webcam face with database vectors
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

            # Log to CSV if they haven't been captured yet during this session
            if name not in logged_today:
                now = datetime.now()
                time_str = now.strftime("%H:%M:%S")
                date_str = now.strftime("%Y-%m-%d")
                
                df = pd.DataFrame([{"Name": name, "Date": date_str, "Time": time_str}])
                df.to_csv(attendance_file, mode='a', header=not os.path.exists(attendance_file), index=False)
                
                print(f"📝 Attendance logged for: {name} at {time_str}")
                logged_today.add(name)

        # Draw visual bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display video in a native macOS UI popup window
    cv2.imshow('Smart Attendance System', frame)

    # Stop the program if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up system processes
video_capture.release()
cv2.destroyAllWindows()
print("👋 System shut down safely.")
