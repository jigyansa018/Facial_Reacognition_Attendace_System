# Required installations:
# pip install cmake
# pip install face_recognition
# pip install opencv-python
# pip install numpy
# pip install pillow

import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
from PIL import Image

# ---------- Function to load and encode a face safely ----------
def load_and_encode(image_path, person_name):
    try:
        # Load image and ensure it’s RGB
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Detect and encode face
        encodings = face_recognition.face_encodings(image_np)
        if len(encodings) == 0:
            print(f"[WARNING] No face found in image: {image_path}")
            return None
        return encodings[0]

    except Exception as e:
        print(f"[ERROR] Could not process {person_name}'s image ({image_path}): {e}")
        return None

# ---------- Load Known Faces ----------
isha_encoding = load_and_encode("faces/isha.jpg", "Isha")
nisha_encoding = load_and_encode("faces/nisha.jpg", "Nisha")

# Filter out any None encodings (in case face not detected)
known_face_encodings = [enc for enc in [isha_encoding, nisha_encoding] if enc is not None]
known_face_names = []
if isha_encoding is not None:
    known_face_names.append("Isha")
if nisha_encoding is not None:
    known_face_names.append("Nisha")


if not known_face_encodings:
    print("No valid faces loaded. Check your image paths and lighting.")
    exit()

# ---------- Webcam Setup ----------
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not access webcam.")
    exit()

# ---------- CSV Setup ----------
students = known_face_names.copy()
current_date = datetime.now().strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)
lnwriter.writerow(["Name", "Time"])

print("Facial Recognition Attendance System Started")
print("Press 'q' to stop...")

# ---------- Main Loop ----------
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error capturing frame.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        name = ""
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Display name and mark attendance
        if name in known_face_names:
            cv2.putText(frame, f"{name} Present", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (255, 0, 0), 3, 2)

            # Mark attendance once
            if name in students:
                students.remove(name)
                lnwriter.writerow([name, datetime.now().strftime("%H:%M:%S")])
                f.flush()
                print(f"Marked {name} present at {datetime.now().strftime('%H:%M:%S')}")

    # Display
    cv2.imshow("Attendance", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(" Exiting...")
        break

# ---------- Cleanup ----------
video_capture.release()
cv2.destroyAllWindows()
f.close()
