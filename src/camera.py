import cv2
import os
import json
from datetime import datetime
import face_recognition
from recognition import load_known_faces, recognize_faces

PROFILES_PATH = "/Users/nicholas/Desktop/WhoAreYou/src/profiles.json"
FACES_DIR = "known_faces"

def run_camera():
    known_encodings, known_names = load_known_faces()

    if os.path.exists(PROFILES_PATH):
        with open(PROFILES_PATH, "r") as f:
            profiles = json.load(f)
    else:
        profiles = {}

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        results = recognize_faces(frame, known_encodings, known_names)

        for (top, right, bottom, left), name, confidence in results:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            label = f"{name} ({confidence}%)"

            cv2.rectangle(
                frame,
                (left, top - 35),
                (right, top),
                (0, 255, 0),
                cv2.FILLED
            )

            cv2.putText(
                frame,
                label,
                (left + 6, top - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

            if name in profiles:
                met_at = profiles[name].get("met_at", "")
                job = profiles[name].get("job", "")
                who = profiles[name].get("who", "")
                DOB = profiles[name].get("DOB", "")

                y_offset = bottom + 20

                cv2.putText(frame, f"Met at: {met_at}", (left, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

                cv2.putText(frame, job, (left, y_offset + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

                cv2.putText(frame, who, (left, y_offset + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

                cv2.putText(frame, DOB, (left, y_offset + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        cv2.imshow("WhoAreYou - AI Memory Assistant", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("n"):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if len(face_locations) == 0:
                continue

            top, right, bottom, left = face_locations[0]

            face_height = bottom - top
            padding = int(face_height * 0.4)

            top = max(0, top - padding)
            bottom = min(frame.shape[0], bottom + padding)
            left = max(0, left - padding)
            right = min(frame.shape[1], right + padding)

            face_image = frame[top:bottom, left:right]
            face_image_resized = cv2.resize(face_image, (300, 300))

            name = input("Name: ").strip()
            if name == "":
                continue

            met_at = input("Where did you meet them? ").strip()
            job = input("What do they do for work? ").strip()
            who = input("Who are they to you? ").strip()
            DOB = input("Date of Birth (DD/MM/YYYY): ").strip()

            person_folder = f"{FACES_DIR}/{name}"
            os.makedirs(person_folder, exist_ok=True)

            image_count = len(os.listdir(person_folder)) + 1
            file_path = f"{person_folder}/{image_count}.jpg"

            cv2.imwrite(file_path, face_image_resized)

            rgb_resized = cv2.cvtColor(face_image_resized, cv2.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(rgb_resized)

            if len(encoding) > 0:
                known_encodings.append(encoding[0])
                known_names.append(name)

                profiles[name] = {
                    "met_at": met_at,
                    "job": job,
                    "who": who,
                    "DOB": DOB
                }

                with open(PROFILES_PATH, "w") as f:
                    json.dump(profiles, f, indent=4)

        elif key == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()