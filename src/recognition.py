import face_recognition
import os
import numpy as np
import cv2

def load_known_faces():
    known_encodings = []
    known_names = []

    base_path = "/Users/nicholas/Desktop/WhoAreYou/known_faces"

    for person_name in os.listdir(base_path):
        person_folder = os.path.join(base_path, person_name)

        if os.path.isdir(person_folder):

            for filename in os.listdir(person_folder):
                if filename.endswith((".jpg", ".png", ".jpeg")):

                    image_path = os.path.join(person_folder, filename)
                    image = face_recognition.load_image_file(image_path)

                    encodings = face_recognition.face_encodings(image)

                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_names.append(person_name)

    print(f"Loaded {len(known_encodings)} total faces.")
    return known_encodings, known_names

def recognize_faces(frame, known_encodings, known_names):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    results = []

    for face_location, face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = "????"
        confidence = 0

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_names[best_match_index]
                distance = face_distances[best_match_index]


                confidence = round((1 - distance) * 100, 2)

        # Scale face location back up
        top, right, bottom, left = face_location
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        results.append(((top, right, bottom, left), name, confidence))

    return results