import cv2
from recognition import load_known_faces, recognize_faces

def run_camera():
    known_encodings, known_names = load_known_faces()

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Camera started. Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        results = recognize_faces(frame, known_encodings, known_names)

        for (top, right, bottom, left), name, confidence in results:

            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            label = f"{name} ({confidence}%)"

            # Draw filled label background
            cv2.rectangle(
                frame,
                (left, top - 35),
                (right, top),
                (0, 255, 0),
                cv2.FILLED
            )

            # Put text above face
            cv2.putText(
                frame,
                label,
                (left + 6, top - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        cv2.imshow("WhoAreYou - Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()