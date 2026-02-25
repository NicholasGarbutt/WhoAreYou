import cv2
import face_recognition


def run_camera():
    # Open webcam (0 = default camera)
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FPS,60)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 192)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 144)

    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Camera started. Press 'q' to quit.")

    while True:
        # Grab frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert BGR → RGB (face_recognition expects RGB)
        rgb_frame = frame[:, :, ::-1]

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)

        # Draw boxes
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Show frame
        cv2.imshow("WhoAreYou - Face Detection", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()