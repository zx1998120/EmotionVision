import cv2
from deepface import DeepFace
import argparse
import os


# ============================================================
# Analyze emotion from a face image
# ============================================================
def analyze_emotion(face_img):
    try:
        result = DeepFace.analyze(
            img_path=face_img,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv"
        )

        # DeepFace may return a list
        if isinstance(result, list):
            result = result[0]

        dominant_emotion = result["dominant_emotion"]
        emotion_scores = result["emotion"]

        return dominant_emotion, emotion_scores

    except Exception as e:
        print("Emotion analysis error:", e)
        return None, None


# ============================================================
# Image Mode
# ============================================================
def image_mode(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    image = cv2.imread(image_path)

    if image is None:
        print("Could not read image.")
        return

    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        print("No face detected.")
        return

    for (x, y, w, h) in faces:

        # Crop face
        face_img = image[y:y+h, x:x+w]

        emotion, scores = analyze_emotion(face_img)

        if emotion is not None:

            # Draw face box
            cv2.rectangle(
                image,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            # Draw emotion
            cv2.putText(
                image,
                emotion,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            print("\nDetected emotion:", emotion)

            print("Emotion probabilities:")

            for emotion_name, score in scores.items():
                print(f"{emotion_name:10s}: {score:.2f}%")

    cv2.imshow("Emotion Recognition", image)

    print("\nPress any key to exit.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Camera Mode
# ============================================================
def camera_mode():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera.")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    print("Camera started.")
    print("Press Q to quit.")

    frame_count = 0

    # Store last emotion result
    last_emotion = "Detecting..."

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Cannot receive frame.")
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:

            face_img = frame[y:y+h, x:x+w]

            # Do not run neural network every frame
            # Analyze every 10 frames
            if frame_count % 10 == 0:

                emotion, scores = analyze_emotion(face_img)

                if emotion is not None:
                    last_emotion = emotion

            # Draw face box
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            # Draw emotion
            cv2.putText(
                frame,
                last_emotion,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        frame_count += 1

        cv2.imshow(
            "Real-Time Emotion Recognition",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Facial Emotion Recognition Demo"
    )

    parser.add_argument(
        "--mode",
        choices=["camera", "image"],
        default="camera",
        help="Choose camera or image mode"
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Image path for image mode"
    )

    args = parser.parse_args()

    if args.mode == "camera":

        camera_mode()

    elif args.mode == "image":

        if args.image is None:
            print(
                "Please provide image path:\n"
                "python emotion_demo.py "
                "--mode image --image test.jpg"
            )
        else:
            image_mode(args.image)