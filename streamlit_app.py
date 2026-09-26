import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="EmotionVision",
    page_icon="😊",
    layout="wide"
)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
st.title("😊 EmotionVision")

st.subheader("Real-Time Facial Emotion Recognition")

st.markdown(
    """
    Detect facial expressions using **DeepFace + OpenCV**.

    Supported emotions:

    **Happy · Sad · Angry · Surprise · Fear · Disgust · Neutral**
    """
)

st.divider()


# ---------------------------------------------------------
# Emotion detection function
# ---------------------------------------------------------
def detect_emotion(image):

    # PIL -> NumPy
    image_rgb = np.array(image)

    # RGB -> BGR
    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    results = DeepFace.analyze(
        img_path=image_bgr,
        actions=["emotion"],
        detector_backend="opencv",
        enforce_detection=True
    )

    if not isinstance(results, list):
        results = [results]

    output = image_rgb.copy()

    detected_faces = []

    for result in results:

        emotion = result["dominant_emotion"]
        scores = result["emotion"]
        region = result["region"]

        x = region["x"]
        y = region["y"]
        w = region["w"]
        h = region["h"]

        confidence = scores[emotion]

        # Face rectangle
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        # Label
        label = f"{emotion.upper()} {confidence:.1f}%"

        cv2.putText(
            output,
            label,
            (x, max(y - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        detected_faces.append(
            {
                "emotion": emotion,
                "confidence": confidence,
                "scores": scores
            }
        )

    return output, detected_faces


# ---------------------------------------------------------
# Input mode
# ---------------------------------------------------------
mode = st.radio(
    "Choose input method",
    ["📷 Camera", "🖼️ Upload Image"],
    horizontal=True
)


image = None


# ---------------------------------------------------------
# Camera
# ---------------------------------------------------------
if mode == "📷 Camera":

    camera_image = st.camera_input(
        "Take a picture"
    )

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")


# ---------------------------------------------------------
# Upload
# ---------------------------------------------------------
else:

    uploaded_file = st.file_uploader(
        "Upload a facial image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")


# ---------------------------------------------------------
# Analyze
# ---------------------------------------------------------
if image is not None:

    st.divider()

    if st.button(
        "🔍 Analyze Emotion",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner("Analyzing facial expression..."):

                output_image, faces = detect_emotion(image)

            st.success("Emotion detected!")

            # -------------------------------------------------
            # Result image
            # -------------------------------------------------

            col1, col2 = st.columns([1.4, 1])

            with col1:

                st.subheader("Detection Result")

                st.image(
                    output_image,
                    use_container_width=True
                )

            # -------------------------------------------------
            # Emotion result
            # -------------------------------------------------

            with col2:

                st.subheader("Emotion Analysis")

                for index, face in enumerate(faces):

                    emotion = face["emotion"]
                    confidence = face["confidence"]

                    st.metric(
                        label=f"Face {index + 1}",
                        value=emotion.upper(),
                        delta=f"{confidence:.1f}% confidence"
                    )

                    st.markdown("### Emotion Probabilities")

                    scores = face["scores"]

                    # Sort emotions
                    sorted_scores = sorted(
                        scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    for emotion_name, score in sorted_scores:

                        st.write(
                            f"**{emotion_name.capitalize()}** "
                            f"— {score:.1f}%"
                        )

                        st.progress(
                            min(float(score) / 100.0, 1.0)
                        )

                    st.divider()

        except Exception as e:

            st.error(
                "No face was detected. "
                "Please use a clear, front-facing image."
            )

            st.caption(f"Error: {e}")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "EmotionVision · Facial Emotion Recognition · "
    "Python · DeepFace · OpenCV · Streamlit"
)