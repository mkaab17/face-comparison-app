import streamlit as st
from deepface import DeepFace
import tempfile
import os

st.set_page_config(
    page_title="Face Comparison App",
    layout="centered"
)

st.title("🧠 Face Comparison")
st.caption("Check whether two face images belong to the same person")

img1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
img2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])

if img1 and img2:
    st.image([img1, img2], caption=["Image 1", "Image 2"], width=220)

    if st.button("Compare Faces"):
        with st.spinner("Analyzing faces..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f1:
                f1.write(img1.read())
                img1_path = f1.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f2:
                f2.write(img2.read())
                img2_path = f2.name

            result = DeepFace.verify(
                img1_path=img1_path,
                img2_path=img2_path,
                model_name="ArcFace",
                enforce_detection=False
            )

            os.remove(img1_path)
            os.remove(img2_path)

        verified = result["verified"]
        distance = result["distance"]
        threshold = result["threshold"]

        # Confidence estimation (clean & intuitive)
        confidence = max(0.0, (1 - distance / threshold)) * 100
        confidence = round(confidence, 2)

        st.divider()

        if verified:
            st.success("✅ Same Person")
        else:
            st.error("❌ Different Person")

        st.metric("Confidence", f"{confidence}%")
        st.caption(f"Model used: {result['model']}")

