import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config MUST be here and only once
st.set_page_config(page_title="Deepfake Detector", layout="centered")

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
MODEL_PATH = "model1.keras"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# --------------------------------------------------
# Preprocessing function
# --------------------------------------------------
IMG_SIZE = 32   # change if your model uses different size

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --------------------------------------------------
# UI Design 
# --------------------------------------------------
st.title("🧠 AI Deepfake Image Detector")
st.write("Upload an image to check whether it is **REAL** or **FAKE** using Vision Transformer/CNN model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze Image"):
        st.write("⏳ Analyzing...")

        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0][0]

        # Binary Classification Threshold
        label = "FAKE" if prediction > 0.5 else "REAL"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # --------------------------------------------------
        # Display Results
        # --------------------------------------------------
        st.subheader("🔍 Prediction Result")
        if label == "FAKE":
            st.error("⚠️ The model predicts: **FAKE** image")
        else:
            st.success("✅ The model predicts: **REAL** image")

        st.subheader("📊 Confidence")
        st.progress(float(confidence))
        st.write(f"Confidence Score: **{confidence*100:.2f}%**")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.write("---")
st.write("Developed using Streamlit & TensorFlow")
