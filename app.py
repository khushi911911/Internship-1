import streamlit as st
import Project_1   # import your notebook code (converted into .py)
import numpy as np
import cv2

st.title("😊 Emotion Detection App")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file into OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR")

    # Call your prediction function from Project_1.py
    emotion = Project_1.predict_emotion(img)   # <-- define this in Project_1.py

    st.subheader(f"Detected Emotion: {emotion}")
