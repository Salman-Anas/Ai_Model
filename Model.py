import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO("Ai_Model\\best.pt")

st.title("🧠 YOLOv8 Object Detection")

option = st.radio("Choose input method:", ("📷 Take Webcam Snippet", "🖼️ Upload an Image"))

if option == "📷 Take Webcam Snippet":
    st.write("Click 'Capture' to take a frame from webcam.")
    capture_button = st.button("Capture")

    if capture_button:
        cap = cv2.VideoCapture(0)  # Open default camera (index 0)

        if not cap.isOpened():
            st.error("Could not open webcam.")
        else:
            ret, frame = cap.read()
            cap.release()

            if ret:
                # Run YOLOv8 inference
                results = model(frame)
                res_plotted = results[0].plot()

                st.image(res_plotted, channels="BGR", caption="Detection from Webcam")
            else:
                st.error("Failed to capture image.")

elif option == "🖼️ Upload an Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = np.array(image)

        results = model(img_array)
        res_plotted = results[0].plot()

        st.image(res_plotted, caption="Detection Result", use_column_width=True)
