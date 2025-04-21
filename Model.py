import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("best.pt")

st.title("🧠 YOLOv8 Object Detection")

uploaded_file = st.file_uploader("🖼️ Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to NumPy array
    img_array = np.array(image)

    # Run YOLOv8 inference
    results = model(img_array)
    res_plotted = results[0].plot()

    # Display result
    st.image(res_plotted, caption="Detection Result", use_column_width=True)
