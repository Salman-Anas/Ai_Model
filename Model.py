import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("best.pt")

st.title("🧠 YOLOv8 Object Detection")

# File uploader for image input
uploaded_file = st.file_uploader("🖼️ Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to NumPy array for processing
    img_array = np.array(image)

    # Run inference with YOLOv8
    results = model(img_array)

    # Plot results using Ultralytics' built-in visualization
    res_plotted = results[0].plot()

    # Show the output image with detections
    st.image(res_plotted, caption="Detection Result", use_column_width=True)
