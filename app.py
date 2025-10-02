import os
import cv2
import numpy as np
import streamlit as st
from pdf2image import convert_from_path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# Load the trained MobileNetV3 model
MODEL_PATH = "my_model.keras"
model = load_model(MODEL_PATH)
st.success("✅ MobileNetV3 Model loaded successfully!")

# Define class names (update based on your dataset)
classes = ["Actinic keratosis", "Basal cell carcinoma", "Benign keratosis", 
           "Chickenpox", "Cowpox", "Dermatofibroma", "HFMD", "Healthy", 
           "Measles", "Melanocytic nevus", "Melanoma", "Monkeypox", 
           "Squamous cell carcinoma", "Vascular lesion"]

# Prediction function
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    pred_class_index = np.argmax(preds, axis=1)[0]
    pred_score = np.max(preds) * 100
    return pred_class_index, pred_score

# Extract images from PDF
def extract_images_from_pdf(pdf_file):
    images = convert_from_path(pdf_file)
    image_paths = []
    for i, img in enumerate(images):
        img_path = f"temp_page_{i}.jpg"
        img.save(img_path, "JPEG")
        image_paths.append(img_path)
    return image_paths

# Streamlit UI
st.title("🧑‍⚕️ MarineDerma - Skin Disease Detection App")
st.write("Upload an image or PDF report for prediction.")

uploaded_file = st.file_uploader("Upload Image/PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    file_path = os.path.join("uploads", uploaded_file.name)
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.lower().endswith(".pdf"):
        st.info("📄 PDF detected, extracting images...")
        image_paths = extract_images_from_pdf(file_path)
        for idx, img_path in enumerate(image_paths):
            st.image(img_path, caption=f"Page {idx+1}", use_column_width=True)
            pred_class_index, pred_score = model_predict(img_path, model)
            st.markdown(
                f"**Prediction (Page {idx+1}):** {classes[pred_class_index]} "
                f"with **{pred_score:.2f}%** confidence"
            )
    else:
        st.image(file_path, caption="Uploaded Image", use_column_width=True)
        pred_class_index, pred_score = model_predict(file_path, model)
        st.markdown(
            f"**Prediction:** {classes[pred_class_index]} "
            f"with **{pred_score:.2f}%** confidence"
        )
