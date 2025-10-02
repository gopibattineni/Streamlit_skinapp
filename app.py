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

# Function to get top 3 predictions
def get_top_predictions(preds, classes, top=3):
    top_indices = preds[0].argsort()[-top:][::-1]
    top_classes = [(classes[i], preds[0][i]*100) for i in top_indices]
    return top_classes

if uploaded_file is not None:
    file_path = os.path.join("uploads", uploaded_file.name)
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.markdown("<h2 style='text-align:center'>CIRMSERVIZI</h2>", unsafe_allow_html=True)

    if uploaded_file.name.lower().endswith(".pdf"):
        st.info("📄 PDF detected, extracting images...")
        image_paths = extract_images_from_pdf(file_path)
        for idx, img_path in enumerate(image_paths):
            st.image(img_path, caption=f"Page {idx+1}", use_container_width=True)
            img = image.load_img(img_path, target_size=(224,224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = model.predict(x)
            
            top_preds = get_top_predictions(preds, classes, top=3)
            
            # Display top 3 predictions with yellow background
            html_str = "<div style='background-color:yellow; padding:10px; border-radius:5px;'>"
            html_str += f"<strong>Top 3 Predictions (Page {idx+1}):</strong><br>"
            for cls, score in top_preds:
                html_str += f"{cls} - {score:.2f}%<br>"
            html_str += "</div>"
            st.markdown(html_str, unsafe_allow_html=True)
    else:
        st.image(file_path, caption="Uploaded Image", use_container_width=True)
        img = image.load_img(file_path, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        
        top_preds = get_top_predictions(preds, classes, top=3)
        
        html_str = "<div style='background-color:yellow; padding:10px; border-radius:5px;'>"
        html_str += "<strong>Top 3 Predictions:</strong><br>"
        for cls, score in top_preds:
            html_str += f"{cls} - {score:.2f}%<br>"
        html_str += "</div>"
        st.markdown(html_str, unsafe_allow_html=True)

