import os
import cv2
import numpy as np
import streamlit as st
from pdf2image import convert_from_path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

# Load a pre-trained MobileNetV3 model (fine-tuned)
MODEL_PATH = "Streamlit_skinapp/my_model.pth"

# Define classes (update according to your dataset)
classes = ["Actinic keratosis", "Basal cell carcinoma", "Benign keratosis", 
           "Chickenpox", "Cowpox", "Dermatofibroma", "HFMD", "Healthy", 
           "Measles", "Melanocytic nevus", "Melanoma", "Monkeypox", 
           "Squamous cell carcinoma", "Vascular lesion"]

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.mobilenet_v3_large(weights=None)  
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))  
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

st.success("✅ MobileNetV3 Model (PyTorch) loaded successfully!")

# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prediction function
def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(img)
        pred_class_index = torch.argmax(preds, dim=1).item()
        pred_score = torch.softmax(preds, dim=1).max().item() * 100
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
