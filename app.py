import os
import numpy as np
import streamlit as st
from pdf2image import convert_from_path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# Set page config
st.set_page_config(page_title="MarineDerma", layout="centered")

# Apply CSS for styling
st.markdown(
    """
    <style>
    /* Light blue background */
    .stApp {
        background-color: #ADD8E6;
    }

    /* Center titles */
    .centered {
        text-align: center;
        color: #0B3D91;  /* Dark blue text for contrast */
    }

    /* Yellow prediction box */
    .prediction-box {
        background-color: #FFFACD;  /* Light yellow */
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    /* Bold and slightly larger text inside prediction box */
    .prediction-box p {
        font-weight: bold;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
MODEL_PATH = "my_model.keras"
model = load_model(MODEL_PATH)
st.success("✅ MobileNetV3 Model loaded successfully!")

# Classes
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
    return preds

# Get top 3 predictions
def get_top_predictions(preds, classes, top=3):
    top_indices = preds[0].argsort()[-top:][::-1]
    return [(classes[i], preds[0][i]*100) for i in top_indices]

# Extract images from PDF
def extract_images_from_pdf(pdf_file):
    images = convert_from_path(pdf_file)
    image_paths = []
    for i, img in enumerate(images):
        img_path = f"temp_page_{i}.jpg"
        img.save(img_path, "JPEG")
        image_paths.append(img_path)
    return image_paths

# UI
# Display logo
st.image("logo.JPG", width=150)  # Adjust width as needed

# Main title and subtitle
st.markdown('<h1 class="centered">CIRM SERVIZI Srl, Roma</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="centered">🧑‍⚕️ MarineDerma - Skin Disease Detection App</h3>', unsafe_allow_html=True)
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
            st.image(img_path, caption=f"Page {idx+1}", use_container_width=True)
            preds = model_predict(img_path, model)
            top_preds = get_top_predictions(preds, classes, top=3)

            # Display top 3 predictions
            html_str = '<div class="prediction-box">'
            html_str += f"<p>Top 3 Predictions (Page {idx+1}):</p>"
            for cls, score in top_preds:
                html_str += f"<p>{cls} - {score:.2f}%</p>"
            html_str += "</div>"
            st.markdown(html_str, unsafe_allow_html=True)
    else:
        st.image(file_path, caption="Uploaded Image", use_container_width=True)
        preds = model_predict(file_path, model)
        top_preds = get_top_predictions(preds, classes, top=3)

        html_str = '<div class="prediction-box">'
        html_str += "<p>Top 3 Predictions:</p>"
        for cls, score in top_preds:
            html_str += f"<p>{cls} - {score:.2f}%</p>"
        html_str += "</div>"
        st.markdown(html_str, unsafe_allow_html=True)
