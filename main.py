import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
from PIL import Image
import pickle

# Load the pre-trained model
with open('Acute Lymphoblastic Leukemia classification.pkl', 'rb') as file:
    model = pickle.load(file)

# Define label mapping
label_map = {
    0: "all_begien",
    1: "all_early",
    2: "all_pre",
    3: "all_pro"
}

# Initialize ImageDataGenerator for preprocessing
datagen = ImageDataGenerator()

# Streamlit app layout
st.title("Acute Lymphoblastic Leukemia Classification")
st.write("Upload an image to classify")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image = image.resize((224, 224))  
    image = img_to_array(image)  
    image = np.expand_dims(image, axis=0)  
    image = datagen.flow(image, batch_size=1).__next__()  

    # Predict using the model
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_label = label_map[predicted_class_index]

    # Display prediction
    st.write("Predicted Label:", predicted_label)
