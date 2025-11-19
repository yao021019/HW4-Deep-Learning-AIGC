import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
from model import NeuralNetwork
from train import train_model

# Load the trained model
@st.cache_resource
def load_model():
    # Ensure model.npz exists
    if not os.path.exists('model.npz'):
        st.write("Training model... This might take a few minutes.")
        train_model()

    input_size = 784
    output_size = 10
    n1 = 128
    n2 = 64
    n3 = 32
    
    model = NeuralNetwork(input_size, n1, n2, n3, output_size)
    model.load_model('model.npz')
    return model

model = load_model()

# App title and instructions
st.title("Handwriting Recognition")
st.write("Draw a digit from 0 to 9 on the canvas below and click 'Predict' to see the model's prediction.")

# Create the drawable canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Get the image data from the canvas
        img_data = canvas_result.image_data

        # Convert to a PIL image
        img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
        img = img.convert('L')

        # Preprocess the image
        bbox = img.getbbox()
        if bbox:
            cropped_image = img.crop(bbox)
            cropped_image.thumbnail((20, 20), Image.Resampling.LANCZOS)
            new_image = Image.new("L", (28, 28), "black")
            paste_x = (28 - cropped_image.width) // 2
            paste_y = (28 - cropped_image.height) // 2
            new_image.paste(cropped_image, (paste_x, paste_y))
            img_array = np.array(new_image).astype('float32') / 255.0
            img_array = img_array.reshape(1, -1)

            # Make a prediction
            probabilities = model.predict_proba(img_array)[0]
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_probs = probabilities[top_3_indices]
            
            # Display the top prediction
            st.write("## Top Prediction")
            st.metric(label="Predicted Digit", value=top_3_indices[0])

            # Display the top 3 predictions with probabilities
            st.write("### Top 3 Predictions")
            chart_data = pd.DataFrame({
                "Digit": [str(i) for i in top_3_indices],
                "Probability": top_3_probs,
            })
            st.bar_chart(chart_data.set_index("Digit"))

        else:
            st.write("Please draw a digit on the canvas.")
    else:
        st.write("Please draw a digit on the canvas.")
