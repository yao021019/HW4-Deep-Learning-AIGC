import tkinter as tk
from tkinter import Canvas, Button, Label
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from model import NeuralNetwork

class HandwritingApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("Handwriting Recognition")
        self.model = model

        # Create the canvas for drawing
        self.canvas = Canvas(root, width=280, height=280, bg="black", highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        # Create a PIL image and a draw object to draw on the image
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)

        # Create the buttons
        self.predict_button = Button(root, text="Predict", command=self.predict)
        self.predict_button.pack(pady=5)

        self.clear_button = Button(root, text="Clear", command=self.clear)
        self.clear_button.pack(pady=5)

        # Create the label to display the prediction
        self.prediction_label = Label(root, text="Prediction: ", font=("Helvetica", 16))
        self.prediction_label.pack(pady=10)

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill="white", outline="white")

    def predict(self):
        # Get the bounding box of the drawing
        bbox = self.image.getbbox()

        if bbox:
            # Crop the image to the bounding box
            cropped_image = self.image.crop(bbox)

            # Resize the cropped image to 20x20, maintaining aspect ratio
            cropped_image.thumbnail((20, 20), Image.Resampling.LANCZOS)

            # Create a new 28x28 black image
            new_image = Image.new("L", (28, 28), "black")

            # Paste the 20x20 digit into the center of the 28x28 image
            paste_x = (28 - cropped_image.width) // 2
            paste_y = (28 - cropped_image.height) // 2
            new_image.paste(cropped_image, (paste_x, paste_y))

            # Convert the image to a numpy array
            img_array = np.array(new_image).astype('float32') / 255.0
            img_array = img_array.reshape(1, -1)

            # Make a prediction
            prediction = self.model.predict(img_array)
            self.prediction_label.config(text=f"Prediction: {prediction[0]}")
        else:
            # If the canvas is empty, do nothing
            self.prediction_label.config(text="Prediction: ")

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill="black")
        self.prediction_label.config(text="Prediction: ")

if __name__ == '__main__':
    # Load the trained model
    input_size = 784
    output_size = 10
    n1 = 128
    n2 = 64
    n3 = 32
    
    model = NeuralNetwork(input_size, n1, n2, n3, output_size)
    model.load_model('model.npz')

    # Create and run the Tkinter app
    root = tk.Tk()
    app = HandwritingApp(root, model)
    root.mainloop()
