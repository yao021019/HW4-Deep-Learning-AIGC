# Handwriting Recognition App

This project is a handwriting recognition application that can recognize digits drawn by a user. It includes a from-scratch neural network implementation, a training script, a `tkinter` GUI, and a `streamlit` web application.

## Final Model Accuracy
The final accuracy of the model on the MNIST test set is **98.16%**.

## File Structure
- `model.py`: Contains the `NeuralNetwork` class, which is a from-scratch implementation of a 3-hidden-layer neural network.
- `train.py`: The script to load the MNIST dataset, preprocess it, train the neural network model, and save the trained weights to `model.npz`.
- `app.py`: A desktop GUI application built with `tkinter` that allows users to draw a digit and get a prediction from the model.
- `streamlit_app.py`: A web application built with `streamlit` that provides a web-based interface for the handwriting recognition model.
- `requirements.txt`: A list of the required Python libraries to run the project.
- `model.npz`: The saved weights and biases of the trained model.
- `promtWithAns.md`: A log of the conversation and actions taken to build this project.
- `visualize_data.py`: A helper script to visualize digits from the MNIST dataset.

## How to Run

### 1. Install Dependencies
First, make sure you have Python installed. Then, install the required libraries using pip:
```bash
pip install -r requirements.txt
```

### 3. Run the Web App (streamlit)
To run the `streamlit` based web application, execute the following command in your terminal:
```bash
streamlit run streamlit_app.py
```
This will open a new tab in your web browser with the handwriting recognition app.

You can also access the deployed Streamlit app directly at: [https://hw4-deep-learning-aigc-2g27aaoibmgvwp3id4zcvq.streamlit.app/](https://hw4-deep-learning-aigc-2g27aaoibmgvwp3id4zcvq.streamlit.app/)