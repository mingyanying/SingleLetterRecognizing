import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
import json
import os

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLU
def relu_derivative(x):
    return (x > 0).astype(float)

# Softmax activation function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]

# CNN class
class ConvolutionalNeuralNetwork:
    def __init__(self, input_size, num_filters, filter_size, hidden_size, output_size):
        self.input_size = input_size
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize filters and weights
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.01
        self.weights_fc = np.random.randn((input_size - filter_size + 1)**2 * num_filters, hidden_size) * 0.01
        self.bias_fc = np.zeros((1, hidden_size))
        self.weights_out = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_out = np.zeros((1, output_size))

    def convolve(self, X):
        batch_size, height, width = X.shape
        conv_output = np.zeros((batch_size, self.num_filters, height - self.filter_size + 1, width - self.filter_size + 1))

        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(height - self.filter_size + 1):
                    for j in range(width - self.filter_size + 1):
                        region = X[b, i:i+self.filter_size, j:j+self.filter_size]
                        conv_output[b, f, i, j] = np.sum(region * self.filters[f])

        return relu(conv_output)

    def forward(self, X):
        self.conv_output = self.convolve(X)
        flattened = self.conv_output.reshape(X.shape[0], -1)
        self.fc_output = relu(np.dot(flattened, self.weights_fc) + self.bias_fc)
        self.output = softmax(np.dot(self.fc_output, self.weights_out) + self.bias_out)
        return self.output

    def backward(self, X, y_true, learning_rate):
        batch_size = X.shape[0]
        d_output = self.output - y_true
        d_weights_out = np.dot(self.fc_output.T, d_output) / batch_size
        d_bias_out = np.sum(d_output, axis=0, keepdims=True) / batch_size

        d_fc_output = np.dot(d_output, self.weights_out.T) * relu_derivative(self.fc_output)
        d_weights_fc = np.dot(self.conv_output.reshape(batch_size, -1).T, d_fc_output) / batch_size
        d_bias_fc = np.sum(d_fc_output, axis=0, keepdims=True) / batch_size

        self.weights_out -= learning_rate * d_weights_out
        self.bias_out -= learning_rate * d_bias_out
        self.weights_fc -= learning_rate * d_weights_fc
        self.bias_fc -= learning_rate * d_bias_fc

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = cross_entropy_loss(y, y_pred)
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def save_model(self, file_path):
        model_data = {
            "filters": self.filters.tolist(),
            "weights_fc": self.weights_fc.tolist(),
            "bias_fc": self.bias_fc.tolist(),
            "weights_out": self.weights_out.tolist(),
            "bias_out": self.bias_out.tolist()
        }
        with open(file_path, "w") as f:
            json.dump(model_data, f)

    def load_model(self, file_path):
        with open(file_path, "r") as f:
            model_data = json.load(f)
            self.filters = np.array(model_data["filters"])
            self.weights_fc = np.array(model_data["weights_fc"])
            self.bias_fc = np.array(model_data["bias_fc"])
            self.weights_out = np.array(model_data["weights_out"])
            self.bias_out = np.array(model_data["bias_out"])

# Global variables
canvas_size = (28, 28)
background_color = 255
image = Image.new("L", canvas_size, background_color)
draw = ImageDraw.Draw(image)
data_collected = []
model_file = "cnn_model.json"
cnn = ConvolutionalNeuralNetwork(input_size=28, num_filters=8, filter_size=3, hidden_size=128, output_size=26)

# Load model if it exists
if os.path.exists(model_file):
    cnn.load_model(model_file)
    print("CNN model loaded successfully!")
else:
    print("No saved CNN model found. Starting fresh.")

# Tkinter functions
def draw_on_canvas(event):
    x, y = event.x, event.y
    draw.rectangle([x, y, x + 1, y + 1], fill=0)
    canvas.create_rectangle(x, y, x + 1, y + 1, fill="black")

def save_image():
    img_resized = image.resize(canvas_size)
    binary_image = np.array(img_resized).flatten() / 255.0
    label = input_box.get().strip().upper()
    if len(label) != 1 or not label.isalpha():
        print("Enter a valid letter.")
        return
    label_index = ord(label) - ord('A')
    data_collected.append((binary_image, label_index))
    print("Image saved with label:", label)

def train_model():
    global data_collected
    if not data_collected:
        print("No data collected yet!")
        return

    X = np.array([x[0] for x in data_collected]).reshape(-1, 28, 28)
    y = np.array([x[1] for x in data_collected])
    y_one_hot = np.zeros((y.size, 26))
    y_one_hot[np.arange(y.size), y] = 1

    cnn.train(X, y_one_hot, epochs=1000, learning_rate=0.01)
    cnn.save_model(model_file)
    print("Training complete and model saved!")

def predict_letter():
    img_resized = image.resize(canvas_size)
    binary_image = np.array(img_resized).flatten() / 255.0
    X = binary_image.reshape(1, 28, 28)
    prediction = cnn.forward(X)
    predicted_label = chr(np.argmax(prediction) + ord('A'))
    print("Predicted letter:", predicted_label)
    result_label.config(text=f"Prediction: {predicted_label}")

def clear_canvas():
    global image, draw
    canvas.delete("all")
    image = Image.new("L", canvas_size, background_color)
    draw = ImageDraw.Draw(image)

# Tkinter UI
root = tk.Tk()
root.title("Handwriting Recognition Pad")

canvas = tk.Canvas(root, width=canvas_size[0] * 10, height=canvas_size[1] * 10, bg="white")
canvas.grid(row=0, column=0, columnspan=4)
canvas.bind("<B1-Motion>", draw_on_canvas)

input_label = tk.Label(root, text="Enter Letter:")
input_label.grid(row=1, column=0)
input_box = tk.Entry(root)
input_box.grid(row=1, column=1)

btn_save = tk.Button(root, text="Save Image", command=save_image)
btn_save.grid(row=1, column=2)
btn_train = tk.Button(root, text="Train Model", command=train_model)
btn_train.grid(row=2, column=0)
btn_predict = tk.Button(root, text="Predict", command=predict_letter)
btn_predict.grid(row=2, column=1)
btn_clear = tk.Button(root, text="Clear Canvas", command=clear_canvas)
btn_clear.grid(row=2, column=2)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.grid(row=3, column=0, columnspan=4)

root.mainloop()
