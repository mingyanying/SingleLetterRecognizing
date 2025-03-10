import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
import json
import os

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Loss function: Mean Squared Error
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        d_a2 = self.a2 - y
        d_z2 = d_a2 * sigmoid_derivative(self.a2)
        d_weights2 = np.dot(self.a1.T, d_z2) / m
        d_bias2 = np.sum(d_z2, axis=0, keepdims=True) / m

        d_a1 = np.dot(d_z2, self.weights2.T)
        d_z1 = d_a1 * sigmoid_derivative(self.a1)
        d_weights1 = np.dot(X.T, d_z1) / m
        d_bias1 = np.sum(d_z1, axis=0, keepdims=True) / m

        self.weights1 -= learning_rate * d_weights1
        self.bias1 -= learning_rate * d_bias1
        self.weights2 -= learning_rate * d_weights2
        self.bias2 -= learning_rate * d_bias2

    # Function to get the loss
    def get_loss(self, X, y):
        y_pred = self.forward(X)
        return mse_loss(y, y_pred)
        

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = mse_loss(y, y_pred)
            if loss < 0.005:
                break
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        

    def predict(self, X):
        return self.forward(X)

    def save_model(self, file_path):
        model_data = {
            "weights1": self.weights1.tolist(),
            "bias1": self.bias1.tolist(),
            "weights2": self.weights2.tolist(),
            "bias2": self.bias2.tolist()
        }
        with open(file_path, "w") as f:
            json.dump(model_data, f)

    def load_model(self, file_path):
        with open(file_path, "r") as f:
            model_data = json.load(f)
            self.weights1 = np.array(model_data["weights1"])
            self.bias1 = np.array(model_data["bias1"])
            self.weights2 = np.array(model_data["weights2"])
            self.bias2 = np.array(model_data["bias2"])

# Initialize global variables
canvas_size = (100, 100)  # Canvas dimensions (larger canvas for drawing)
input_size = (28, 28)  # Neural network input size (fixed at 28x28)
draw_color = 0  # Black
background_color = 255  # White
image = Image.new("L", canvas_size, background_color)
draw = ImageDraw.Draw(image)
data_collected = []
model_file = "saved_model.json"
dataset_path = "dataset"
n = NeuralNetwork(input_size=784, hidden_size=128, output_size=26)

# Ensure the dataset folder exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    for letter in range(26):
        os.makedirs(os.path.join(dataset_path, chr(ord('A') + letter)))

# Load model if it exists
if os.path.exists(model_file):
    n.load_model(model_file)
    print("Model loaded successfully!")
else:
    print("No saved model found. Starting fresh.")

# Function to draw on the canvas
def draw_on_canvas(event):
    x, y = event.x, event.y
    draw.ellipse([x, y, x + 5, y + 5], fill=draw_color)
    canvas.create_oval(x, y, x + 5, y + 5, fill="black")

# Function to augment the image
def augment_image(image, num_augmentations=10):
    augmented_images = []
    for _ in range(num_augmentations):
        # Random transformations
        angle = np.random.uniform(-15, 15)  # Rotate by a random angle
        scale = np.random.uniform(0.9, 1.1)  # Random scaling
        translate_x = np.random.randint(-5, 6)  # Random translation (x-axis)
        translate_y = np.random.randint(-5, 6)  # Random translation (y-axis)

        # Apply transformations
        augmented = image.copy()
        augmented = augmented.rotate(angle, fillcolor=background_color)
        augmented = augmented.resize(
            (int(canvas_size[0] * scale), int(canvas_size[1] * scale)),
            resample=Image.BILINEAR
        )
        augmented = augmented.crop((translate_x, translate_y, canvas_size[0] + translate_x, canvas_size[1] + translate_y))
        augmented = augmented.resize(canvas_size)  # Restore to original size

        # Add noise
        noise = np.random.normal(0, 10, (canvas_size[1], canvas_size[0]))
        augmented_array = np.array(augmented) + noise
        augmented_array = np.clip(augmented_array, 0, 255).astype('uint8')
        augmented = Image.fromarray(augmented_array)

        # Fill the black slivers with white (background color)
        augmented = augmented.convert("RGB")
        augmented = augmented.crop((5, 5, canvas_size[0]-5, canvas_size[1]-5))  # Optional crop to remove black slivers

        augmented_images.append(augmented)
    return augmented_images

# Function to save the drawn image into the dataset
def save_image():
    global data_collected
    img_resized = image.resize(input_size)  # Resize to 28x28 for neural network input
    binary_image = np.array(img_resized).flatten() / 255.0  # Normalize to [0, 1]
    
    # Ensure binary_image has the correct shape (784,)
    if binary_image.shape != (784,):
        print("Error: Image shape is not consistent.")
        return

    label = input_box.get().strip().upper()
    if len(label) != 1 or not label.isalpha():
        print("Please enter a single valid letter.")
        return
    label_index = ord(label) - ord('A')
    data_collected.append((binary_image, label_index))
    print("Image saved with label:", label)

    # Save the image in the dataset folder
    letter_folder = os.path.join(dataset_path, label)
    file_count = len(os.listdir(letter_folder))
    original_file = os.path.join(letter_folder, f"{file_count + 1}.png")
    img_resized.save(original_file)

    # Generate and save augmented images
    augmented_images = augment_image(img_resized)
    for i, aug_img in enumerate(augmented_images, start=1):
        aug_file = os.path.join(letter_folder, f"{file_count + 1}_{i}.png")
        aug_img.save(aug_file)
    print(f"Original and augmented images saved for letter '{label}'.")

# Function to load dataset for training
def load_dataset():
    data = []
    for letter in range(26):
        letter_folder = os.path.join(dataset_path, chr(ord('A') + letter))
        for file_name in os.listdir(letter_folder):
            image_path = os.path.join(letter_folder, file_name)
            img = Image.open(image_path).convert("L")
            img_resized = img.resize(input_size)  # Resize to 28x28 for neural network input
            binary_image = np.array(img_resized).flatten() / 255.0  # Normalize to [0, 1]
            
            # Ensure binary_image has the correct shape (784,)
            if binary_image.shape != (784,):
                print(f"Error: Image {image_path} has inconsistent shape.")
                continue
            
            data.append((binary_image, letter))
    return data

# Function to train the neural network
def train_model():
    global data_collected
    data_collected.extend(load_dataset())
    if len(data_collected) == 0:
        print("No data collected yet!")
        return

    X = np.array([x[0] for x in data_collected])
    y = np.array([x[1] for x in data_collected]).reshape(-1, 1)
    y_one_hot = np.zeros((y.size, 26))  # One-hot encode labels for 26 letters
    y_one_hot[np.arange(y.size), y.flatten()] = 1

    n.train(X, y_one_hot, epochs=1000, learning_rate=0.1)
    n.save_model(model_file)
    print("Training complete and model saved!")

# Function to predict a letter
def predict_letter():
    img_resized = image.resize(input_size)  # Resize to 28x28 for neural network input
    binary_image = np.array(img_resized).flatten() / 255.0  # Normalize to [0, 1]
    prediction = n.predict(binary_image.reshape(1, -1))
    predicted_label = chr(np.argmax(prediction) + ord('A'))
    print("Predicted letter:", predicted_label) 
    result_label.config(text=f"Prediction: {predicted_label}")

def print_loss():
    global data_collected
    data_collected.extend(load_dataset())
    if len(data_collected) == 0:
        print("No data collected yet!")
        return

    X = np.array([x[0] for x in data_collected])
    y = np.array([x[1] for x in data_collected]).reshape(-1, 1)
    y_one_hot = np.zeros((y.size, 26))  # One-hot encode labels for 26 letters
    y_one_hot[np.arange(y.size), y.flatten()] = 1

    loss = n.get_loss(X, y_one_hot)
    print("loss: ", loss)

# Function to clear the canvas
def clear_canvas():
    global image, draw
    canvas.delete("all")
    image = Image.new("L", canvas_size, background_color)
    draw = ImageDraw.Draw(image)

# Create the Tkinter window
root = tk.Tk()
root.title("Handwriting Recognition Pad")

# Create a canvas for drawing
canvas = tk.Canvas(root, width=canvas_size[0], height=canvas_size[1], bg="white")
canvas.grid(row=0, column=0, columnspan=4)
canvas.bind("<B1-Motion>", draw_on_canvas)

# Input box for letter label
input_label = tk.Label(root, text="Enter Letter:")
input_label.grid(row=1, column=0)
input_box = tk.Entry(root)
input_box.grid(row=1, column=1)

# Buttons for saving, training, predicting, and clearing
btn_save = tk.Button(root, text="Save Image", command=save_image)
btn_save.grid(row=1, column=2)
root.bind("<Control-s>", lambda event: save_image())

btn_train = tk.Button(root, text="Train Model", command=train_model)
btn_train.grid(row=2, column=0)
root.bind("<Control-t>", lambda event: train_model())

btn_predict = tk.Button(root, text="Predict", command=predict_letter)
btn_predict.grid(row=2, column=1)
root.bind("<Control-p>", lambda event: predict_letter())

btn_clear = tk.Button(root, text="Clear Canvas", command=clear_canvas)
btn_clear.grid(row=2, column=2)
root.bind("<Control-c>", lambda event: clear_canvas())

# Label to display prediction result
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.grid(row=4, column=0, columnspan=4)

# Button to get the loss
btn_get_loss = tk.Button(root, text="Get Loss", command=print_loss)
btn_get_loss.grid(row=3, column=1)
root.bind("<Control-l>", lambda event: print_loss())

root.mainloop()
