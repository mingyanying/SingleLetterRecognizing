import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
import json
import os

# ReLU activation
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Softmax for output
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
    return exps / np.sum(exps, axis=1, keepdims=True)

# Cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
    loss = -np.sum(y_true * np.log(y_pred)) / n_samples
    return loss

# CNN class
class CNN:
    def __init__(self, input_shape=(28, 28), num_filters=4, filter_size=3, pool_size=2, hidden_size=32, output_size=26):
        self.input_shape = input_shape
        self.input_channels = 1
        self.conv_filters = np.random.randn(num_filters, self.input_channels, filter_size, filter_size) * 0.01
        self.conv_bias = np.zeros(num_filters)  # Scalar per filter
        self.conv_output_shape = (input_shape[0] - filter_size + 1, input_shape[1] - filter_size + 1)  # 26x26
        self.pool_size = pool_size
        self.pool_output_shape = (self.conv_output_shape[0] // pool_size, self.conv_output_shape[1] // pool_size)  # 13x13
        self.flatten_size = num_filters * self.pool_output_shape[0] * self.pool_output_shape[1]  # 676 with 4 filters
        self.weights_hidden = np.random.randn(self.flatten_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros(output_size)

    def convolve(self, X):
        batch_size = X.shape[0]
        conv_output = np.zeros((batch_size, self.conv_filters.shape[0], *self.conv_output_shape))
        for b in range(batch_size):
            for f in range(self.conv_filters.shape[0]):
                for i in range(self.conv_output_shape[0]):
                    for j in range(self.conv_output_shape[1]):
                        patch = X[b, 0, i:i+3, j:j+3]
                        conv_output[b, f, i, j] = np.sum(patch * self.conv_filters[f]) + self.conv_bias[f]
        return relu(conv_output)

    def max_pool(self, X):
        batch_size, channels, height, width = X.shape
        out_height, out_width = self.pool_output_shape
        pooled = np.zeros((batch_size, channels, out_height, out_width))
        self.pool_mask = np.zeros_like(X)  # For backprop
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        patch = X[b, c, i*2:i*2+2, j*2:j*2+2]
                        pooled[b, c, i, j] = np.max(patch)
                        max_idx = np.argmax(patch)
                        self.pool_mask[b, c, i*2 + max_idx//2, j*2 + max_idx%2] = 1
        return pooled

    def forward(self, X):
        X = X.reshape(-1, self.input_channels, *self.input_shape)
        self.conv_out = self.convolve(X)
        self.pool_out = self.max_pool(self.conv_out)
        self.flat_out = self.pool_out.reshape(X.shape[0], -1)
        self.hidden = relu(np.dot(self.flat_out, self.weights_hidden) + self.bias_hidden)
        self.output = softmax(np.dot(self.hidden, self.weights_output) + self.bias_output)
        return self.output

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        X = X.reshape(-1, self.input_channels, *self.input_shape)
        
        # Output layer
        d_output = self.output - y  # Simplified for softmax + cross-entropy
        d_weights_output = np.dot(self.hidden.T, d_output) / m
        d_bias_output = np.sum(d_output, axis=0) / m
        
        # Hidden layer
        d_hidden = np.dot(d_output, self.weights_output.T) * relu_derivative(self.hidden)
        d_weights_hidden = np.dot(self.flat_out.T, d_hidden) / m
        d_bias_hidden = np.sum(d_hidden, axis=0) / m
        
        # Flatten to pool
        d_flat = np.dot(d_hidden, self.weights_hidden.T)
        d_pool = d_flat.reshape(self.pool_out.shape)
        
        # Pool to conv
        d_conv = np.zeros_like(self.conv_out)
        for b in range(m):
            for c in range(self.pool_out.shape[1]):
                for i in range(self.pool_output_shape[0]):
                    for j in range(self.pool_output_shape[1]):
                        d_conv[b, c, i*2:i*2+2, j*2:j*2+2] += d_pool[b, c, i, j] * self.pool_mask[b, c, i*2:i*2+2, j*2:j*2+2]
        d_conv *= relu_derivative(self.conv_out)
        
        # Conv layer
        d_filters = np.zeros_like(self.conv_filters)
        d_conv_bias = np.zeros_like(self.conv_bias)
        for b in range(m):
            for f in range(self.conv_filters.shape[0]):
                for i in range(self.conv_output_shape[0]):
                    for j in range(self.conv_output_shape[1]):
                        patch = X[b, 0, i:i+3, j:j+3]
                        d_filters[f] += d_conv[b, f, i, j] * patch
                d_conv_bias[f] = np.sum(d_conv[b, f])
        d_filters /= m
        d_conv_bias /= m

        # Update weights
        self.weights_output -= learning_rate * d_weights_output
        self.bias_output -= learning_rate * d_bias_output
        self.weights_hidden -= learning_rate * d_weights_hidden
        self.bias_hidden -= learning_rate * d_bias_hidden
        self.conv_filters -= learning_rate * d_filters
        self.conv_bias -= learning_rate * d_conv_bias

    def train(self, X, y, epochs, learning_rate, batch_size=32):
        n_samples = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                y_pred_batch = self.forward(X_batch)
                batch_loss = cross_entropy_loss(y_batch, y_pred_batch)
                print(f"Epoch {epoch}, Batch {start//batch_size}, Loss: {batch_loss:.4f}")
                self.backward(X_batch, y_batch, learning_rate)
            
            y_pred_full = self.forward(X)
            loss = cross_entropy_loss(y, y_pred_full)
            print(f"Epoch {epoch}, Full Loss: {loss:.4f}")
            
            if loss < 0.05:
                print(f"Early stopping at epoch {epoch}, Loss: {loss:.4f}")
                break

    def predict(self, X):
        return self.forward(X)

    def save_model(self, file_path):
        model_data = {
            "conv_filters": self.conv_filters.tolist(),
            "conv_bias": self.conv_bias.tolist(),
            "weights_hidden": self.weights_hidden.tolist(),
            "bias_hidden": self.bias_hidden.tolist(),
            "weights_output": self.weights_output.tolist(),
            "bias_output": self.bias_output.tolist()
        }
        with open(file_path, "w") as f:
            json.dump(model_data, f)

    def load_model(self, file_path):
        with open(file_path, "r") as f:
            model_data = json.load(f)
            self.conv_filters = np.array(model_data["conv_filters"])
            self.conv_bias = np.array(model_data["conv_bias"])
            self.weights_hidden = np.array(model_data["weights_hidden"])
            self.bias_hidden = np.array(model_data["bias_hidden"])
            self.weights_output = np.array(model_data["weights_output"])
            self.bias_output = np.array(model_data["bias_output"])

# Global variables
canvas_size = (100, 100)
input_size = (28, 28)
draw_color = 0
background_color = 255
image = Image.new("L", canvas_size, background_color)
draw = ImageDraw.Draw(image)
data_collected = []
model_file = "CNN_model.json"
dataset_path = r"C:\Local Documents\SingleLetterRecognizing\dataset"  # Raw string for Windows path
n = CNN(input_shape=input_size, num_filters=4, filter_size=3, pool_size=2, hidden_size=32, output_size=26)

# Ensure dataset folder exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    for letter in range(26):
        os.makedirs(os.path.join(dataset_path, chr(ord('A') + letter)))

# Model compatibility check
def is_cnn_model_compatible(file_path):
    if not os.path.exists(file_path):
        return False
    with open(file_path, "r") as f:
        model_data = json.load(f)
        required_keys = {"conv_filters", "conv_bias", "weights_hidden", "bias_hidden", "weights_output", "bias_output"}
        return all(key in model_data for key in required_keys)

if is_cnn_model_compatible(model_file):
    n.load_model(model_file)
    print("CNN model loaded successfully!")
else:
    print("No compatible CNN model found. Starting fresh.")
    if os.path.exists(model_file):
        os.remove(model_file)
        print(f"Deleted incompatible {model_file}.")

# UI functions
def draw_on_canvas(event):
    x, y = event.x, event.y
    draw.ellipse([x, y, x + 5, y + 5], fill=draw_color)
    canvas.create_oval(x, y, x + 5, y + 5, fill="black")

def augment_image(image, num_augmentations=10):
    augmented_images = []
    for _ in range(num_augmentations):
        angle = np.random.uniform(-15, 15)
        scale = np.random.uniform(0.9, 1.1)
        translate_x = np.random.randint(-5, 6)
        translate_y = np.random.randint(-5, 6)
        augmented = image.copy()
        augmented = augmented.rotate(angle, fillcolor=background_color)
        augmented = augmented.resize((int(canvas_size[0] * scale), int(canvas_size[1] * scale)), resample=Image.BILINEAR)
        augmented = augmented.crop((translate_x, translate_y, canvas_size[0] + translate_x, canvas_size[1] + translate_y))
        augmented = augmented.resize(canvas_size)
        noise = np.random.normal(0, 10, (canvas_size[1], canvas_size[0]))
        augmented_array = np.array(augmented) + noise
        augmented_array = np.clip(augmented_array, 0, 255).astype('uint8')
        augmented = Image.fromarray(augmented_array)
        augmented = augmented.convert("RGB")
        augmented = augmented.crop((5, 5, canvas_size[0]-5, canvas_size[1]-5))
        augmented_images.append(augmented)
    return augmented_images

def save_image():
    global data_collected
    img_resized = image.resize(input_size)
    binary_image = np.array(img_resized).flatten() / 255.0
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
    letter_folder = os.path.join(dataset_path, label)
    file_count = len(os.listdir(letter_folder))
    original_file = os.path.join(letter_folder, f"{file_count + 1}.png")
    img_resized.save(original_file)
    augmented_images = augment_image(img_resized)
    for i, aug_img in enumerate(augmented_images, start=1):
        aug_file = os.path.join(letter_folder, f"{file_count + 1}_{i}.png")
        aug_img.save(aug_file)
    print(f"Original and augmented images saved for letter '{label}'.")

def load_dataset():
    data = []
    for letter in range(26):
        letter_folder = os.path.join(dataset_path, chr(ord('A') + letter))
        for file_name in os.listdir(letter_folder):
            image_path = os.path.join(letter_folder, file_name)
            img = Image.open(image_path).convert("L")
            img_resized = img.resize(input_size)
            binary_image = np.array(img_resized).flatten() / 255.0
            if binary_image.shape != (784,):
                print(f"Error: Image {image_path} has inconsistent shape.")
                continue
            data.append((binary_image, letter))
    return data

def train_model():
    global data_collected
    data_collected.extend(load_dataset())
    if len(data_collected) == 0:
        print("No data collected yet!")
        return
    
    X = np.array([x[0] for x in data_collected])
    y = np.array([x[1] for x in data_collected]).reshape(-1, 1)
    y_one_hot = np.zeros((y.size, 26))
    y_one_hot[np.arange(y.size), y.flatten()] = 1
    
    print("X shape:", X.shape)
    print("y_one_hot shape:", y_one_hot.shape)
    
    n.train(X, y_one_hot, epochs=1000, learning_rate=0.5, batch_size=32)
    n.save_model(model_file)
    print("Training complete and model saved!")

def predict_letter():
    img_resized = image.resize(input_size)
    binary_image = np.array(img_resized).flatten() / 255.0
    prediction = n.predict(binary_image.reshape(1, -1))
    predicted_label = chr(np.argmax(prediction) + ord('A'))
    print("Predicted letter:", predicted_label)
    result_label.config(text=f"Prediction: {predicted_label}")

def clear_canvas():
    global image, draw
    canvas.delete("all")
    image = Image.new("L", canvas_size, background_color)
    draw = ImageDraw.Draw(image)

# Tkinter UI setup
root = tk.Tk()
root.title("Handwriting Recognition Pad")
canvas = tk.Canvas(root, width=canvas_size[0], height=canvas_size[1], bg="white")
canvas.grid(row=0, column=0, columnspan=4)
canvas.bind("<B1-Motion>", draw_on_canvas)
input_label = tk.Label(root, text="Enter Letter:")
input_label.grid(row=1, column=0)
input_box = tk.Entry(root)
input_box.grid(row=1, column=1)
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
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.grid(row=3, column=0, columnspan=4)
root.mainloop()