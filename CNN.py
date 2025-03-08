import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Global variables
canvas_size = (100, 100)  # Size of the drawing canvas
input_size = (28, 28)     # Size expected by the CNN (matches MNIST-like data)
draw_color = 0            # Black for drawing
background_color = 255    # White background
image = Image.new("L", canvas_size, background_color)  # Grayscale image
draw = ImageDraw.Draw(image)
data_collected = []       # Store drawn images and labels
model_file = "CNN_model.h5"  # File to save the TensorFlow model
dataset_path = "dataset"  # Folder to store dataset images

# Create dataset folders for letters A-Z if they don’t exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    for letter in range(26):
        os.makedirs(os.path.join(dataset_path, chr(ord('A') + letter)))

# Build the CNN model using TensorFlow Keras
def build_model(input_shape=(28, 28, 1), num_classes=26):
    model = models.Sequential([
        layers.Conv2D(4, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize the model
n = build_model()

# Load the model if it exists
if os.path.exists(model_file):
    n = models.load_model(model_file)
    print("Model loaded successfully!")
else:
    print("No saved model found. Starting fresh.")

# Draw on the canvas
def draw_on_canvas(event):
    x, y = event.x, event.y
    draw.ellipse([x, y, x + 5, y + 5], fill=draw_color)
    canvas.create_oval(x, y, x + 5, y + 5, fill="black")

# Augment the image by rotating it
def augment_image(image, num_augmentations=10):
    augmented_images = []
    for _ in range(num_augmentations):
        angle = np.random.uniform(-15, 15)  # Random rotation between -15 and 15 degrees
        image_rotated = image.rotate(angle, fillcolor=background_color)
        augmented_images.append(image_rotated)
    return augmented_images

# Save the drawn image to the dataset
def save_image():
    global data_collected
    img_resized = image.resize(input_size)
    binary_image = np.array(img_resized) / 255.0  # Normalize to [0, 1]
    binary_image = binary_image.reshape(1, 28, 28, 1)  # Reshape for CNN
    
    label = input_box.get().strip().upper()
    if len(label) != 1 or not label.isalpha():
        print("Please enter a single valid letter.")
        return
    label_index = ord(label) - ord('A')  # Convert A-Z to 0-25
    data_collected.append((binary_image, label_index))
    print("Image saved with label:", label)

    # Save original image to dataset folder
    letter_folder = os.path.join(dataset_path, label)
    file_count = len(os.listdir(letter_folder))
    original_file = os.path.join(letter_folder, f"{file_count + 1}.png")
    img_resized.save(original_file)

    # Save augmented images
    augmented_images = augment_image(img_resized)
    for i, aug_img in enumerate(augmented_images, start=1):
        aug_file = os.path.join(letter_folder, f"{file_count + 1}_{i}.png")
        aug_img.save(aug_file)
    print(f"Original and augmented images saved for letter '{label}'.")

# Load the dataset from the folder
def load_dataset():
    data = []
    for letter in range(26):
        letter_folder = os.path.join(dataset_path, chr(ord('A') + letter))
        for file_name in os.listdir(letter_folder):
            image_path = os.path.join(letter_folder, file_name)
            img = Image.open(image_path).convert("L")
            img_resized = img.resize(input_size)
            binary_image = np.array(img_resized) / 255.0
            binary_image = binary_image.reshape(1, 28, 28, 1)
            data.append((binary_image, letter))
    return data

# Train the model
def train_model():
    global data_collected
    data_collected.extend(load_dataset())
    if len(data_collected) == 0:
        print("No data collected yet!")
        return

    X = np.vstack([x[0] for x in data_collected])  # Stack all images
    y = np.array([x[1] for x in data_collected])   # Array of labels
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=26)  # One-hot encoding

    print("X shape:", X.shape)
    print("y_one_hot shape:", y_one_hot.shape)

    # Train the model
    n.fit(X, y_one_hot, epochs=10, batch_size=32, verbose=1)
    n.save(model_file)
    print("Training complete and model saved!")

# Predict the letter from the drawn image
def predict_letter():
    img_resized = image.resize(input_size)
    binary_image = np.array(img_resized) / 255.0
    binary_image = binary_image.reshape(1, 28, 28, 1)
    prediction = n.predict(binary_image)
    predicted_label = chr(np.argmax(prediction) + ord('A'))  # Convert index to letter
    print("Predicted letter:", predicted_label)
    result_label.config(text=f"Prediction: {predicted_label}")

# Clear the canvas
def clear_canvas():
    global image, draw
    canvas.delete("all")
    image = Image.new("L", canvas_size, background_color)
    draw = ImageDraw.Draw(image)

# Set up the Tkinter UI
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