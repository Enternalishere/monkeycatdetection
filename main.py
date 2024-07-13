import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
import tkinter as tk
from tkinter import filedialog, messagebox

# Step 4: Load and Preprocess Images
def load_and_preprocess_image(file_path, target_size=(128, 128)):
    try:
        image = Image.open(file_path)
        image = image.resize(target_size)
        image = np.array(image)
        image = image / 255.0  # Normalize to [0, 1]
        return image
    except Exception as e:
        print("Error loading image:", file_path, e)
        return None

def load_images(data, image_dir, target_size=(128, 128)):
    images = []
    labels = []
    bboxes = []
    for idx, row in data.iterrows():
        file_path = os.path.join(image_dir, row['filename'])
        image = load_and_preprocess_image(file_path, target_size)
        if image is not None:
            images.append(image)
            labels.append(row['class'])
            bboxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
    return np.array(images), np.array(labels), np.array(bboxes)

def build_and_train_model(train_images, train_labels, train_bboxes, test_images, test_labels, test_bboxes):
    input_shape = (128, 128, 3)
    inputs = Input(shape=input_shape)

    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Fully connected layers for classification
    class_output = Dense(3, activation='softmax', name='class_output')(x)

    # Fully connected layers for bounding box regression
    bbox_output = Dense(4, activation='sigmoid', name='bbox_output')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=[class_output, bbox_output])

    # Compile the model
    model.compile(optimizer='adam', 
                  loss={'class_output': 'sparse_categorical_crossentropy', 'bbox_output': 'mse'},
                  metrics={'class_output': 'accuracy'})

    # Display the model architecture
    model.summary()

    # Step 6: Train the Model
    try:
        history = model.fit(
            train_images, 
            {'class_output': train_labels, 'bbox_output': train_bboxes},
            validation_data=(test_images, {'class_output': test_labels, 'bbox_output': test_bboxes}),
            epochs=20,
            batch_size=32
        )
        print("Model training completed successfully.")
        return model
    except Exception as e:
        print("Error during model training:", e)
        return None

def save_model(model, file_path='monkey_cat_dog_detection_model.h5'):
    try:
        model.save(file_path)
        print("Model trained and saved successfully.")
    except Exception as e:
        print("Error saving the model:", e)

def run_pipeline(csv_path, image_dir):
    # Step 1: Load and Inspect the Dataset
    try:
        data = pd.read_csv(csv_path, encoding='utf-8')
        print("Dataset loaded successfully.")
    except UnicodeDecodeError:
        data = pd.read_csv(csv_path, encoding='ISO-8859-1')
        print("Dataset loaded with ISO-8859-1 encoding.")
    except Exception as e:
        print("Error loading dataset:", e)
        return

    # Step 2: Preprocess the Data
    try:
        # Encode class labels
        label_encoder = LabelEncoder()
        data['class'] = label_encoder.fit_transform(data['class'])

        # Normalize bounding box coordinates
        data['xmin'] = data['xmin'] / data['width']
        data['ymin'] = data['ymin'] / data['height']
        data['xmax'] = data['xmax'] / data['width']
        data['ymax'] = data['ymax'] / data['height']
        print("Data preprocessing completed successfully.")
    except Exception as e:
        print("Error during data preprocessing:", e)
        return

    # Step 3: Split the Data
    try:
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        print("Data split into training and testing sets successfully.")
    except Exception as e:
        print("Error during data splitting:", e)
        return

    try:
        # Load and preprocess training and testing images
        train_images, train_labels, train_bboxes = load_images(train_data, image_dir)
        test_images, test_labels, test_bboxes = load_images(test_data, image_dir)
        print("Images loaded and preprocessed successfully.")
    except Exception as e:
        print("Error during image loading:", e)
        return

    model = build_and_train_model(train_images, train_labels, train_bboxes, test_images, test_labels, test_bboxes)
    if model:
        save_model(model)
        # Now perform predictions and visualize results
        visualize_predictions(model, test_images, test_labels, test_bboxes, test_data, image_dir, label_encoder)

def visualize_predictions(model, test_images, test_labels, test_bboxes, test_data, image_dir, label_encoder):
    predictions = model.predict(test_images)
    class_predictions = np.argmax(predictions[0], axis=1)
    bbox_predictions = predictions[1]

    for i in range(len(test_images)):
        original_image_path = os.path.join(image_dir, test_data.iloc[i]['filename'])
        original_image = Image.open(original_image_path)
        draw = ImageDraw.Draw(original_image)
        
        # Original bounding box
        orig_bbox = [test_data.iloc[i]['xmin'] * test_data.iloc[i]['width'],
                     test_data.iloc[i]['ymin'] * test_data.iloc[i]['height'],
                     test_data.iloc[i]['xmax'] * test_data.iloc[i]['width'],
                     test_data.iloc[i]['ymax'] * test_data.iloc[i]['height']]
        draw.rectangle(orig_bbox, outline="green", width=2)
        
        # Predicted bounding box
        pred_bbox = [bbox_predictions[i][0] * test_data.iloc[i]['width'],
                     bbox_predictions[i][1] * test_data.iloc[i]['height'],
                     bbox_predictions[i][2] * test_data.iloc[i]['width'],
                     bbox_predictions[i][3] * test_data.iloc[i]['height']]
        draw.rectangle(pred_bbox, outline="red", width=2)
        
        # Predicted class label
        class_label = label_encoder.inverse_transform([class_predictions[i]])[0]
        draw.text((pred_bbox[0], pred_bbox[1]), class_label, fill="red")

        # Save or show the image with predictions
        original_image.show()
        original_image.save(f"output_{i}.png")
        
# GUI Part
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Monkey, Cat, Dog Detection")
        self.geometry("400x200")

        self.csv_path = tk.StringVar()
        self.image_dir = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text="Select CSV File:").pack(pady=10)
        tk.Entry(self, textvariable=self.csv_path, width=50).pack(pady=5)
        tk.Button(self, text="Browse", command=self.browse_csv).pack(pady=5)

        tk.Label(self, text="Select Image Directory:").pack(pady=10)
        tk.Entry(self, textvariable=self.image_dir, width=50).pack(pady=5)
        tk.Button(self, text="Browse", command=self.browse_images).pack(pady=5)

        tk.Button(self, text="Run Pipeline", command=self.run_pipeline).pack(pady=20)

    def browse_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.csv_path.set(file_path)

    def browse_images(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.image_dir.set(dir_path)

    def run_pipeline(self):
        csv_path = self.csv_path.get()
        image_dir = self.image_dir.get()

        if not csv_path or not image_dir:
            messagebox.showerror("Error", "Please select both CSV file and Image Directory.")
            return

        run_pipeline(csv_path, image_dir)
        messagebox.showinfo("Success", "Pipeline executed successfully.")

if __name__ == "__main__":
    app = Application()
    app.mainloop()
