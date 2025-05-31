import os
import json
import numpy as np
from PIL import Image
from neural_net.network import Sequential
from neural_net.layers import Conv2D, MaxPool2D, Flatten, Dense
from neural_net.activations import ReLU, Softmax
from neural_net.utils import normalize_images


class CardPredictor:
    def __init__(self, config_path="config.json"):
        with open(config_path) as f:
            self.config = json.load(f)

        self.model = self.build_model()
        self.model.load_weights(self.config["weights_path"])

        self.class_names = ["king", "queen", "jack", "ace", "other"]

    def build_model(self):
        """Reconstruct the model architecture"""
        return Sequential(
            [
                Conv2D(
                    num_filters=self.config["conv1_filters"],
                    kernel_size=self.config["conv1_kernel"],
                    stride=self.config["conv1_stride"],
                    padding=self.config["conv1_padding"],
                ),
                ReLU(),
                MaxPool2D(
                    pool_size=self.config["pool1_size"],
                    stride=self.config["pool1_stride"],
                ),
                Conv2D(
                    num_filters=self.config["conv2_filters"],
                    kernel_size=self.config["conv2_kernel"],
                    stride=self.config["conv2_stride"],
                    padding=self.config["conv2_padding"],
                ),
                ReLU(),
                MaxPool2D(
                    pool_size=self.config["pool2_size"],
                    stride=self.config["pool2_stride"],
                ),
                Flatten(),
                Dense(self.config["dense_units"]),
                ReLU(),
                Dense(5),
                Softmax(),
            ]
        )

    def preprocess_image(self, img_path):
        """Preprocess a single image for prediction"""
        img = Image.open(img_path).convert("L")
        img = img.resize((self.config["img_width"], self.config["img_height"]))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=(0, -1))
        return normalize_images(img_array)

    def predict(self, img_path):
        """Make prediction on a single image"""
        processed_img = self.preprocess_image(img_path)
        probs = self.model.predict(processed_img)[0]
        class_idx = np.argmax(probs)
        return {
            "image": os.path.basename(img_path),
            "class": self.class_names[class_idx],
            "confidence": float(probs[class_idx]),
            "probabilities": {
                name: float(prob) for name, prob in zip(self.class_names, probs)
            },
        }

    def predict_folder(self, folder_path):
        """Run prediction on all images in the folder"""
        image_files = [
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            print("No image files found in the folder.")
            return

        for img_file in sorted(image_files):
            full_path = os.path.join(folder_path, img_file)
            try:
                prediction = self.predict(full_path)
                print(f"\n🖼️ Image: {prediction['image']}")
                print(f"Class: {prediction['class']}")
                print(f"Confidence: {prediction['confidence']:.2%}")
                print("Probabilities:")
                for cls, prob in prediction["probabilities"].items():
                    print(f"  {cls}: {prob:.2%}")
            except Exception as e:
                print(f"Failed to process {img_file}: {e}")


if __name__ == "__main__":
    folder_path = input("Enter folder path: ").strip()
    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
    else:
        predictor = CardPredictor()
        predictor.predict_folder(folder_path)
