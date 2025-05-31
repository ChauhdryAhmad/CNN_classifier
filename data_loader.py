import os
import numpy as np
from PIL import Image
from neural_net.utils import normalize_images, one_hot_encode, train_test_split


class CardDataLoader:
    def __init__(self, config):
        self.config = config
        self.class_map = {"king": 0, "queen": 1, "jack": 2, "ace": 3, "other": 4}
        self.reverse_class_map = {v: k for k, v in self.class_map.items()}

    def load_data(self, data_dir):
        """Load and preprocess image data from directory structure"""
        images = []
        labels = []

        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            # Use the folder name directly as the class label
            if class_name.lower() not in self.class_map:
                continue  # Skip unknown class folders

            label = self.class_map[class_name.lower()]

            # Load images
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_dir, img_file)
                    try:
                        img = Image.open(img_path).convert("L")  # Grayscale
                        img = img.resize(
                            (self.config["img_width"], self.config["img_height"])
                        )
                        images.append(np.array(img))
                        labels.append(label)
                    except Exception as e:
                        print(f"Warning: Could not process image {img_path}: {e}")

        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        # Add channel dimension (for grayscale)
        images = np.expand_dims(images, axis=-1)

        # Normalize and one-hot encode
        images = normalize_images(images)
        labels = one_hot_encode(labels, num_classes=5)

        return train_test_split(
            images,
            labels,
            test_size=self.config["test_size"],
            shuffle=self.config["shuffle"],
        )
