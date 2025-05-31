import json
import time
import numpy as np
from neural_net.network import Sequential
from neural_net.layers import Conv2D, MaxPool2D, Flatten, Dense
from neural_net.activations import ReLU, Softmax
from neural_net.losses import CrossEntropyLoss
from neural_net.optimizers import Adam
from neural_net.utils import accuracy
from neural_net.gpu_util import GPU_AVAILABLE
from data_loader import CardDataLoader


def build_model(config):
    """Build the CNN model from configuration"""
    return Sequential(
        [
            Conv2D(
                num_filters=config["conv1_filters"],
                kernel_size=config["conv1_kernel"],
                stride=config["conv1_stride"],
                padding=config["conv1_padding"],
            ),
            ReLU(),
            MaxPool2D(pool_size=config["pool1_size"], stride=config["pool1_stride"]),
            Conv2D(
                num_filters=config["conv2_filters"],
                kernel_size=config["conv2_kernel"],
                stride=config["conv2_stride"],
                padding=config["conv2_padding"],
            ),
            ReLU(),
            MaxPool2D(pool_size=config["pool2_size"], stride=config["pool2_stride"]),
            Flatten(),
            Dense(config["dense_units"]),
            ReLU(),
            Dense(5),  # 5 output classes
            Softmax(),
        ]
    )


def train():
    # Load configuration
    with open("config.json") as f:
        config = json.load(f)

    # Initialize data loader and load data
    loader = CardDataLoader(config)
    X_train, X_val, y_train, y_val = loader.load_data(config["train_data_dir"])

    # Build and compile model
    model = build_model(config)
    if GPU_AVAILABLE:
        model.to_gpu()
        print("Using GPU acceleration")
    else:
        print("Using CPU")
    model.compile(loss=CrossEntropyLoss(), optimizer=Adam(lr=config["learning_rate"]))

    # Test parameter shapes
    # Initialize parameters with dummy forward pass
    dummy_input = np.random.randn(2, config["img_height"], config["img_width"], 1)
    _ = model.forward(dummy_input)

    # Verify parameter shapes
    print("Network parameter shapes:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, "params"):
            print(f"Layer {i} ({layer.__class__.__name__}):")
            for param, value in layer.params.items():
                print(f"  {param}: {value.shape}")

    # Test with different batch sizes
    # for batch_size in [1, 2, 4, 8, 16]:
    #     print(f"\nTesting with batch_size = {batch_size}")
    #     test_input = np.random.randn(batch_size, 64, 64, 1)
    #     test_output = np.random.randn(batch_size, 5)
    #
    #     model = build_model(config)
    #     model.compile(loss=CrossEntropyLoss(), optimizer=Adam(lr=0.001))
    #
    #     # Initialize parameters
    #     _ = model.forward(test_input)
    #
    #     # Test training step
    #     loss = model.train_step(test_input, test_output)
    #     print(f"Training step completed successfully! Loss: {loss:.4f}")

    # Training loop
    best_val_acc = 0
    for epoch in range(config["epochs"]):
        epoch_loss = 0
        epoch_acc = 0

        batch = 0
        # Mini-batch training
        for i in range(0, len(X_train), config["batch_size"]):
            X_batch = X_train[i : i + config["batch_size"]]
            y_batch = y_train[i : i + config["batch_size"]]

            print(f"Batch {batch}")
            batch += 1

            loss = model.train_step(X_batch, y_batch)
            preds = model.predict(X_batch)
            acc = accuracy(preds, y_batch)

            epoch_loss += loss * len(X_batch)
            epoch_acc += acc * len(X_batch)

        # Validation
        val_preds = model.predict(X_val)
        val_acc = accuracy(val_preds, y_val)

        # Print epoch statistics
        epoch_loss /= len(X_train)
        epoch_acc /= len(X_train)
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(config["weights_path"])
            print("Saved new best weights!")

        print("---------------------")


if __name__ == "__main__":
    # Test convolution dimensions
    test_input = np.random.randn(2, 64, 64, 1)
    conv = Conv2D(16, 3, padding=1)
    test_output = conv.forward(test_input)
    print("Conv2D test:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")

    # Test backward pass
    dout = np.random.randn(*test_output.shape)
    dx = conv.backward(dout)
    print(f"Gradient shape: {dx.shape}")
    print("All shapes match!")

    train()
