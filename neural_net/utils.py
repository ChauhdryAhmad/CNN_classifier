import numpy as np


def one_hot_encode(y, num_classes):
    """Convert labels to one-hot encoding"""
    return np.eye(num_classes)[y]


def accuracy(y_pred, y_true):
    """Calculate accuracy"""
    if y_true.ndim == 2:  # If one-hot encoded
        y_true = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)
    return np.mean(pred_labels == y_true)


def train_test_split(X, y, test_size=0.2, shuffle=True):
    """Split data into train/test sets"""
    if shuffle:
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]

    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def normalize_images(images):
    """Normalize image data to [0, 1] range"""
    return images.astype("float32") / 255.0


def im2col(images, kernel_size, stride=1, padding=0):
    """Convert image batch to column matrix for convolution"""
    batch_size, h, w, c = images.shape
    k_h, k_w = kernel_size

    # Calculate output dimensions
    out_h = (h + 2 * padding - k_h) // stride + 1
    out_w = (w + 2 * padding - k_w) // stride + 1

    # Create column matrix
    cols = np.zeros((batch_size * out_h * out_w, k_h * k_w * c))

    for b in range(batch_size):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride - padding
                h_end = h_start + k_h
                w_start = j * stride - padding
                w_end = w_start + k_w

                # Extract patch with boundary checks
                patch = np.zeros((k_h, k_w, c))
                ph_start = max(0, -h_start)
                pw_start = max(0, -w_start)
                h_start = max(0, h_start)
                w_start = max(0, w_start)
                h_end = min(h, h_end)
                w_end = min(w, w_end)

                actual_patch = images[b, h_start:h_end, w_start:w_end, :]
                patch[
                    ph_start : ph_start + h_end - h_start,
                    pw_start : pw_start + w_end - w_start,
                    :,
                ] = actual_patch

                cols[b * out_h * out_w + i * out_w + j, :] = patch.flatten()

    return cols


def col2im(col, input_shape, kernel_size, stride=1, padding=0):
    """
    Transform column matrix back to image batch
    Args:
        col: (batch_size*out_h*out_w, k_height*k_width*channels)
        input_shape: (batch_size, height, width, channels)
        kernel_size: (k_height, k_width)
        stride: int
        padding: int
    Returns:
        images: (batch_size, height, width, channels)
    """
    batch_size, h, w, c = input_shape
    k_h, k_w = kernel_size
    out_h = (h + 2 * padding - k_h) // stride + 1
    out_w = (w + 2 * padding - k_w) // stride + 1

    images = np.zeros((batch_size, h + 2 * padding, w + 2 * padding, c))

    for b in range(batch_size):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                h_end = h_start + k_h
                w_start = j * stride
                w_end = w_start + k_w

                idx = b * out_h * out_w + i * out_w + j
                patch = col[idx, :].reshape(k_h, k_w, c)
                images[b, h_start:h_end, w_start:w_end, :] += patch

    # Remove padding if needed
    if padding > 0:
        images = images[:, padding:-padding, padding:-padding, :]

    return images
