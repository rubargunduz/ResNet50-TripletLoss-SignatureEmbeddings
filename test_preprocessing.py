import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from preprocessing import (
    binarize_otsu,
    crop_to_contour,
    pad_to_square,
    add_border_padding,
    resize_to_target,
    to_3_channels,
    preprocess_signature
)

# Suppress Tkinter root window
def select_file():
    Tk().withdraw()
    file_path = filedialog.askopenfilename(title="Select a signature image")
    return file_path

# Show images side-by-side
def plot_steps(titles, images):
    n = len(images)
    plt.figure(figsize=(4 * n, 4))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        if len(images[i].shape) == 2:  # grayscale
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    path = select_file()
    if not path:
        print("No file selected.")
        return

    original = cv2.imread(path)
    binary = binarize_otsu(original)
    cropped = crop_to_contour(binary)
    squared = pad_to_square(cropped, pad_value=0)
    padded = add_border_padding(squared, pad_value=0)
    resized = resize_to_target(padded)
    color = to_3_channels(resized)
    normalized = color.astype(np.float32) / 255.0

    pipeline = preprocess_signature(path, True)

    plot_steps(
        ["Original", "Binary (Otsu)", "Cropped", "Square", "Padded", "Resized & 3-channel", "Pipeline"],
        [original, binary, cropped, squared, padded, color, pipeline]
    )

if __name__ == "__main__":
    main()
