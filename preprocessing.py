import cv2
import numpy as np

# Step 1: Binarize with Otsu's threshold
def binarize_otsu(image, sensitivity=32):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adjusted_thresh = max(otsu_thresh + sensitivity, 0)
    _, binary = cv2.threshold(gray, adjusted_thresh, 255, cv2.THRESH_BINARY_INV)
    return binary

# Step 2: Crop to bounding box of the signature
def crop_to_contour(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary_image  # fallback if no contour
    x, y, w, h = cv2.boundingRect(cv2.convexHull(np.vstack(contours)))
    cropped = binary_image[y:y+h, x:x+w]
    return cropped

# Step 3: Pad to square shape (white background)
def pad_to_square(img, pad_value=0):
    h, w = img.shape[:2]
    size = max(h, w)
    padded = np.full((size, size), pad_value, dtype=img.dtype)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img
    return padded

# Step 4: Add 25% padding around (white background)
def add_border_padding(img, pad_value=0):
    h, w = img.shape
    pad_h = int(h * 0.25)
    pad_w = int(w * 0.25)
    padded = cv2.copyMakeBorder(
        img, pad_h, pad_h, pad_w, pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value
    )
    return padded

# Step 5: Resize to 224x224
def resize_to_target(img, size=(224, 224)):
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return resized

# Step 6: Expand grayscale to 3 channels for ResNet
def to_3_channels(img):
    return cv2.merge([img, img, img])


# Optional: Connected components analysis
def clean_small_components(binary_img, threshold_ratio=0.25):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=4)
    h, w = binary_img.shape
    img_diag = np.sqrt(h**2 + w**2)
    cleaned = np.zeros_like(binary_img)

    for i in range(1, num_labels):  # skip background (label 0)
        x, y, bw, bh, area = stats[i]
        comp_diag = np.sqrt(bw**2 + bh**2)
        if comp_diag >= threshold_ratio * img_diag:
            cleaned[labels == i] = 255

    return cleaned

# 🔄 Full pipeline
def preprocess_signature(image_path, clean = False):
    original = cv2.imread(image_path)
    binary = binarize_otsu(original)

    if clean:
        cleaned = clean_small_components(binary)
    else: 
        cleaned = binary

    cropped = crop_to_contour(cleaned)
    squared = pad_to_square(cropped, pad_value=0)  # black background in binary image
    extended = add_border_padding(squared, pad_value=0)
    resized = resize_to_target(extended)
    color = to_3_channels(resized)
    normalized = color.astype(np.float32) / 255.0

    return normalized
