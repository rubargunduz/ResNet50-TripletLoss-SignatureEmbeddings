import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess_signature

# Load the trained embedding model
embedding_model = load_model("margin5_fully.h5")

def select_image(title="Select an image"):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    root.destroy()
    return file_path

def get_embedding(image_path):
    img = preprocess_signature(image_path)
    img = np.expand_dims(img, axis=0)
    emb = embedding_model.predict(img)[0]
    return emb, img[0]

def main():
    print("Choose first signature image:")
    img1_path = select_image("Select first signature image")
    print("Choose second signature image:")
    img2_path = select_image("Select second signature image")

    if not img1_path or not img2_path:
        print("Image selection cancelled.")
        return

    emb1, proc1 = get_embedding(img1_path)
    emb2, proc2 = get_embedding(img2_path)
    similarity = cosine_similarity([emb1], [emb2])[0][0]

    # Visualization
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 0].imshow(plt.imread(img1_path), cmap='gray')
    axs[0, 0].set_title("Original Signature 1")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(plt.imread(img2_path), cmap='gray')
    axs[0, 1].set_title("Original Signature 2")
    axs[0, 1].axis('off')

    axs[1, 0].imshow((proc1 * 255).astype(np.uint8), cmap='gray')
    axs[1, 0].set_title("Preprocessed Signature 1")
    axs[1, 0].axis('off')

    axs[1, 1].imshow((proc2 * 255).astype(np.uint8), cmap='gray')
    axs[1, 1].set_title("Preprocessed Signature 2")
    axs[1, 1].axis('off')

    plt.suptitle(f"Cosine Similarity: {similarity:.3f}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



if __name__ == "__main__":
    main()