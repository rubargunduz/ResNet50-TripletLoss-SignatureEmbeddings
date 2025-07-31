import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Concatenate

from preprocessing import preprocess_signature


# ------------------- Config ------------------- #
INPUT_SHAPE = (224, 224, 3)
EMBEDDING_DIM = 128
MARGIN = 5
BATCH_SIZE = 32
EPOCHS = 10
STEPS_PER_EPOCH = 100

TRIPLET_PATH = "triplets.txt"
VAL_TRIPLET_PATH = "val_triplets.txt"
VALIDATION_STEPS = 20

MODEL_SAVE_PATH = "signature_embedding_model.h5"


# ----------------- Embedding Model ----------------- #
def create_embedding_model(input_shape=INPUT_SHAPE, embedding_dim=EMBEDDING_DIM):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base.output)
    output = Dense(embedding_dim, activation='linear')(x)
    return Model(inputs=base.input, outputs=output)


# ----------------- Triplet Model ----------------- #
def create_triplet_model(embedding_model):
    anchor = Input(shape=INPUT_SHAPE)
    positive = Input(shape=INPUT_SHAPE)
    negative = Input(shape=INPUT_SHAPE)

    emb_anchor = embedding_model(anchor)
    emb_positive = embedding_model(positive)
    emb_negative = embedding_model(negative)

    merged = Concatenate(axis=1)([emb_anchor, emb_positive, emb_negative])
    return Model(inputs=[anchor, positive, negative], outputs=merged)


# ----------------- Triplet Loss ----------------- #
def triplet_loss(margin=MARGIN):
    def loss(y_true, y_pred):
        # y_pred shape: (batch_size, embedding_dim * 3)
        embedding_dim = EMBEDDING_DIM
        anchor = y_pred[:, 0:embedding_dim]
        positive = y_pred[:, embedding_dim:2*embedding_dim]
        negative = y_pred[:, 2*embedding_dim:3*embedding_dim]
        pos_dist = K.sum(K.square(anchor - positive), axis=-1)
        neg_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.mean(K.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss


# ----------------- Load Triplets ----------------- #
def load_triplets(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    triplets = [tuple(line.strip().split(",")) for line in lines]
    return triplets


# ----------------- Data Augmentation ----------------- #
augmentor = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    fill_mode='nearest'
)

def load_image(path, augment=False):
    img = preprocess_signature(path)  # Preprocessing pipeline
    if augment:
        img = augmentor.random_transform(img)
    return img



# ----------------- Triplet Generator ----------------- #
def generate_triplet_batch(triplets, batch_size):
    while True:
        anchor_imgs, positive_imgs, negative_imgs = [], [], []
        batch = random.sample(triplets, batch_size)
        for a, p, n in batch:
            anchor_imgs.append(load_image(a, augment=True))
            positive_imgs.append(load_image(p, augment=True))
            negative_imgs.append(load_image(n, augment=False))
        # Concatenate along axis 0 for batch, axis 1 for embedding
        anchors = np.array(anchor_imgs)
        positives = np.array(positive_imgs)
        negatives = np.array(negative_imgs)
        # Keras expects inputs as a tuple and dummy labels
        yield (anchors, positives, negatives), np.zeros((batch_size,))

def get_triplet_dataset(triplets, batch_size):
    output_signature = (
        (
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
    return tf.data.Dataset.from_generator(
        lambda: generate_triplet_batch(triplets, batch_size),
        output_signature=output_signature
    )


from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def visualize_validation(model, val_triplets, n=5):
    samples = random.sample(val_triplets, n)
    for a, p, n in samples:
        ea = model.predict(np.expand_dims(load_image(a), axis=0))[0]
        ep = model.predict(np.expand_dims(load_image(p), axis=0))[0]
        en = model.predict(np.expand_dims(load_image(n), axis=0))[0]

        sim_pos = cosine_similarity([ea], [ep])[0][0]
        sim_neg = cosine_similarity([ea], [en])[0][0]

        print(f"A-P sim: {sim_pos:.3f}, A-N sim: {sim_neg:.3f}")





# ----------------- Main Training ----------------- #
def main():
    print("Loading triplets...")
    triplets = load_triplets(TRIPLET_PATH)
    print(f"Loaded {len(triplets)} triplets.")

    val_triplets = load_triplets(VAL_TRIPLET_PATH)
    print(f"Loaded {len(val_triplets)} triplets.")

    print("Creating model...")
    embedding_model = create_embedding_model()
    triplet_model = create_triplet_model(embedding_model)

    print("Compiling model...")
    triplet_model.compile(loss=triplet_loss(), optimizer='adam')

    print("Starting training...")


    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('best_signature_model.h5', monitor='val_loss', save_best_only=True)
    ]

    # In main(), replace model.fit call:
    train_dataset = get_triplet_dataset(triplets, BATCH_SIZE)
    val_dataset = get_triplet_dataset(val_triplets, BATCH_SIZE)

    triplet_model.fit(
        train_dataset,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_dataset,
        validation_steps=VALIDATION_STEPS,
        epochs=EPOCHS,
        callbacks=callbacks
)


    print("Saving model...")
    embedding_model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    print("Validation samples:")
    visualize_validation(embedding_model, val_triplets)




if __name__ == "__main__":
    main()
