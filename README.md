# Signature Similarity on Embeddings from ResNet Trained with Triplet Loss

This project implements offline signature similarity calculation with cosine similarity using a ResNet-based embedding model trained with triplet loss.

The goal here can be:
- Signature similarity calculation
- Signature verification system
- Signature identification 

## Model

- **Embedding Model**: ResNet50
- **Triplet Loss**: Aiming anchor-positive distance to be smaller than the anchor-negative distance by a margin

- Cosine similarity is used to compare embeddings.
<img width="400" height="400" alt="Figure_1" src="https://github.com/user-attachments/assets/791fdcae-c397-4c92-8b6d-611db093b0f8" />
<img width="400" height="400" alt="Figure_2" src="https://github.com/user-attachments/assets/94edbe9e-599c-43e1-81c2-a2a423d66e64" />
<img width="800" height="432" alt="threshold-092-margin5" src="https://github.com/user-attachments/assets/2be2a6e7-ee79-4427-8ce7-63b69dd0aafc" />

## Dataset

Dataset used: https://www.kaggle.com/datasets/robinreni/signature-verification-dataset from Kaggle
The dataset has been reorganized for training,

Example structure:
```
dataset/
  train/
    person1/
      sig1.png
      sig2.png
      ...
    person2/
      ...
  test/
    personX/
      ...
```

Usage
1. Place your reorganized dataset in the expected folder structure.
2. Run `train.ipynb` to preprocess images, generate triplets and train the model.
3. Best performing model achieved by trainig all layers of resnet with high triplet loss margin values (eg. 5 or 10). 
