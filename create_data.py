import os
import random
from glob import glob

def make_triplets(data_dir, num_triplets=10000, seed=42):
    random.seed(seed)
    people = os.listdir(data_dir)
    people = [p for p in people if os.path.isdir(os.path.join(data_dir, p))]

    print("Number of people: ", len(people))

    triplets = []
    for _ in range(num_triplets):
        
        # Pick anchor/positive person
        person = random.choice(people)
        person_path = os.path.join(data_dir, person)

        images = glob(os.path.join(person_path, "*.png"))
        if len(images) < 2:
            continue  # skip if not enough images

        anchor, positive = random.sample(images, 2)

        # Pick negative person
        neg_person = random.choice([p for p in people if p != person])
        neg_path = os.path.join(data_dir, neg_person)
        
        neg_images = glob(os.path.join(neg_path, "*.png"))
        if not neg_images:
            continue

        negative = random.choice(neg_images)
        triplets.append((anchor, positive, negative))


    return triplets


triplets = make_triplets("dataset/train", num_triplets=10000, seed=42)
with open("triplets.txt", "w") as f:
    for a, p, n in triplets:
        f.write(f"{a},{p},{n}\n")
