import os
import numpy as np
from PIL import Image
from models.clip_encoder import get_embedding

IMAGE_DIR = "data/images"

embeddings = []
image_paths = []

for img_name in os.listdir(IMAGE_DIR):
    path = os.path.join(IMAGE_DIR, img_name)

    try:
        # Ensure image is valid
        Image.open(path).convert("RGB")

        emb = get_embedding(path)
        embeddings.append(emb[0])
        image_paths.append(path)

        print(f"Processed: {img_name}")

    except Exception as e:
        print(f"Skipping {img_name}: {e}")

embeddings = np.array(embeddings)

print("\nSUMMARY")
print("Total valid images:", len(image_paths))
print("Embeddings shape:", embeddings.shape)

if len(embeddings) > 0:
    np.save("data/embeddings.npy", embeddings)
    np.save("data/image_paths.npy", image_paths)
