import os
import numpy as np
import faiss
from models.clip_encoder import get_embedding

IMAGE_DIR = "data/images"
embeddings = []
image_paths = []

for img_name in os.listdir(IMAGE_DIR):
    path = os.path.join(IMAGE_DIR, img_name)
    try:
        emb = get_embedding(path)
        embeddings.append(emb[0])
        image_paths.append(path)
    except Exception as e:
        print(f"Skipping {img_name} due to error: {e}")


embeddings = np.array(embeddings).astype("float32")

os.makedirs("index", exist_ok=True)
np.save("data/embeddings.npy", embeddings)
np.save("data/image_paths.npy", image_paths)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "index/faiss_index.bin")
