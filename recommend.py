import numpy as np
from models.clip_encoder import get_embedding

embeddings = np.load("data/embeddings.npy")
image_paths = np.load("data/image_paths.npy", allow_pickle=True)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recommend(image_path, top_k=5):
    query_emb = get_embedding(image_path)[0]

    scores = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_emb, emb)
        scores.append((score, image_paths[i]))

    scores.sort(reverse=True)
    return scores[:top_k]
