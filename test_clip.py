from models.clip_encoder import get_embedding


image_path = "data/images/1.jpg"

emb = get_embedding(image_path)
print("Embedding shape:", emb.shape)
