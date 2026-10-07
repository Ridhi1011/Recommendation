import os
import shutil
from recommend import recommend

query_image = "data/images/144.jpg"
output_dir = "results"

os.makedirs(output_dir, exist_ok=True)

results = recommend(query_image)

for i, (score, path) in enumerate(results):
    dst = os.path.join(output_dir, f"rank_{i+1}.jpg")
    shutil.copy(path, dst)
    print(score, dst)
