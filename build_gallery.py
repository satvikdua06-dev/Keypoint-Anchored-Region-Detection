import os
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms

# Use ResNet18 as a feature extractor — lightweight, fast
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Remove the classification head — we want the 512-dim feature vector
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

REFERENCE_DIR = "reference_images"

# Store each embedding with its label
gallery_embeddings = []  # list of 512-dim vectors
gallery_labels = []      # list of strings like "glasses", "no_glasses"

for class_name in os.listdir(REFERENCE_DIR):
    class_dir = os.path.join(REFERENCE_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    count = 0
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
            tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                emb = feature_extractor(tensor).squeeze().numpy()
                emb = emb / np.linalg.norm(emb)  # L2 normalize
            gallery_embeddings.append(emb)
            gallery_labels.append(class_name)
            count += 1
            print(f"  [OK] {img_path}")
        except Exception as e:
            print(f"  [FAIL] {img_path}: {e}")

    print(f"[STORED] {class_name}: {count} images stored")

# Save as a single dict
gallery = {
    "embeddings": np.array(gallery_embeddings),  # shape: (N, 512)
    "labels": gallery_labels,                     # length: N
}
np.save("ppe_gallery.npy", gallery)
print(f"\n[DONE] Gallery saved! {len(gallery_labels)} total embeddings across {len(set(gallery_labels))} classes")