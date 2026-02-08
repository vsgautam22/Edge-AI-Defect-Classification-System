import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter

# ================= CONFIG =================
DATA_DIR = "dataset"
MODEL_PATH = "phase1_multiclass_model.pth"
NUM_CLASSES = 10
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============== TRANSFORMS ===============
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ============== LOAD DATA ===============
test_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/test",
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("\n=== DATASET CHECK ===")
print("Classes:", test_dataset.classes)
print("Test samples:", len(test_dataset))
print("Class distribution:", Counter(test_dataset.targets))

# ============== LOAD MODEL ===============
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("\n=== MODEL CHECK ===")
print("Model loaded successfully")
print("Total parameters:",
      sum(p.numel() for p in model.parameters()))
print("Trainable parameters:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))

# ============== FORWARD PASS CHECK ===============
softmax = nn.Softmax(dim=1)
all_probs = []
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = softmax(outputs)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(torch.argmax(probs, dim=1).cpu().numpy())
        all_labels.append(labels.numpy())

all_probs = np.concatenate(all_probs)
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

print("\n=== OUTPUT SANITY CHECK ===")
print("Probabilities range:",
      all_probs.min(), "to", all_probs.max())
print("Mean confidence:",
      np.mean(np.max(all_probs, axis=1)))

# ============== CLASS COVERAGE CHECK ===============
predicted_classes = Counter(all_preds)
true_classes = Counter(all_labels)

print("\n=== CLASS COVERAGE ===")
print("True class counts:")
for i, cls in enumerate(test_dataset.classes):
    print(f"{cls:15s}: {true_classes.get(i, 0)}")

print("\nPredicted class counts:")
for i, cls in enumerate(test_dataset.classes):
    print(f"{cls:15s}: {predicted_classes.get(i, 0)}")

# ============== CONFIDENCE RISK CHECK ===============
high_confidence = np.sum(np.max(all_probs, axis=1) > 0.95)
low_confidence = np.sum(np.max(all_probs, axis=1) < 0.40)

print("\n=== CONFIDENCE RISK ANALYSIS ===")
print(f"High confidence predictions (>95%): {high_confidence}")
print(f"Low confidence predictions (<40%): {low_confidence}")

print("\n=== AUDIT COMPLETE ===")
