import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# ================= CONFIG =================
DATA_DIR = "dataset"
BATCH_SIZE = 8
EPOCHS = 20
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============== TRANSFORMS ===============
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomRotation(90),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

print("All images are resized to 224x224 during training via transforms.Resize")
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ============== DATASETS =================
train_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/val",
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/test",
    transform=val_transform
)

from torch.utils.data import WeightedRandomSampler
from collections import Counter

class_counts = Counter(train_dataset.targets)
total = sum(class_counts.values())

sample_weights = [total / class_counts[label] for label in train_dataset.targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Class mapping:", train_dataset.class_to_idx)

# ================ MODEL ==================
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

from collections import Counter

class_counts = Counter(train_dataset.targets)
total = sum(class_counts.values())

weights = [total / class_counts[i] for i in range(NUM_CLASSES)]
weights = torch.tensor(weights).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ============ VALIDATION FUNCTION ============
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# ============== TRAIN LOOP ===============
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    val_acc = evaluate(model, val_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"- Loss: {avg_loss:.4f} "
        f"- Val Acc: {val_acc*100:.2f}%"
    )

# ============== SAVE MODEL ===============
torch.save(model.state_dict(), "phase1_multiclass_model.pth")

print("Model saved as phase1_multiclass_model.pth")

# ============== TEST EVALUATION ==============
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_preds,
    target_names=test_dataset.classes
))
