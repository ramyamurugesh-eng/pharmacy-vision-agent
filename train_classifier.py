import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
import os

DATA_DIR_RAW = "raw"
DATA_DIR_AUG = "augmented"

# stronger augmentation to help bridge the real-world domain gap
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class TransformSubset(Dataset):
    """Wraps a Subset so we can apply a different transform than the base dataset used."""
    def __init__(self, base_dataset_no_transform, indices, transform):
        self.base = base_dataset_no_transform
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, label = self.base[self.indices[i]]
        return self.transform(image), label


# load WITHOUT any transform first, so we can split cleanly, then attach transforms per split
raw_dataset_raw = datasets.ImageFolder(DATA_DIR_RAW, transform=None)
aug_dataset_raw = datasets.ImageFolder(DATA_DIR_AUG, transform=None)

assert raw_dataset_raw.classes == aug_dataset_raw.classes, "Class mismatch between raw and augmented folders!"
class_names = raw_dataset_raw.classes
print(f"Classes ({len(class_names)}): {class_names}")

full_dataset_raw = torch.utils.data.ConcatDataset([raw_dataset_raw, aug_dataset_raw])
print(f"Total images: {len(full_dataset_raw)}")

train_size = int(0.8 * len(full_dataset_raw))
val_size = len(full_dataset_raw) - train_size
train_indices, val_indices = random_split(range(len(full_dataset_raw)), [train_size, val_size])

train_dataset = TransformSubset(full_dataset_raw, list(train_indices), train_transform)
val_dataset = TransformSubset(full_dataset_raw, list(val_indices), val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# --- Model setup: transfer learning with MobileNetV2, partially unfrozen ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

for param in model.features.parameters():
    param.requires_grad = False
for param in model.features[-4:].parameters():  # unfreeze last 4 feature blocks
    param.requires_grad = True

num_classes = len(class_names)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005
)


def train_one_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)} - running loss: {running_loss/total:.4f}")

    return running_loss / total, correct / total


def validate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


if __name__ == "__main__":
    EPOCHS = 8  # slightly more epochs since we're now training more parameters

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = validate()
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.2%} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.2%}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }, "medicine_classifier.pt")
    print("\nModel saved to medicine_classifier.pt")