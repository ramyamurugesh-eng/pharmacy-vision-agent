import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os

DATA_DIR_RAW = "raw"
DATA_DIR_AUG = "augmented"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# combine raw + augmented into one dataset (ImageFolder needs one root with class subfolders,
# so we load both separately and concatenate)
raw_dataset = datasets.ImageFolder(DATA_DIR_RAW, transform=transform)
aug_dataset = datasets.ImageFolder(DATA_DIR_AUG, transform=transform)

assert raw_dataset.classes == aug_dataset.classes, "Class mismatch between raw and augmented folders!"
class_names = raw_dataset.classes
print(f"Classes ({len(class_names)}): {class_names}")

full_dataset = torch.utils.data.ConcatDataset([raw_dataset, aug_dataset])
print(f"Total images: {len(full_dataset)}")

# 80/20 train/val split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# --- Model setup: transfer learning with MobileNetV2 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# freeze the pretrained feature layers — we only train the final classifier layer
for param in model.features.parameters():
    param.requires_grad = False

# replace the final layer to output 53 classes instead of ImageNet's 1000
num_classes = len(class_names)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

# --- Training loop ---
def train_one_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
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
    EPOCHS = 5

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = validate()
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.2%} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.2%}")

    # save the trained model and class names together
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }, "medicine_classifier.pt")
    print("\nModel saved to medicine_classifier.pt")