import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_data = datasets.ImageFolder("train", transform=transform)
val_data = datasets.ImageFolder("valid", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

num_classes = len(train_data.classes)

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # freeze feature extractor

model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    
    acc = 100 * correct / len(train_data)
    print(f"Epoch {epoch+1}, Loss: {running_loss:.3f}, Acc: {acc:.2f}%")

model.eval()
correct = 0
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        correct += (outputs.argmax(1) == labels).sum().item()

val_acc = 100 * correct / len(val_data)
print("Validation Accuracy:", val_acc)

torch.save(model.state_dict(), "cow_breed_classifier.pth")