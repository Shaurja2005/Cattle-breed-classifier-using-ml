import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Apply the same transforms as training/validation
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Load test dataset
test_dir = "./test"  # <-- change this path if needed
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
num_classes = len(test_dataset.classes)
model = models.resnet18(pretrained=False)  
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("./cow_breed_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# Evaluate
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
# Optional: print class names
print("Classes:", test_dataset.classes)