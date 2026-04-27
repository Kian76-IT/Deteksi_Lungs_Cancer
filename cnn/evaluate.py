import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# ================================
# CONFIG
# ================================
DATA_PATH = "processed_dataset"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# TRANSFORM (NO AUGMENTATION)
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================================
# LOAD DATA
# ================================
test_data = datasets.ImageFolder(f"{DATA_PATH}/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ================================
# MODEL (HARUS SAMA)
# ================================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ================================
# LOAD MODEL
# ================================
model = CNN().to(DEVICE)
model.load_state_dict(torch.load("cnn_model.pth", map_location=DEVICE))
model.eval()

# ================================
# EVALUATION
# ================================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ================================
# METRICS
# ================================
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=test_data.classes)

accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

print("\n=== TEST RESULT ===")
print("Accuracy:", accuracy * 100, "%")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)