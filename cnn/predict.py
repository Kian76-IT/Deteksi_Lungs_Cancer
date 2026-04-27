import torch
import torch.nn as nn
import cv2
import numpy as np

# ================================
# MODEL
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
model = CNN()
model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
model.eval()

# ================================
# PREDICT FUNCTION
# ================================
def predict(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        return "Image not found!"

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)

    return "Cancer" if pred.item() == 0 else "Normal"

# ================================
# TEST
# ================================
if __name__ == "__main__":
    result = predict("cnn/images/test3.jpg")  # ganti dengan gambar kamu
    print("Prediction:", result)