import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
import os

# =========================
# SET DEVICE (CPU / GPU)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# CNN MODEL
# =========================
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# =========================
# LOAD CNN
# =========================
cnn_model = CNNModel().to(device)
cnn_model.load_state_dict(torch.load("cnn/cnn_model.pth", map_location=device))
cnn_model.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# LOAD YOLO
# =========================
yolo_model = YOLO("yolo/best.pt")

# =========================
# PREDICT FUNCTION
# =========================
def predict(image_path):
    if not os.path.exists(image_path):
        print("Image tidak ditemukan!")
        return

    filename = os.path.basename(image_path).split(".")[0]

    # ABSOLUTE PATH (BIAR TIDAK MASUK RUNS)
    output_dir = os.path.abspath("hasil")

    print("Output folder:", output_dir)

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # =========================
    # CNN
    # =========================
    with torch.no_grad():
        outputs = cnn_model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    class_names = ["cancer", "normal"]
    result = class_names[predicted.item()]

    print("\n=== HASIL CNN ===")
    print(f"Prediction : {result}")
    print(f"Confidence : {confidence.item()*100:.2f}%")

    # =========================
    # YOLO (JIKA CANCER)
    # =========================
    if result == "cancer":
        print("\nMenjalankan YOLO detection...")

        yolo_model.predict(
            source=image_path,
            device=0 if torch.cuda.is_available() else "cpu",
            save=True,
            conf=0.25,
            project=output_dir,   # 🔥 FIX (keluar dari runs)
            name=filename,       # 🔥 folder dinamis
            exist_ok=True
        )

        print(f"Hasil disimpan di: {output_dir}/{filename}/")
    else:
        print("Tidak ada indikasi cancer → YOLO tidak dijalankan")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    image_path = "test2.jpg"  # 🔥 ganti sesuai file kamu
    predict(image_path)