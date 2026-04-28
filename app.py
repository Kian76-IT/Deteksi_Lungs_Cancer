import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
import os
import io
import time

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Lung Cancer Detection", layout="centered")

st.title("🫁 Lung Cancer Detection AI")
st.write("Upload gambar rontgen paru-paru untuk mendeteksi kanker")

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

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
# LOAD MODELS (CACHE)
# =========================
@st.cache_resource
def load_models():
    cnn_model = CNNModel().to(device)
    cnn_model.load_state_dict(torch.load("cnn/cnn_model.pth", map_location=device))
    cnn_model.eval()

    yolo_model = YOLO("yolo/best.pt")

    return cnn_model, yolo_model

cnn_model, yolo_model = load_models()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Input", use_column_width=True)

    if st.button("🔍 Deteksi Sekarang"):

        # =========================
        # CNN
        # =========================
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = cnn_model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        class_names = ["cancer", "normal"]
        result = class_names[predicted.item()]

        st.subheader("🧠 Hasil CNN")
        st.write(f"Prediction: **{result}**")
        st.write(f"Confidence: **{confidence.item()*100:.2f}%**")

        # =========================
        # YOLO
        # =========================
        if result == "cancer":
            st.subheader("📍 Deteksi Lokasi (YOLO)")

            temp_path = "temp.jpg"
            image.save(temp_path)

            results = yolo_model.predict(
                source=temp_path,
                conf=0.25
            )

            result_img = results[0].plot()

            # tampilkan
            st.image(result_img, caption="Hasil YOLO", use_column_width=True)

            # =========================
            # DOWNLOAD BUTTON
            # =========================
            img_pil = Image.fromarray(result_img)

            buf = io.BytesIO()
            img_pil.save(buf, format="JPEG")
            byte_im = buf.getvalue()

            filename = f"hasil_{int(time.time())}.jpg"

            st.download_button(
                label="💾 Download Hasil",
                data=byte_im,
                file_name=filename,
                mime="image/jpeg"
            )

            os.remove(temp_path)

        else:
            st.success("Tidak ada indikasi kanker")