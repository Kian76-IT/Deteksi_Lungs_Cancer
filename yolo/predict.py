from ultralytics import YOLO

model = YOLO("runs/detect/lung_cancer_detection2/weights/best.pt")

results = model("yolo/test1.jpg", save=True)