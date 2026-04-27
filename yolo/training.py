from ultralytics import YOLO
import torch

def main():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    
    # LOAD MODEL
    model = YOLO("yolov8n.pt")  

    
    # TRAINING
    results = model.train(
        data="yolo/dataset\data.yaml",  
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,  
        name="lung_cancer_detection"
    )

   
    # EVALUASI (VALIDATION)
    metrics = model.val()

    print("\n=== HASIL YOLO ===")
    print(f"mAP50      : {metrics.box.map50:.4f}")
    print(f"mAP50-95   : {metrics.box.map:.4f}")
    print(f"Precision  : {metrics.box.mp:.4f}")
    print(f"Recall     : {metrics.box.mr:.4f}")

  
    # SAVE PATH MODEL
    print("\nModel tersimpan di:")
    print("runs/detect/lung_cancer_detection/weights/best.pt")

if __name__ == "__main__":
    main()