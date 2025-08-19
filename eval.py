import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from datasets.plant_dataset import PlantDataset
from models.mobilevit import MobileViTClassifier
from torchvision import transforms
from config import *

def evaluate(model_path, data_dir="data/PlantVillage", split="test"):
    # Load model
    model = MobileViTClassifier(image_size=(IMAGE_SIZE,IMAGE_SIZE), num_classes=len(CLASS_MAP))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    
    # Data loader
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    dataset = PlantDataset(data_dir, split, CLASS_MAP, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu()
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    print(classification_report(y_true, y_pred, target_names=list(CLASS_MAP.keys())))

if __name__ == "__main__":
    import sys
    model_file = sys.argv[1] if len(sys.argv)>1 else "outputs/checkpoints/mobilevit_final.pth"
    evaluate(model_file)