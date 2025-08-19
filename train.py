import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from models.mobilevit import MobileViTClassifier
from datasets.plant_dataset import PlantDataset
from utils.trainer import trainer



# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileViT for Plant Disease")
    parser.add_argument('--data-dir', type=str, default='data/PlantVillage')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_map = {'Potato___healthy': 0,
                 'Potato___Early_blight': 1,
                 'Pepper,_bell___Bacterial_spot': 2,
                 'Tomato___Tomato_mosaic_virus': 3,
                 'Tomato___Spider_mites Two-spotted_spider_mite': 4,
                 'Tomato___Late_blight': 5,
                 'Strawberry___healthy': 6,
                 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 7,
                 'Apple___Cedar_apple_rust': 8,
                 'Orange___Haunglongbing_(Citrus_greening)': 9,
                 'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 10,
                 'Tomato___Leaf_Mold': 11,
                 'Squash___Powdery_mildew': 12,
                 'Apple___Apple_scab': 13,
                 'Peach___Bacterial_spot': 14,
                 'Soybean___healthy': 15,
                 'Cherry_(including_sour)___Powdery_mildew': 16,
                 'Peach___healthy': 17,
                 'Raspberry___healthy': 18,
                 'Corn_(maize)___Common_rust_': 19,
                 'Tomato___Early_blight': 20,
                 'Grape___healthy': 21,
                 'Cherry_(including_sour)___healthy': 22,
                 'Tomato___Target_Spot': 23,
                 'Pepper,_bell___healthy': 24,
                 'Tomato___healthy': 25,
                 'Strawberry___Leaf_scorch': 26,
                 'Corn_(maize)___healthy': 27,
                 'Apple___Black_rot': 28,
                 'Blueberry___healthy': 29,
                 'Apple___healthy': 30,
                 'Potato___Late_blight': 31,
                 'Grape___Black_rot': 32,
                 'Tomato___Septoria_leaf_spot': 33,
                 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 34,
                 'Tomato___Bacterial_spot': 35,
                 'Grape___Esca_(Black_Measles)': 36,
                 'Corn_(maize)___Northern_Leaf_Blight': 37}

    # Transforms
    prob = 0.5
    image_size = 224  # define your size

    train_tf = transforms.Compose([transforms.Resize((image_size, image_size)),
                                   transforms.RandomApply(torch.nn.ModuleList([transforms.RandomHorizontalFlip(p=1.0),
                                                                               transforms.RandomPerspective(distortion_scale=0.1, p=1.0),
                                                                               transforms.RandomRotation(degrees=10),
                                                                               transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
                                                                              ]), p=prob
                                                         ),
                                    transforms.ColorJitter(brightness=0.3, hue=0.1),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225]
                                                        )
                                  ])

    test_tf = transforms.Compose([transforms.Resize((image_size, image_size)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                            ])

    # DataLoaders
    train_set = PlantDataset(args.data_dir, "train", class_map, train_tf)
    val_set = PlantDataset(args.data_dir, "test", class_map, test_tf)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # Model, optimizer, criterion, scheduler
    model = MobileViTClassifier(num_classes=len(class_map)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Trainer
    trainer = trainer(model, optimizer, criterion, scheduler,
                      train_loader, val_loader, device,
                      output_dir='outputs', resume_ckpt=args.resume)

    trainer.train(epochs=args.epochs)

if __name__ == "__main__":
    main()



