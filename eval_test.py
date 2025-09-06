# eval_test.py
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets.plant_dataset import PlantDataset
from models.mobilevit import MobileViTClassifier
from utils.transforms import inference_transform

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/checkpoints/best.pth")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split",  default="test")
    ap.add_argument("--img-size", type=int, default=192)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")

    # Lock class mapping using train split
    train_root = Path(args.data_dir) / args.train_split
    classes = sorted([d.name for d in train_root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    class_map = {c:i for i,c in enumerate(classes)}
    num_classes = len(classes)

    tfm = inference_transform(img_size=args.img_size)
    test_ds = PlantDataset(args.data_dir, args.test_split, class_map=class_map, transform=tfm, strict=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=(device.type=="cuda"))

    model = MobileViTClassifier(image_size=(args.img_size,args.img_size), num_classes=num_classes).to(device)
    ckpt = torch.load(args.model, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt: state = ckpt["model_state"]
    elif isinstance(ckpt, dict): state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    else: state = ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    total_loss = 0.0; total = 0; correct = 0

    for x,y in test_loader:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()*x.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += x.size(0)
        for t,p in zip(y.view(-1), pred.view(-1)):
            conf[t.long(), p.long()] += 1

    test_loss = total_loss/max(1,total)
    test_acc  = correct/max(1,total)

    eps=1e-12
    tp = conf.diag().float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp
    prec = tp/(tp+fp+eps); rec = tp/(tp+fn+eps)
    f1 = 2*prec*rec/(prec+rec+eps)
    macro_p, macro_r, macro_f1 = float(prec.mean()), float(rec.mean()), float(f1.mean())

    out = Path("outputs/logs"); out.mkdir(parents=True, exist_ok=True)
    (out/"confusion_matrix.csv").write_text(
        "\n".join([",".join(map(str,conf[i].tolist())) for i in range(num_classes)]), encoding="utf-8")
    with (out/"test_per_class.csv").open("w", encoding="utf-8") as f:
        f.write("class,precision,recall,f1,support\n")
        for i,c in enumerate(classes):
            f.write(f"{c},{float(prec[i])},{float(rec[i])},{float(f1[i])},{int(conf[i].sum())}\n")
    metrics={"loss":test_loss,"acc":test_acc,"precision_macro":macro_p,"recall_macro":macro_r,"f1_macro":macro_f1}
    (out/"test_metrics.json").write_text(json.dumps(metrics,indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__=="__main__":
    main()
