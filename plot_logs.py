# plot_logs.py
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt

LOG_CSV = Path("outputs/logs/metrics.csv")
TEST_JSON = Path("outputs/logs/test_metrics.json")
OUT_PNG = Path("outputs/logs/loss_curve.png")

epochs, train_loss, val_loss = [], [], []

with LOG_CSV.open("r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        val_loss.append(float(row["val_loss"]))

test_loss = None
test_acc = None
if TEST_JSON.exists():
    m = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    test_loss = float(m.get("loss", None)) if m.get("loss") is not None else None
    test_acc = float(m.get("acc", None)) if m.get("acc") is not None else None

plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, label="Train loss")
plt.plot(epochs, val_loss, label="Val loss")

if test_loss is not None:
    # Mark test loss at the final epoch position
    plt.scatter([epochs[-1]], [test_loss], s=60, label=f"Test loss (acc={test_acc:.3f})")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training / Validation Loss (and Test Loss marker)")
plt.legend()
plt.grid(True, alpha=0.3)
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"Saved {OUT_PNG}")
