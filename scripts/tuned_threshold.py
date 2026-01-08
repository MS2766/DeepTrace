import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score

from src.data.frame_dataset import DeepfakeFrameDataset
from src.models.efficientnet import EfficientNetB0Video


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VAL_SPLIT = 0.2

    dataset = DeepfakeFrameDataset(
        root_dir="data_frames",
        num_frames=8,
        image_size=224,
        augment=False,
    )

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    _, val_ds = random_split(dataset, [train_size, val_size])

    loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    model = EfficientNetB0Video().to(DEVICE)
    model.load_state_dict(
        torch.load(
            "checkpoints/efficientnet_b0_tuned_best.pt",
            map_location=DEVICE,
            weights_only=True
        )
    )
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(DEVICE)
            logits = model(frames)

            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    probs = 1 / (1 + np.exp(-logits))

    best_acc = 0
    best_thresh = 0.5

    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs > t).astype(int)
        acc = accuracy_score(labels, preds)

        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    print(f"Best Threshold (VAL): {best_thresh:.2f}")
    print(f"Best Val Accuracy   : {best_acc:.4f}")


if __name__ == "__main__":
    main()
