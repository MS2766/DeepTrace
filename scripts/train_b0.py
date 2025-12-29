import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from pathlib import Path

from src.data.frame_dataset import DeepfakeFrameDataset
from src.models.efficientnet import EfficientNetB0Video


def main():
    # ---------------- CONFIG ----------------
    DATA_ROOT = "data_frames"
    NUM_FRAMES = 8
    IMAGE_SIZE = 224

    BATCH_SIZE = 6          # ✅ safe for RTX 4050 (6 GB)
    EPOCHS = 15
    LR = 1e-4
    VAL_SPLIT = 0.2
    NUM_WORKERS = 0         # safe now (image loading, not video)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()

    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # ---------------- DATA ----------------
    dataset = DeepfakeFrameDataset(
        root_dir=DATA_ROOT,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
    )

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=False
    )

    # ---------------- MODEL ----------------
    model = EfficientNetB0Video().to(DEVICE)

    # ✅ channels-last memory format (VRAM optimization)
    model = model.to(memory_format=torch.channels_last)

    # Handle class imbalance
    pos_weight = torch.tensor([4.0], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler("cuda") if use_amp else None

    best_acc = 0.0

    # ---------------- TRAINING ----------------
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for frames, labels in train_loader:
            frames = frames.to(DEVICE)
            labels = labels.float().to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast("cuda"):
                    logits = model(frames)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(frames)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        correct = total = 0

        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(DEVICE)
                labels = labels.float().to(DEVICE)

                logits = model(frames)
                loss = criterion(logits, labels)

                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds == labels.long()).sum().item()
                total += labels.size(0)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_acc = correct / total

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        # Save best model only
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "efficientnet_b0_best.pt"
            )

        # ✅ reduce fragmentation
        torch.cuda.empty_cache()

    print(f"Training complete. Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
