import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from pathlib import Path

from src.data.frame_dataset import DeepfakeFrameDataset
from src.models.efficientnet_b4 import EfficientNetB4Video


def main():
    # ---------------- CONFIG ----------------
    DATA_ROOT = "data_frames"
    NUM_FRAMES = 8
    IMAGE_SIZE = 380   # constrained resolution (GPU limit)

    BATCH_SIZE = 3     # physical batch
    ACCUM_STEPS = 2    # gradient accumulation → effective batch = 6
    EPOCHS = 20
    LR = 1e-4
    VAL_SPLIT = 0.2
    NUM_WORKERS = 0    # Windows-stable

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
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

    # ---------------- MODEL ----------------
    model = EfficientNetB4Video().to(DEVICE)
    model = model.to(memory_format=torch.channels_last)

    pos_weight = torch.tensor([4.0], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ✅ LR scheduler (VERY important)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    scaler = GradScaler("cuda") if use_amp else None

    best_acc = 0.0

    # ---------------- TRAIN ----------------
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (frames, labels) in enumerate(train_loader):
            frames = frames.to(DEVICE)
            labels = labels.float().to(DEVICE)

            if use_amp:
                with autocast("cuda"):
                    logits = model(frames)
                    loss = criterion(logits, labels)
                    loss = loss / ACCUM_STEPS
                scaler.scale(loss).backward()
            else:
                logits = model(frames)
                loss = criterion(logits, labels)
                loss = loss / ACCUM_STEPS
                loss.backward()

            if (step + 1) % ACCUM_STEPS == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * ACCUM_STEPS

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

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "efficientnet_b4_best.pt"
            )

        scheduler.step()
        torch.cuda.empty_cache()

    print(f"Training complete. Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
