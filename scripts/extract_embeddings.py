import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from src.data.frame_dataset import DeepfakeFrameDataset
from src.models.efficientnet import EfficientNetB0Video

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "checkpoints/efficientnet_b0_tuned_best.pt"
SAVE_DIR = Path("embeddings")
SAVE_DIR.mkdir(exist_ok=True)

def main():
    dataset = DeepfakeFrameDataset(
        root_dir="data_frames",
        num_frames=8,
        image_size=224,
        augment=False  # inference mode
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = EfficientNetB0Video()
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_emb = []
    all_lbl = []
    all_ids = []

    print("Extracting embeddings...")
    with torch.no_grad():
        for frames, labels in tqdm(loader):
            frames = frames.to(DEVICE)
            feats = model.extract_features(frames)  # (B, 1280)
            all_emb.append(feats.cpu().numpy())
            all_lbl.append(labels.numpy())
            # IDs can be added later if needed

    np.save(SAVE_DIR / "video_embeddings.npy", np.concatenate(all_emb))
    np.save(SAVE_DIR / "labels.npy", np.concatenate(all_lbl))

    print(f"Done! Saved {len(all_emb)} embeddings (shape: {np.concatenate(all_emb).shape})")

if __name__ == "__main__":
    main()