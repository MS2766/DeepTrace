from src.data.dataset import DeepfakeVideoDataset
from torch.utils.data import DataLoader

def main():
    dataset = DeepfakeVideoDataset(
        root_dir="data",
        num_frames=8,
        image_size=224,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
    )

    frames, labels = next(iter(loader))
    print(frames.shape, labels)

if __name__ == "__main__":
    main()
