from src.data.frame_dataset import DeepfakeFrameDataset
ds = DeepfakeFrameDataset("data_frames", num_frames=8)
print(len(ds))
x, y = ds[0]
print(x.shape, y)