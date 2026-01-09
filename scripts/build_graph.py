import torch
from torch_geometric.data import HeteroData
import numpy as np
from pathlib import Path

SAVE_DIR = Path("graphs")
SAVE_DIR.mkdir(exist_ok=True)

def build_hetero_graph():
    data = HeteroData()

    # Load embeddings & labels
    emb_path = Path("embeddings/video_embeddings.npy")
    lbl_path = Path("embeddings/labels.npy")
    
    video_x = torch.from_numpy(np.load(emb_path)).float()
    video_y = torch.from_numpy(np.load(lbl_path)).long()

    num_videos = video_x.size(0)
    num_users = 2000  # synthetic users - adjust as needed

    # Video nodes
    data['video'].x = video_x
    data['video'].y = video_y  # 0=real, 1=fake

    # User nodes (dummy features)
    data['user'].x = torch.randn(num_users, 64)

    # 1. User → User: follow edges (sparse random graph)
    edge_index_uu = []
    for i in range(num_users):
        for j in range(num_users):
            if np.random.rand() < 0.005:  # ~0.5% connection probability
                edge_index_uu.append([i, j])
    if edge_index_uu:
        data['user', 'follows', 'user'].edge_index = torch.tensor(edge_index_uu, dtype=torch.long).t()

    # 2. User → Video: authorship (random assignment)
    edge_index_uv = []
    for v in range(num_videos):
        u = np.random.randint(0, num_users)
        edge_index_uv.append([u, v])
    data['user', 'authors', 'video'].edge_index = torch.tensor(edge_index_uv, dtype=torch.long).t()

    # 3. Video → User: reshare edges (higher probability for fake videos)
    edge_index_vu = []
    for v in range(num_videos):
        label = video_y[v].item()
        prob = 0.02 if label == 1 else 0.005  # fake videos spread more
        for u in range(num_users):
            if np.random.rand() < prob:
                edge_index_vu.append([v, u])
    if edge_index_vu:
        data['video', 'reshared_by', 'user'].edge_index = torch.tensor(edge_index_vu, dtype=torch.long).t()

    torch.save(data, SAVE_DIR / "propagation_graph.pt")
    print(f"Graph saved: {data}")
    print(f"  - Video nodes: {num_videos}")
    print(f"  - User nodes: {num_users}")
    print(f"  - Follow edges: {data['user', 'follows', 'user'].edge_index.size(1) if 'follows' in data.edge_types else 0}")
    print(f"  - Authorship edges: {data['user', 'authors', 'video'].edge_index.size(1)}")
    print(f"  - Reshare edges: {data['video', 'reshared_by', 'user'].edge_index.size(1) if 'reshared_by' in data.edge_types else 0}")

if __name__ == "__main__":
    build_hetero_graph()