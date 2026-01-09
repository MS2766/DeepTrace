import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from src.gnn.models import HeteroGraphSAGE

# Load the full graph (CPU)
data = torch.load("graphs/propagation_graph.pt", weights_only=False).to("cpu")

# Create positive & negative edges for reshare prediction (video → user)
pos_edge_index = data['video', 'reshared_by', 'user'].edge_index
num_pos = pos_edge_index.size(1)

neg_edge_index = negative_sampling(
    edge_index=pos_edge_index,
    num_nodes=(data['video'].num_nodes, data['user'].num_nodes),
    num_neg_samples=num_pos
)

# Combine into training edges
edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
edge_label = torch.cat([
    torch.ones(num_pos, dtype=torch.float),
    torch.zeros(num_pos, dtype=torch.float)
])

model = HeteroGraphSAGE(hidden_channels=128, metadata=data.metadata())
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.BCEWithLogitsLoss()

print("Starting full-batch GNN training on CPU...")

for epoch in range(50):  # 50 epochs — enough for CPU
    model.train()
    optimizer.zero_grad()

    # Forward pass on full graph
    out_dict = model(data.x_dict, data.edge_index_dict)

    # Get predictions for all training edges
    src = out_dict['video'][edge_label_index[0]]
    dst = out_dict['user'][edge_label_index[1]]
    pred = model.edge_classifier(torch.cat([src, dst], dim=-1)).squeeze(-1)

    loss = criterion(pred, edge_label)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "checkpoints/gnn_cpu.pt")
print("GNN training complete! Model saved to checkpoints/gnn_cpu.pt")