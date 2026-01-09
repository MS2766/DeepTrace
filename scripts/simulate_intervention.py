import torch
import numpy as np
from pathlib import Path
from src.gnn.models import HeteroGraphSAGE

device = "cpu"
data = torch.load("graphs/propagation_graph.pt", weights_only=False).to(device)

model = HeteroGraphSAGE(hidden_channels=128, metadata=data.metadata())
model.load_state_dict(torch.load("checkpoints/gnn_cpu.pt", map_location=device))
model.to(device)
model.eval()

print("Model & graph loaded successfully.")

with torch.no_grad():
    out_dict = model(data.x_dict, data.edge_index_dict)

    pos_edge_index = data['video', 'reshared_by', 'user'].edge_index
    src = out_dict['video'][pos_edge_index[0]]
    dst = out_dict['user'][pos_edge_index[1]]
    pred_probs = model.edge_classifier(torch.cat([src, dst], dim=-1)).sigmoid().squeeze().cpu().numpy()

print(f"Sample predicted reshare probabilities (first 10): {pred_probs[:10]}")

def independent_cascade(edge_index_vu, seeds, prob_dict, max_iter=50):
    infected_videos = set(seeds)
    newly_infected = set(seeds)
    for _ in range(max_iter):
        next_infected = set()
        for v in newly_infected:
            p = prob_dict.get(v, 0.1)
            neighbors = edge_index_vu[1][edge_index_vu[0] == v].tolist()
            for u in neighbors:
                if np.random.rand() < p:
                    next_infected.add(u)
        if not next_infected:
            break
        infected_videos.update(next_infected)
        newly_infected = next_infected
    return len(infected_videos)

# Improved seed selection: only fake videos
fake_videos = (data['video'].y == 1).nonzero(as_tuple=True)[0].tolist()
seeds = np.random.choice(fake_videos, size=min(20, len(fake_videos)), replace=False).tolist()

# Risk score = avg predicted prob × out-degree
risk_scores = []
for v in fake_videos:
    mask = pos_edge_index[0] == v
    avg_prob = pred_probs[mask].mean() if mask.any() else 0.05
    out_degree = (pos_edge_index[0] == v).sum().item()
    risk_scores.append((v, avg_prob * out_degree))

top_risky = [v for v, score in sorted(risk_scores, key=lambda x: x[1], reverse=True)[:5]]

# Baseline
baseline_size = independent_cascade(pos_edge_index, seeds, {}, max_iter=50)

# Intervention
intervened_seeds = [s for s in seeds if s not in top_risky]
intervened_size = independent_cascade(pos_edge_index, intervened_seeds, {}, max_iter=50)

reduction = baseline_size - intervened_size
reduction_pct = (reduction / baseline_size * 100) if baseline_size > 0 else 0

print(f"Baseline cascade size (infected users): {baseline_size}")
print(f"After removing top-5 risky fake videos: {intervened_size} infected")
print(f"Reduction: {reduction} users ({reduction_pct:.1f}%)")

# Save results
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
with open(results_dir / "intervention_results.txt", "w") as f:
    f.write(f"Baseline: {baseline_size}\n")
    f.write(f"After intervention: {intervened_size}\n")
    f.write(f"Reduction: {reduction} ({reduction_pct:.1f}%)\n")
print("Results saved to results/intervention_results.txt")