import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        
        conv_dict = {}
        for edge_type in metadata[1]:
            conv_dict[edge_type] = SAGEConv((-1, -1), hidden_channels, aggr="mean")
        
        self.convs = HeteroConv(conv_dict, aggr="sum")
        
        self.lin_video = Linear(-1, hidden_channels)
        self.lin_user = Linear(-1, hidden_channels)
        
        self.edge_classifier = torch.nn.Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.convs(x_dict, edge_index_dict)
        
        x_dict['video'] = self.lin_video(x_dict['video'])
        x_dict['user'] = self.lin_user(x_dict['user'])
        
        return x_dict

    def predict_reshare(self, x_dict, edge_index_vu):
        src = x_dict['video'][edge_index_vu[0]]
        dst = x_dict['user'][edge_index_vu[1]]
        edge_emb = torch.cat([src, dst], dim=-1)
        logits = self.edge_classifier(edge_emb).squeeze(-1)
        return torch.sigmoid(logits)