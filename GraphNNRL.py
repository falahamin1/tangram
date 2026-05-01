import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class GCNLayer(nn.Module):
    """A standard Graph Convolutional Layer."""
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: [Batch, Nodes, Feats]
        # adj: [Batch, Nodes, Nodes]
        
        # 1. Message Passing: Sum features from neighbors
        # (Batch matrix multiplication)
        support = torch.bmm(adj, x) 
        
        # 2. Update: Linear projection + Non-linearity
        return F.relu(self.projection(support))

class GNNActorCritic(nn.Module):
    def __init__(self, node_dim=3, hidden_dim=128, num_actions=24):
        super().__init__()
        
        # 1. GNN Encoder
        # Processes the constraints and their connectivity
        self.gcn1 = GCNLayer(node_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        
        # 2. Refinement Layer (Shared)
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 3. Policy Head (Actor)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # 4. Value Head (Critic)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, h_rep, adj):
        """
        Inputs:
            h_rep: [Batch, 6, 5, 3] -> (6 pieces, 5 constraints, 3 params)
            adj:   [Batch, 6, 5, 5] -> (6 pieces, 5x5 adjacency)
        """
        batch_size = h_rep.shape[0]
        
        # --- A. Pre-processing: Flatten pieces into one big graph ---
        # We treat all 30 nodes (6 pieces * 5 constraints) as one graph 
        # with disjoint components.
        
        # x: [Batch, 30, 3]
        x = h_rep.view(batch_size, -1, 3) 
        
        # To make adj [Batch, 30, 30], we place the 6 adj matrices 
        # along the diagonal of a 30x30 block-diagonal matrix.
        big_adj = torch.zeros(batch_size, 30, 30, device=h_rep.device)
        for i in range(6):
            big_adj[:, i*5:(i+1)*5, i*5:(i+1)*5] = adj[:, i, :, :]
            
        # Add self-loops to the adjacency matrix (A + I)
        # This ensures nodes consider their own features during update
        big_adj += torch.eye(30, device=h_rep.device).unsqueeze(0)

        # --- B. GNN Layers ---
        h = self.gcn1(x, big_adj)      # [Batch, 30, 128]
        h = self.gcn2(h, big_adj)      # [Batch, 30, 128]
        
        # --- C. Readout (Global Pooling) ---
        # We aggregate all 30 nodes into one representation.
        # Max pooling is often better for geometry than Sum pooling.
        global_pool = torch.max(h, dim=1)[0] # [Batch, 128]
        
        latent = self.rho(global_pool)
        
        # --- D. Heads ---
        logits = self.actor(latent)
        value = self.critic(latent)
        
        return logits, value