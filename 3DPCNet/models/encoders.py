import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder for 3D pose features
    """
    def __init__(self, input_dim, hidden_dim=512, output_dim=256, dropout=0.1, num_layers=4):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
            hidden_dim = hidden_dim // 2  # Gradually reduce dimension
        
        # Final layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights like 6DRepNet"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) flattened 3D pose coordinates
        Returns:
            features: (batch, output_dim)
        """
        return self.encoder(x)


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder for 3D poses
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=256, 
                 num_joints=17, dropout=0.1, num_layers=3):
        super().__init__()
        
        self.num_joints = num_joints
        self.num_layers = num_layers
        
        # Define skeleton connectivity (custom 17-joint format)
        self.register_buffer('edge_index', self._get_skeleton_edges())
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        current_dim = input_dim
        
        for i in range(num_layers):
            self.gcn_layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        
        self.dropout = nn.Dropout(dropout)
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.Linear(num_joints * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
    def _get_skeleton_edges(self):
        """Define custom 17-joint skeleton connections"""
        connections = [
            # Spine and head
            (0, 7), (7, 8), (8, 9), (9, 10),
            # Left leg
            (0, 1), (1, 2), (2, 3),
            # Right leg
            (0, 4), (4, 5), (5, 6),
            # Right arm from neck
            (8, 11), (11, 12), (12, 13),
            # Left arm from neck
            (8, 14), (14, 15), (15, 16)
        ]
        
        edge_index = torch.tensor(connections).t().contiguous()
        # Make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return edge_index
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _gcn_forward(self, x, edge_index, layer):
        """Simple GCN message passing"""
        row, col = edge_index
        
        # Aggregate neighbors
        out = torch.zeros_like(x)
        for i in range(x.size(0)):
            neighbors = col[row == i]
            if len(neighbors) > 0:
                out[i] = torch.mean(x[neighbors], dim=0)
            else:
                out[i] = x[i]
        
        # Apply linear transformation
        out = layer(out)
        return out
    
    def forward(self, x):
        """
        Args:
            x: (batch, joints, 3) 3D pose coordinates
        Returns:
            features: (batch, output_dim)
        """
        batch_size, num_joints, input_dim = x.shape
        
        # Process each graph in the batch
        outputs = []
        for b in range(batch_size):
            graph_x = x[b]  # (joints, input_dim)
            # Apply GCN layers
            for i, layer in enumerate(self.gcn_layers):
                graph_x = self._gcn_forward(graph_x, self.edge_index, layer)
                if i < len(self.gcn_layers) - 1:  # No activation on last layer
                    graph_x = F.relu(graph_x)
                    graph_x = self.dropout(graph_x)
            outputs.append(graph_x)
        # Stack batch
        x = torch.stack(outputs, dim=0)  # (batch, joints, hidden_dim)
        # Global pooling
        x = x.reshape(batch_size, -1)  # Flatten
        x = self.global_pool(x)
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for 3D pose sequences
    """
    def __init__(self, input_dim=3, hidden_dim=256, num_heads=8, 
                 num_layers=4, num_joints=17, dropout=0.1, output_dim=256):
        super().__init__()
        
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        # Joint embedding
        self.joint_embed = nn.Linear(input_dim, hidden_dim)
        # Positional encoding for joints (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, num_joints, hidden_dim) * 0.02)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.Linear(num_joints * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, joints, 3) 3D pose coordinates
        Returns:
            features: (batch, output_dim)
        """
        batch_size, num_joints, _ = x.shape
        
        # Joint embedding
        x = self.joint_embed(x)  # (batch, joints, hidden_dim)
        # Add positional encoding
        x = x + self.pos_embed
        # Transformer encoding
        x = self.transformer(x)  # (batch, joints, hidden_dim)
        # Global pooling
        x = x.view(batch_size, -1)
        x = self.global_pool(x)
        return x