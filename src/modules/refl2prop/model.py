# refl2prop/model.py
import torch
import torch.nn as nn
from typing import Dict

class InversionNet(nn.Module):
    def __init__(self, input_size: int = 17, output_size: int = 4, hidden_dim: int = 512):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hidden_dim // 2),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hidden_dim // 4),
            
            nn.Linear(hidden_dim // 4, 64),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            
            nn.Linear(64, output_size),
            nn.Tanh() # Normalizes outputs to [-1, 1] range
        )

    def forward(self, x):
        return self.network(x)

class NormalizationWrapper(nn.Module):
    """
    Wraps the InversionNet to handle normalization internally.
    Stats are registered as buffers, so they save/load automatically with torch.save().
    """
    def __init__(self, model: nn.Module, input_stats: Dict[str, list], output_stats: Dict[str, list]):
        super().__init__()
        self.model = model
        
        # Register constants as buffers (non-trainable tensors)
        self.register_buffer('in_min', torch.tensor(input_stats['min'], dtype=torch.float32))
        self.register_buffer('in_max', torch.tensor(input_stats['max'], dtype=torch.float32))
        
        self.register_buffer('out_min', torch.tensor(output_stats['min'], dtype=torch.float32))
        self.register_buffer('out_max', torch.tensor(output_stats['max'], dtype=torch.float32))

    def normalize_input(self, x):
        # Scale to [-1, 1]
        # Epsilon to avoid div by zero if max == min
        denom = (self.in_max - self.in_min)
        denom[denom == 0] = 1.0 
        return 2 * (x - self.in_min) / denom - 1

    def denormalize_output(self, y_norm):
        # Scale back from [-1, 1] to physical units
        return (y_norm + 1) / 2 * (self.out_max - self.out_min) + self.out_min

    def forward(self, x):
        """
        Inference Mode: Takes raw physical inputs -> Returns raw physical outputs.
        """
        x_norm = self.normalize_input(x)
        y_norm = self.model(x_norm)
        return self.denormalize_output(y_norm)
    
    @property
    def ranges(self):
        # Return dictionary of input/output ranges
        in_ranges = (list(zip(self.in_min.cpu().numpy(), self.in_max.cpu().numpy())))
        out_ranges = (list(zip(self.out_min.cpu().numpy(), self.out_max.cpu().numpy())))
        return {'input_ranges': in_ranges, 'output_ranges': out_ranges}